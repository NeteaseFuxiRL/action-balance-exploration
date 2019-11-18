import numpy as np
import argparse
import os
import datetime
import pandas as pd

import config
from utils.replay_buffer import DictReplayBuffer

from envs.grid_world import GridWorldEnv, DelayRewardWrapper, StochasticGrid, StochasticGridV2


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)

    import random
    random.seed(myseed)


def init_config(args):
    if args.save_name == 'default':
        args.save_name = datetime.datetime.now().strftime("logs/default-%Y-%m-%d-%H-%M-%S-%f")

    for key, value in vars(args).items():
        if value is not None:
            config.DEFAULT_PARAMS[key] = value

    print(config.DEFAULT_PARAMS)


class ReplayBufferWrapper(object):
    def __init__(self, buffer, gamma=0.99, lam=0.95, lr=0.1):
        self.buffer = buffer
        self.gamma = gamma
        self.lam = lam
        self.lr = lr

        self.obs = []
        self.pis = []
        self.rewards = []
        self.obs_tp1 = []

        self.size = self.buffer.size
        self.sample = self.buffer.sample

        self.state_values = {}
        self.int_rew = {}

    def add(self, ob, pi, reward, obs_tp1=None):
        self.obs.append(ob)
        self.pis.append(pi)
        self.rewards.append(reward)
        self.obs_tp1.append(obs_tp1)

    def get_state_value(self, obs, default=0.):
        return self.state_values.get(self.buffer.obs2key(obs), default)

    def get_int_reward(self, obs, default=0.):
        return self.int_rew.get(self.buffer.obs2key(obs), default)

    def update_state_value(self, ob, delta):
        obs_key = self.buffer.obs2key(ob)
        if obs_key not in self.state_values:
            self.state_values[obs_key] = 0.
        self.state_values[obs_key] += self.lr * delta

    def make_sample(self):
        """
        Make each sample's reward to sum(rewards), and add to replay buffer.
        :return: None
        """
        # 1, calculate reward sum
        reward_sum = np.sum(self.rewards)

        mb_advs_int = np.zeros_like(self.rewards, dtype=np.float)
        deltas = np.zeros_like(self.rewards, dtype=np.float)
        lastgaelam = 0
        max_step = len(self.rewards)
        for t in reversed(range(max_step)):
            if t == max_step - 1:
                nextnonterminal = 0.
                nextvalues = 0.
            else:
                nextnonterminal = 1.0
                # replace values with discount rewards
                nextvalues = self.get_state_value(self.obs_tp1[t+1])
            delta = self.get_state_count_reward(
                self.obs_tp1[t]) + self.gamma * nextvalues * nextnonterminal - self.get_state_value(self.obs_tp1[t])
            deltas[t] = delta
            mb_advs_int[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

            # update intrinsic reward
            ob_key = self.buffer.obs2key(self.obs[t])
            if ob_key not in self.int_rew:
                self.int_rew[ob_key] = np.zeros(shape=len(self.pis[0]))
            self.int_rew[ob_key][np.argmax(self.pis[t])] = mb_advs_int[t]

        # 2, modify sample's reward and add to replay buffer
        for i in range(len(self.obs)):
            self.update_state_value(ob=self.obs_tp1[i], delta=deltas[i])
            self.buffer.add(self.obs[i], self.pis[i], reward_sum, ns_int=self.int_rew[self.buffer.obs2key(self.obs[i])])

        # 3, clear cache
        self.clear_cache()

    def clear_cache(self):
        self.obs = []
        self.pis = []
        self.rewards = []
        self.obs_tp1 = []

    def clear(self):
        self.buffer.clear()

    def get_pi(self, obs):
        data = self.buffer.get_data(obs)  # [obs_t, total_scores, total_counts, pi, Q]
        if data is None:
            pi = None
        else:
            pi = data[3]
        return pi

    def get_state_count_reward(self, obs):
        return self.buffer.get_state_count_reward(obs)


def SimpleGridWorld(seed=None, state_mode='mask', end_point=None):
    '''
    '''

    class MyGridWorld(GridWorldEnv):
        def __init__(self, state_mode='mask', *args, **kwargs):
            self.state_mode = state_mode

            super(MyGridWorld, self).__init__(*args, **kwargs)

        def _get_obs(self):

            return self.xy2state(x=self.current_x, y=self.current_y)

        def xy2state(self, x, y):
            if self.state_mode == 'mask':
                state = np.zeros((self.n_width, self.n_height))
                state[x, y] = 1.
                state = state.reshape(-1)
            elif self.state_mode == 'one-hot':
                state = np.zeros(self.n_width + self.n_height)
                state[x] = 1
                state[self.n_width + y] = 1
            elif self.state_mode == 'scale':
                state = np.array([x, y])

            return state

        def state2xy(self, state):
            if self.state_mode == 'mask':
                x, y = np.where(np.reshape(state, (self.n_width, self.n_height)) == 1)
                x, y = x[0], y[0]
            elif self.state_mode == 'one-hot':
                x, y = np.where(state == 1)[0]
                y -= self.n_width
            elif self.state_mode == 'scale':
                x, y = state

            return x, y

    n = 40
    n_width = n
    n_heigh = n
    end_reward = 1
    env = MyGridWorld(n_width=n_width,
                      n_height=n_heigh,
                      action_num=4,
                      u_size=20,
                      # default_reward=-0.005,
                      default_reward=0,
                      default_type=0,
                      max_step=200,
                      windy=False,
                      state_mode=state_mode)
    # env.action_space = spaces.Discrete(8)
    env.start = (0, 0)
    # env.ends = [(n_width - 1, n_heigh - 1)]
    if end_point is not None:
        env.ends = [end_point]
        env.rewards = [(*x, end_reward) for x in env.ends]
    else:
        env.ends = []
        env.rewards = []
    env.refresh_setting()

    env.print_env_infos()
    env = DelayRewardWrapper(env=env)

    return env


def stochastic_grid(seed=None, state_mode='mask'):
    class MyGridWorld(StochasticGridV2):
        def __init__(self, state_mode='mask', *args, **kwargs):
            self.state_mode = state_mode

            super(MyGridWorld, self).__init__(*args, **kwargs)

        def _get_obs(self):

            return self.xy2state(x=self.current_x, y=self.current_y)

        def xy2state(self, x, y):
            if self.state_mode == 'mask':
                state = np.zeros((self.n_width, self.n_height))
                state[x, y] = 1.
                state = state.reshape(-1)
            elif self.state_mode == 'one-hot':
                state = np.zeros(self.n_width + self.n_height)
                state[x] = 1
                state[self.n_width + y] = 1

            return state

        def state2xy(self, state):
            if self.state_mode == 'mask':
                x, y = np.where(np.reshape(state, (self.n_width, self.n_height)) == 1)
                x, y = x[0], y[0]
            elif self.state_mode == 'one-hot':
                x, y = np.where(state == 1)[0]
                y -= self.n_width

            return x, y

    env = MyGridWorld(n_height=20, n_width=20,
                      wall_start_x=3,
                      action_num=4,
                      end_reward=1,
                      teleport_prob=0.8,
                      seed=seed,
                      # teleport_point=(0, 0),
                      default_reward=-0.005,
                      max_step=200,
                      state_mode=state_mode,
                      teleport_point=(3, 0)
                      )

    env.print_env_infos()
    env = DelayRewardWrapper(env=env)

    return env


class BufferLogger(object):
    def __init__(self, buffer_for_logger, save_freq=1000, save_path=None):
        self.save_freq = save_freq
        self.save_path = save_path

        # self.care_keys = ['reward', 'state_count_reward']
        self.care_keys = ['state_count_reward']
        # self.care_keys = ['visit_num']
        self.data_dict = {'visit_num': [],
                          'seen_rate': []}
        for key in self.care_keys:
            self.data_dict[key] = []

        self.buffer = buffer_for_logger

        self.count = 0

    def add(self, ob, pi, reward, obs_tp1=None):
        self.buffer.add(ob, pi, reward)

    def get_visit_num(self, env):
        width, height = env.n_width, env.n_height
        visit_num_array = np.zeros(shape=(height, width))

        for x in range(width):
            for y in range(height):
                obs = env.xy2state(x, y)
                s = self.buffer._storage.get(bytes(obs), None)
                if s is not None:
                    visit_num_array[y, x] = sum(s[2])

        return visit_num_array

    def get_buffer_info(self, dict_replay_buffer, env):
        width, height = env.n_width, env.n_height
        state_count_reward = np.zeros(shape=(height, width))

        for x in range(width):
            for y in range(height):
                obs = env.xy2state(x, y)
                state_count_reward[y, x] = np.sum(dict_replay_buffer.get_int_reward(obs))

        # np.set_printoptions(precision=5, suppress=True, linewidth=9999999)
        # print('visit_num')
        # print(visit_num_array)

        return dict(zip(self.care_keys, [state_count_reward]))

    def update(self, env, dict_replay_buffer):
        """
        Update logger data, add one frame.
        :param dict_replay_buffer:
        :param env:
        :return:
        """
        visit_nums = self.get_visit_num(env)
        self.data_dict['visit_num'].append(pd.DataFrame(visit_nums))
        self.data_dict['seen_rate'].append((visit_nums != 0).sum() / visit_nums.size)

        new_data_dict = self.get_buffer_info(dict_replay_buffer, env)
        for key in self.care_keys:
            self.data_dict[key].append(pd.DataFrame(new_data_dict[key]))

        # self.count += 1
        #
        # if self.count % self.save_freq == 0:
        #     self.save()

    def save(self):
        if self.save_path is None:
            raise ValueError('save path must not be None')
        os.makedirs(self.save_path, exist_ok=True)
        try:
            # for key in self.care_keys:
            for key in self.data_dict.keys():
                if key == 'seen_rate':
                    df = pd.DataFrame({'seen_rate': self.data_dict['seen_rate'],
                                       'step': range(len(self.data_dict['seen_rate']))})
                else:
                    df = pd.concat(self.data_dict[key], keys=range(len(self.data_dict[key])))
                df.to_pickle(os.path.join(self.save_path, '{}.pkl'.format(key)))
                print('save {} to {}'.format(key, self.save_path))
        except Exception as e:
            print('Save buffer info failed! {}'.format(e))


def train(args, end_point=None):
    # env = gym.make(args.env)
    env = SimpleGridWorld(state_mode='scale', end_point=end_point)
    # env = stochastic_grid(seed=args.seed)
    obs = env.reset()
    state_shape = env.observation_space.shape
    action_num = env.action_space.n
    max_step = env.unwrapped.max_step

    default_params = config.DEFAULT_PARAMS

    print_freq = default_params['print_freq']
    save_freq = default_params['save_freq']
    update_buffer_freq = 10

    stochastic = args.stochastic
    exploration_prob = args.exploration_prob
    action_list = list(range(action_num))

    # game_num = 0
    same_prob_pi = np.ones(shape=(action_num,)) * (1 / action_num)
    game_steps = []
    for run_time in range(args.run_times):
        game_num = 0
        t = 0
        point = None
        care_step = 50
        episode_lengths = []
        episode_rewards = []

        def get_buffer():
            replay_buffer = DictReplayBuffer(size=default_params['buffer_size'],
                                             action_num=action_num,
                                             action_count_coef=default_params['action_count_coef'],
                                             q_coef=default_params['q_coef'],
                                             state_reward_coef=default_params['state_reward_coef'],
                                             only_positive=default_params['only_positive'],
                                             global_count_coef=default_params['global_count_coef'])
            return replay_buffer

        replay_buffer = ReplayBufferWrapper(get_buffer())
        buffer_logger = BufferLogger(buffer_for_logger=get_buffer(),
                                     save_path=os.path.join(args.save_name, str(run_time)), save_freq=10)

        pi_func = {'random': lambda x: same_prob_pi,
                   'buffer': replay_buffer.get_pi}[args.pi_func]

        while game_num < default_params['total_times']:
            pi = pi_func(obs)
            if pi is None:
                pi = same_prob_pi.copy()

            if exploration_prob is not None and np.random.random() < exploration_prob:
                # exploration with same probability
                pi = same_prob_pi.copy()
                action = np.random.choice(action_list, p=pi)

            elif stochastic:
                action = np.random.choice(action_list, p=pi)
            else:
                if np.all(pi == 1./len(pi)):
                    action = np.random.choice(action_list, p=pi)
                else:
                    action = np.argmax(pi)

            new_obs, reward, done, info = env.step(action)
            t += 1

            if t == care_step:
                point = new_obs

            # print(obs, pi, reward, done)
            # print(pi, action, [env.env.current_x, env.env.current_y], reward, done,
            #       replay_buffer.buffer._storage.get(bytes(obs), [None, None, None, None])[1:4])

            pi = np.zeros(action_num)
            pi[action] = 1.
            replay_buffer.add(obs, pi, reward, new_obs)
            buffer_logger.add(obs, pi, reward, new_obs)
            obs = new_obs.copy()

            if t % update_buffer_freq == 0:
                if args.log_run_index == -1 or args.log_run_index == run_time:
                    # buffer_logger.update(dict_replay_buffer=replay_buffer.buffer, env=env.unwrapped)
                    buffer_logger.update(dict_replay_buffer=replay_buffer, env=env.unwrapped)
            # env.render()

            if t % args.make_sample_freq == 0:
                replay_buffer.make_sample()

            episode_rewards.append(reward)
            if done:
                # buffer_logger.update(dict_replay_buffer=replay_buffer.buffer, env=env.unwrapped)
                obs = env.reset()
                episode_lengths.append(len(episode_rewards))

                if episode_lengths[-1] < max_step:
                    if args.log_run_index == -1 or args.log_run_index == run_time:
                        # buffer_logger.update(dict_replay_buffer=replay_buffer.buffer, env=env.unwrapped)
                        buffer_logger.update(dict_replay_buffer=replay_buffer, env=env.unwrapped)
                    print('first reach goal at game {}'.format(game_num))
                    print('step_info:{}\t{}\t{}\t{}'.format(run_time, care_step, point, t))
                    break

                episode_rewards = []
                # replay_buffer.make_sample()
                game_num += 1

        game_steps.append(sum(episode_lengths))
        if args.log_run_index == -1 or args.log_run_index == run_time:
            buffer_logger.save()
        print('mean_episode_length:{}, reach_goal:{}/{}'.format(
            np.mean(episode_lengths), sum(np.array(episode_lengths) < max_step), len(episode_lengths)))
    mean_steps = np.mean(game_steps)
    print("total mean game length {}\t{}".format(mean_steps, game_steps))
    return mean_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training args')
    parser.add_argument('--total_times', type=int, default=1000)
    parser.add_argument('--env', type=str, default='CartPole-v0', help="game env name.")
    # parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4', help="game env name.")
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)

    parser.add_argument('--save_name', '--sn', type=str, default='default', help="Save name.")

    parser.add_argument('--test', '--t', action="store_true", help="Test.")

    parser.add_argument('--exploration_prob', '--ep', type=float, default=0, help="exploration probability")
    parser.add_argument('--action_count_coef', '--acc', type=float, default=1, help="action_count_coef")
    parser.add_argument('--q_coef', '--qc', type=float, default=0., help="q_coef")
    parser.add_argument('--state_reward_coef', '--src', type=float, default=0.1, help="state_reward_coef")
    parser.add_argument('--global_count_coef', '--gcc', type=float, default=0, help="global_count_coef")

    parser.add_argument('--pi_func', '--pf', type=str, default='buffer', choices=['random', 'buffer'], help="pi get mode")

    parser.add_argument('--log_run_index', help='', type=int, default=None)
    parser.add_argument('--run_times', type=int, default=100)
    parser.add_argument('--ends', type=str, default=None)
    parser.add_argument('--make_sample_freq', '--msf', type=int, default=8)

    parser.add_argument('--stochastic', type=int, default=1)

    # args, unknown = parser.parse_known_args()
    args = parser.parse_args()
    print('addition_params', args)
    init_config(args)

    results = []
    n = 40
    # default_end_points = [(0, n//2), (n//2, 0), (n//2, n-1), (n-1, n//2), (n-1, n-1)]
    default_end_points = [(0, n//2), (n//2, 0), (n//4, n//2), (n//2.5, n//2.5), (n//2, n//4)]
    default_end_points = [(int(x[0]), int(x[1])) for x in default_end_points]

    if args.ends is None:
        args.ends = []
    elif args.ends == 'all':
        args.ends = list(range(len(default_end_points)))
    else:
        args.ends = [int(x) for x in args.ends.split(' ')]

    if args.ends:
        ends_str = '\t'.join([str(default_end_points[i]) for i in args.ends])
        for i in args.ends:
            set_global_seeds(args.seed)
            results.append(train(args, end_point=default_end_points[i]))
    else:
        ends_str = ""
        set_global_seeds(args.seed)
        results.append(train(args, end_point=None))

    # print log
    print('ends:{}'.format(ends_str))
    print('\t'.join([str(x) for x in results]))