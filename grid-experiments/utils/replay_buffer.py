import numpy as np
import random
from collections import OrderedDict

# from utils.segment_tree import SumSegmentTree, MinSegmentTree
# from utils.math_util import normalize
# from utils.logger import logger

from .segment_tree import SumSegmentTree, MinSegmentTree
from .math_util import normalize, normalize_v3
from .logger import logger


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def is_full(self):
        return len(self._storage) == self._maxsize

    def add(self, obs_t, action, reward):
        data = (obs_t, action, reward)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
        return np.array(obses_t), np.array(actions), np.array(rewards)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def get_data(self):
        idxes = list(range(self._maxsize))
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class DictReplayBuffer(object):
    def __init__(self, size, action_num, action_count_coef=1., q_coef=10, state_reward_coef=10,
                 global_count_coef=10, only_positive=False):
        self._maxsize = size
        self.action_num = action_num
        self._storage = OrderedDict()  # {obs_key:[obs_t, total_scores, total_counts, new_action, new_reward]}
        self.action_count_coef = action_count_coef
        self.q_coef = q_coef
        self.state_reward_coef = state_reward_coef
        self.global_count_coef = global_count_coef

        self.only_positive = only_positive

        self.total_visit_counts = 0

    def __len__(self):
        return len(self._storage)

    def size(self):
        return len(self._storage)

    def clear(self):
        self._storage = OrderedDict()

    def is_full(self):
        return len(self._storage) == self._maxsize

    def obs2key(self, obs):
        return bytes(obs)

    def get_state_count_reward(self, obs_tp1):
        # state based intrinsic reward
        obs_tp1_info = self._storage.get(self.obs2key(obs_tp1), None)
        if obs_tp1_info is not None:
            obs_tp1_visit_num = np.sum(obs_tp1_info[2])
        else:
            obs_tp1_visit_num = 0

        # rew = c_puct * np.sqrt(self.total_visit_counts) / (1 + obs_tp1_visit_num)
        rew = 1.02 ** -obs_tp1_visit_num * self.state_reward_coef

        return rew

    def get_global_count_reward(self, counts):
        rew = 1.02 ** -counts * self.global_count_coef

        return rew

    def add(self, obs_t, pi, reward, obs_tp1=None, ns_int=None):
        # debug
        # if len(self._storage) >= self._maxsize:
        #     return

        # update cache
        add_visited_count = 1
        skip_sample = self.only_positive and reward < 0
        # obs_key = bytes(obs_t)
        obs_key = self.obs2key(obs_t)
        if obs_key in self._storage:
            # when reward < 0, only add counts
            if skip_sample:
                # print("[replay_buffer]make reward to 0")
                add_visited_count = abs(reward // 10)
                reward = 0.

            action_index = np.argmax(pi)

            intrinsic_reward_list = []
            if ns_int is not None:
                intrinsic_reward_list.append(ns_int)
            if obs_tp1 is not None:
                obs_tp1_reward = np.zeros(self.action_num, dtype=np.float64)
                obs_tp1_reward[action_index] = self.get_state_count_reward(obs_tp1)
                intrinsic_reward_list.append(obs_tp1_reward)

            old = self._storage.pop(obs_key)

            # old = self._storage[obs_key].copy()

            old[1][action_index] += reward
            old[2][action_index] += add_visited_count
            # old[1][action_index] += obs_tp1_reward

            if self.global_count_coef != 0:
                intrinsic_reward_list.append(self.get_global_count_reward(old[2]))
            new_action, new_reward = self.generate_policy_and_reward(old[1], old[2],
                                                                     intrinsic_reward_list=intrinsic_reward_list)
            old[3], old[4] = new_action, new_reward

            self._storage[obs_key] = old

            # logger.info("[replay_buffer]Add_update\tscores:{}\tcounts:{}\taction:{}\treward:{}\tobs_tp1_reward:{}".format(
            #     list(self._storage[obs_key][1]), list(self._storage[obs_key][2]),
            #     list(self._storage[obs_key][3]), reward, obs_tp1_reward
            # ))
        else:
            if skip_sample:
                # print("[replay_buffer]obs not in buffer, skip")
                return

            if len(self._storage) >= self._maxsize:
                self._storage.popitem(last=False)

            total_scores = np.zeros(self.action_num, dtype=np.float64)
            total_counts = np.zeros(self.action_num)

            action_index = np.argmax(pi)

            intrinsic_reward_list = []
            if ns_int is not None:
                intrinsic_reward_list.append(ns_int)
            if obs_tp1 is not None:
                obs_tp1_reward = np.zeros(self.action_num, dtype=np.float64)
                obs_tp1_reward[action_index] = self.get_state_count_reward(obs_tp1)
                intrinsic_reward_list.append(obs_tp1_reward)

            total_scores[action_index] = reward
            total_counts[action_index] += add_visited_count
            # total_scores[action_index] += obs_tp1_reward

            if self.global_count_coef != 0:
                intrinsic_reward_list.append(self.get_global_count_reward(total_counts))
            new_action, new_reward = self.generate_policy_and_reward(total_scores, total_counts,
                                                                     intrinsic_reward_list=intrinsic_reward_list)

            self._storage[obs_key] = [obs_t, total_scores, total_counts, new_action, new_reward]

            # logger.info("[replay_buffer]Add_new\tscores:{}\tcounts:{}\taction:{}\treward:{}\tobs_tp1_reward:{}".format(
            #     list(self._storage[obs_key][1]), list(self._storage[obs_key][2]),
            #     list(self._storage[obs_key][3]), reward, obs_tp1_reward
            # ))

        self.total_visit_counts += 1

    def _encode_sample(self, idxes):
        obses_t, actions, rewards = [], [], []
        samples_list = list(self._storage.values())
        for i in idxes:
            obs_t, total_scores, total_counts, action, reward = samples_list[i]
            # action, reward = self.generate_policy_and_reward(total_scores, total_counts, self.c_puct)

            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
        return np.array(obses_t), np.array(actions), np.array(rewards)

    def get_visited_counts(self, obs_t):
        obs_key = bytes(obs_t)
        if obs_key in self._storage:
            return self._storage[obs_key][2]
        else:
            return None

    def generate_policy_and_reward(self, total_scores, total_counts, intrinsic_reward_list=None):
        # logger.info("[replay_buffer][generate_policy_and_reward]total_scores:{}\ttotal_counts:{}".format(list(total_scores),
        #                                                                                                  list(total_counts)))
        avg_scores = np.divide(total_scores, total_counts, out=np.zeros_like(total_scores), where=total_counts != 0)

        # positive_index = avg_scores > 0
        # non_negtive_index = avg_scores >= 0
        #
        # if non_negtive_index.any():
        #     Q_card_index = non_negtive_index
        #     U_card_index = positive_index
        # else:
        #     # calculate Q on negative elements
        #     Q_card_index = ~non_negtive_index
        #     U_card_index = ~non_negtive_index

        # avg_scores = np.where(positive_index, avg_scores, 0)
        # Q = np.zeros_like(avg_scores, dtype=np.float64)
        # Q[Q_card_index] = normalize(avg_scores[Q_card_index])
        # Q = normalize_v3(avg_scores)
        Q = avg_scores * self.q_coef

        # U = np.zeros_like(total_counts, dtype=np.float64)
        # U[U_card_index] = c_puct * np.sqrt(np.sum(total_counts[U_card_index])) / (1 + total_counts[U_card_index])
        U = self.action_count_coef * np.sqrt(np.sum(total_counts)) / (1 + total_counts)

        s = Q + U
        if intrinsic_reward_list is not None:
            for intrinsic_reward in intrinsic_reward_list:
                s += intrinsic_reward

        pi = normalize_v3(s)
        # pi = normalize(s)
        # pi = Q
        V = np.sum(pi * avg_scores)

        # logger.info("[replay_buffer][generate_policy_and_reward]Q:{}\tU:{}\tpi:{}\tv:{}".format(list(Q), list(U), list(pi), V))
        # logger.info("[replay_buffer][generate_policy_and_reward]Q:{}\tpi:{}\tv:{}".format(list(Q), list(pi), V))

        return pi, V

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def get_data(self, obs):
        """

        :param obs:
        :return: list, [obs_t, total_scores, total_counts, pi, Q]
        """
        obs_key = self.obs2key(obs)
        return self._storage.get(obs_key)


class PrioritizedReplayBufferDictWrapper(object):
    def __init__(self, size, alpha, action_num, c_puct=1., only_positive=False):
        self._maxsize = size
        self.action_num = action_num
        self.sample_cache = {}  # {obs:[scores, counts]}
        self.c_puct = c_puct
        self.only_positive = only_positive
        self.alpha = alpha

        self.replay_buffer = PrioritizedReplayBuffer(size=self._maxsize, alpha=self.alpha)

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, obs_t, action, reward):
        # debug
        # if len(self.replay_buffer) >= self._maxsize:
        #     return

        skip_sample = self.only_positive and reward < 0
        obs_key = bytes(obs_t)
        if obs_key in self.sample_cache:
            # when reward < 0, only add counts
            if skip_sample:
                # TODO: optimize
                reward = 0.

            action_index = np.argmax(action)
            old = self.sample_cache[obs_key]

            old[1][action_index] += reward
            old[2][action_index] += 1
            new_action, new_reward = DictReplayBuffer.generate_policy_and_reward(old[1], old[2], self.c_puct)

            self.replay_buffer.add(obs_t, new_action, new_reward)
            self.sample_cache[obs_key] = old

            logger.info("[replay_buffer]Add_update\tscores:{}\tcounts:{}\taction:{}\treward:{}".format(
                list(self.sample_cache[obs_key][1]), list(self.sample_cache[obs_key][2]),
                new_action, new_reward
            ))
        else:
            if skip_sample:
                return

            total_scores = np.zeros(self.action_num, dtype=np.float64)
            total_counts = np.zeros(self.action_num)

            action_index = np.argmax(action)
            total_scores[action_index] = reward
            total_counts[action_index] += 1

            new_action, new_reward = DictReplayBuffer.generate_policy_and_reward(total_scores, total_counts, self.c_puct)

            self.replay_buffer.add(obs_t, new_action, new_reward)
            self.sample_cache[obs_key] = [obs_t, total_scores, total_counts]

            logger.info("[replay_buffer]Add_new\tscores:{}\tcounts:{}\taction:{}\treward:{}".format(
                list(self.sample_cache[obs_key][1]), list(self.sample_cache[obs_key][2]),
                new_action, new_reward
            ))

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.replay_buffer.update_priorities(*args, **kwargs)


if __name__ == "__main__":
    r = DictReplayBuffer(size=3, action_num=3, c_puct=1, only_positive=True)

    # r = PrioritizedReplayBufferDictWrapper(size=10, action_num=3, c_puct=1, alpha=0.6, only_positive=True)
    r.add(obs_t=0, pi=np.array([1, 0, 0]), reward=10)
    r.add(obs_t=0, pi=np.array([0, 1, 0]), reward=-21)
    r.add(obs_t=0, pi=np.array([0, 0, 1]), reward=-5)
    r.add(obs_t=1, pi=np.array([0, 0, 1]), reward=10)
    r.add(obs_t=2, pi=np.array([0, 0, 1]), reward=10)
    r.add(obs_t=2, pi=np.array([0, 1, 1]), reward=10)

    r.update_priorities([2], [1000000000000.])
    r.sample(1, beta=0.6)
    # result = r.sample(1)
    # print(result)
    # r.add(obs_t=3, action=np.array([0,0,1]), reward=10)
    #
    # r.sample(3)
    # counts = np.array([1, 1, 1])
    # scores = np.array([-1, -10, -20])
    # DictReplayBuffer.generate_policy_and_reward(scores, counts)

    # r = PrioritizedReplayBuffer(size=3, alpha=0.1)
    # for i in range(5):
    #     r.add(i, i, i)