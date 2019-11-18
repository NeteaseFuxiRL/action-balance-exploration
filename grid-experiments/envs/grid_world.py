import numpy as np
import random
import sys
import os

import gym
from gym import spaces
from gym.utils import seeding


class DelayRewardEnv(gym.Wrapper):
    def __init__(self, env, delay_steps=50, reward_pattern=None, remove_inner_reward=True):
        """Makes getting_reward == end-of-episode, but only reset on true game over.
        Done for games like Bowling, where you get reward after 2 shoots, but the game
        itself takes very long to finish.
        """
        gym.Wrapper.__init__(self, env)
        self.reward = 0
        self.current_step = 0
        self.delay_steps = delay_steps
        self.was_real_done = True
        # self.counter = 0
        # self.frames_with_no_reward_counter = 0
        # self.frames_with_no_reward = 1000
        self.minimal_reward = 0
        self.reward_pattern = np.array(reward_pattern)
        self.trajectory = []

        self.reward_mode = ['reward_pattern_num']
        self.remove_inner_reward = remove_inner_reward

    def check_real_done(self):
        done = (self.current_step == self.delay_steps)
        #  y:10 hp:350 max_hp:35("env current_step:{} total:{}".format(self.current_step, self.delay_steps))

        return done

    def get_reward(self, mode="reward_pattern_num"):
        """
        Calculate delay reward.
        :return:
        """
        reward = 0
        if self.was_real_done:
            if mode == "reward_pattern_num":
                # reward = pattern_appear_num * coef
                coef = 2.

                # states = list(zip(*self.trajectory))[0]
                states = np.stack(np.array(self.trajectory)[:, 0])
                coordinates = states[:, 0:2]  # if change states, don't forget modify here.
                step_length = len(self.reward_pattern)
                appear_time = 0

                i = 0
                while i < len(coordinates):
                    if np.array_equal(coordinates[i:i+step_length], self.reward_pattern):
                        appear_time += 1
                        i += step_length
                    else:
                        i += 1

                reward = appear_time * coef

        return reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        # TODO: init_state need be added?
        self.trajectory.append([obs, reward, done, info])

        self.was_real_done = done = (self.check_real_done() or done)
        reward = self.get_reward()


        # if reward == 0:
        #     self.frames_with_no_reward_counter += 1
        # else:
        #     self.frames_with_no_reward_counter = 0

        # if self.frames_with_no_reward_counter > self.frames_with_no_reward:
        #     self.was_real_done = True
        #     reward = self.minimal_reward

        # if (reward != self.reward and np.abs(reward) > 0) or (lives < self.lives and lives > 0):
        #     self.counter += 1
        #     if self.counter == self.reward_counter:
        #         # reward = self.counter * reward
        #         done = True
        #         self.counter = 0
        #     # else:
        #     # make 'DoubleDunk' internal reward return in true done step.
        #     # reward = 0.

        self.reward = reward
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        # if self.was_real_done:
        #     obs = self.env.reset(**kwargs)
        #     reward = 0
        # else:
        #     # no-op step to advance from terminal/lost life state
        #     obs, reward, done, _ = self.env.step(0)
        #     self.was_real_done = done
        #
        #     if self.was_real_done:
        #         obs = self.env.reset(**kwargs)
        #         reward = 0
        obs = self.env.reset(**kwargs)
        self.reward = 0
        self.current_step = 0
        self.was_real_done = True
        self.trajectory = []

        return obs


class Grid(object):
    def __init__(self, x: int = None,
                 y: int = None,
                 type: int = 0,
                 reward: int = 0.0,
                 value: float = 0.0):  # value 属性备用
        self.x = x  # 坐标x
        self.y = y
        self.type = value  # 类别值(0:空；1:障碍或边界)
        self.reward = reward  # 该格子的即时奖励
        self.value = value  # 该格子的价值，暂没用上
        self.name = None  # 该格子的名称
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(self.x, self.y, self.type, self.reward, self.value,
                                                                   self.name)


class GridMatrix(object):
    '''格子矩阵，通过不同的设置，模拟不同的格子世界环境
    '''

    def __init__(self, n_width: int,  # 水平方向格子数
                 n_height: int,  # 竖直方向格子数
                 default_type: int = 0,  # 默认类型
                 default_reward: float = 0.0,  # 默认即时奖励值
                 default_value: float = 0.0  # 默认价值（这个有点多余）
                 ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x, y, self.default_type, self.default_reward, self.default_value))

    def get_grid(self, x, y=None):
        '''获取一个格子信息
        args: 坐标信息，由x,y表示或仅有一个类型为tuple的x表示
        return: grid object
        '''
        xx, yy = None, None
        if isinstance(x, (int, np.integer)):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert (xx >= 0 and yy >= 0 and xx < self.n_width and yy < self.n_height), \
            "任意坐标值应在合理区间"
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.reward = reward
        else:
            raise ("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.value = value
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type


class GridWorldEnv(gym.Env):
    '''格子世界环境，可以模拟各种不同的格子世界
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 4
    }

    def __init__(self, n_width: int = 10,
                 n_height: int = 7,
                 action_num: int = 4,
                 u_size=40,
                 default_reward: float = 0,
                 default_type=0, max_step=100,
                 windy=False, seed=None,
                 coords_scale=None):
        """

        Args:
            n_width:
            n_height:
            action_num:
            u_size:
            default_reward:
            default_type:
            max_step:
            windy:
            seed:
            coords_scale:
        """
        self.u_size = u_size  # 当前格子绘制尺寸
        self.n_width = n_width  # 格子世界宽度（以格子数计）
        self.n_height = n_height  # 高度
        self.width = u_size * n_width  # 场景宽度 screen width
        self.height = u_size * n_height  # 场景长度
        self.default_reward = default_reward
        self.default_type = default_type
        self.max_step = max_step
        self._adjust_size()
        self.coords_scale = coords_scale

        self.current_step = 0

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0.0)
        self.reward = 0  # for rendering
        self.action = None  # for rendering
        self.windy = windy  # 是否是有风格子世界

        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_num = action_num
        self.action_space = spaces.Discrete(action_num)
        self.reward_range = (-float('inf'), float('inf'))
        # 观察空间由low和high决定
        # self.observation_space = (spaces.Discrete(self.n_height), spaces.Discrete(self.n_width))
        # 坐标原点为左下角，这个pyglet是一致的
        # 通过设置起始点、终止点以及特殊奖励和类型的格子可以构建各种不同类型的格子世界环境
        # 比如：随机行走、汽车租赁、悬崖行走等David Silver公开课中的示例
        self.ends = [(4, 4)]  # 终止格子坐标，可以有多个
        self.start = (0, 0)  # 起始格子坐标，只有一个
        self.current_x, self.current_y = self.start[0], self.start[1]
        self.types = []  # 特殊种类的格子在此设置。[(3,2,1)]表示(3,2)处值为1
        self.rewards = []  # 特殊奖励的格子在此设置，终止格子奖励0
        self.refresh_setting()
        self.viewer = None  # 图形接口对象
        self._seed(seed=seed)  # 产生一个随机子
        self.state = self.reset()

        # self.observation_space = spaces.Box(0, max(self.n_width, self.n_height), shape=self.state.shape,
        #                                     dtype=np.int64)
        self.observation_space = spaces.Box(0, max(self.n_width, self.n_height), shape=self.state.shape,
                                            dtype=np.float32)

        self.movable_types_dict = {}

        self.rng = np.random.RandomState(seed=seed)

    def _adjust_size(self):
        '''调整场景尺寸适合最大宽度、高度不超过800
        '''
        pass

    def _seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        # obs = np.array([self.current_x, self.current_y, self.ends[0][0], self.ends[0][1]])
        obs = np.array([self.current_x, self.current_y])
        if self.coords_scale is not None:
            obs = obs * self.coords_scale

        return obs

    def print_env_infos(self):
        print("obs_type:{} action_type:{} state:{}".format(self.observation_space, self.action_space, self.state))

    def set_start_point(self, start_point_tuple):
        self.start = start_point_tuple  # 起始格子坐标，只有一个
        self.current_x, self.current_y = self.start[0], self.start[1]
        # self.refresh_setting()
        self.state = self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        self.action = action  # action for rendering
        # old_x, old_y = self._state_to_xy(self.state)
        # old_x, old_y = int(self.state[0]), int(self.state[1])
        old_x, old_y = self.current_x, self.current_y
        new_x, new_y = old_x, old_y

        # wind effect:
        # 有风效果，其数字表示个体离开(而不是进入)该格子时朝向别的方向会被吹偏离的格子数
        if self.windy:
            if new_x in [3, 4, 5, 8]:
                new_y += 1
            elif new_x in [6, 7]:
                new_y += 2

        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down

        elif action == 4:
            new_x, new_y = new_x - 1, new_y - 1
        elif action == 5:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 6:
            new_x, new_y = new_x - 1, new_y + 1
        elif action == 7:
            new_x, new_y = new_x + 1, new_y + 1

        # boundary effect
        if new_x < 0 or new_x >= self.n_width or new_y < 0 or new_y >= self.n_height:
            new_x = old_x
            new_y = old_y

        # wall effect:
        # 类型为1的格子为障碍格子，不可进入
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)

        # done = self._is_end_state(new_x, new_y)
        done = self._is_done(new_x, new_y)
        # self.state = self._xy_to_state(new_x, new_y)
        self.current_x, self.current_y = new_x, new_y

        # self.state = [new_x, new_y, self.ends[0][0], self.ends[0][1]]
        self.state = self._get_obs()
        # 提供格子世界所有的信息在info内
        # info = {"x": new_x, "y": new_y, "grids": self.grids}
        info = {}
        # time.sleep(0.01)
        self.current_step += 1

        return self.state, self.reward, done, info

    # 将状态变为横纵坐标
    def _statenum_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_statenum(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # 未知状态

    def refresh_setting(self):
        '''用户在使用该类创建格子世界后可能会修改格子世界某些格子类型或奖励值
        的设置，修改设置后通过调用该方法使得设置生效。
        '''
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        # self.state = self._xy_to_state(self.start)
        self.current_x, self.current_y = self.start[0], self.start[1]
        # self.state = np.array([self.start[0], self.start[1], self.ends[0][0], self.ends[0][1]])
        self.state = self._get_obs()
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=self._get_obs().shape, dtype='float32')
        self.current_step = 0
        return self.state

    def random_reset(self):
        # self.ends = [(9, 7)]
        # self.rewards = [(9, 7, 1)]
        num = random.randint(0, 98)
        x, y = self._statenum_to_xy(num)
        self.state = [x, y, self.ends[0][0], self.ends[0][1]]
        return self.state

    def _is_done(self, x, y=None):
        # is_end or is_max_step

        return self._is_end_state(x, y) or (self.current_step + 1 >= self.max_step)

    # 判断是否是终止状态
    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._statenum_to_xy(x)
        else:
            assert (isinstance(x, tuple)), "坐标数据不完整"
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    # 图形化界面
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # 格子之间的间隙尺寸

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
            #    属性就是变换属性，
            #    该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制
            ''' 绘制水平竖直格子线，由于设置了格子之间的间隙，可不用此段代码
            for i in range(self.n_width+1):
                line = rendering.Line(start = (i*u_size, 0), 
                                      end =(i*u_size, u_size*self.n_height))
                line.set_color(0.5,0,0)
                self.viewer.add_geom(line)
            for i in range(self.n_height):
                line = rendering.Line(start = (0, i*u_size),
                                      end = (u_size*self.n_width, i*u_size))
                line.set_color(0,0,1)
                self.viewer.add_geom(line)
            '''

            # 绘制格子
            self.translations_dict = {}

            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x * u_size + m, y * u_size + m),
                         ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m),
                         (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 100
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    # 绘制边框
                    v_outline = [(x * u_size + m, y * u_size + m),
                                 ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m),
                                 (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x, y):
                        # 给终点方格添加金黄色边框
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y) == 1:  # 障碍格子用深灰色表示
                        rect.set_color(0.3, 0.3, 0.3)
                    elif self.grids.get_type(x, y) == 2:  # red
                        rect.set_color(1., 0., 0.)

                    else:
                        pass

                    agent_tmp = rendering.make_circle(u_size / 10, 30, True)
                    agent_tmp.set_color(0, 0.9, 0)
                    self.viewer.add_geom(agent_tmp)
                    trans_tmp = rendering.Transform()
                    agent_tmp.add_attr(trans_tmp)

                    self.translations_dict[(x, y)] = trans_tmp

            # 绘制个体
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            # self.translations_dict = {}
            # color_dict = {3: (1.0, 1.0, 0.0), 4: (0, 0.4, 0)}
            # for key, color_type in self.movable_types_dict.items():
            #     agent_tmp = rendering.make_circle(u_size / 5, 20, False)
            #     agent_tmp.set_color(*color_dict[color_type])
            #     self.viewer.add_geom(agent_tmp)
            #     trans_tmp = rendering.Transform()
            #     agent_tmp.add_attr(trans_tmp)
            #
            #     self.translations_dict[key] = trans_tmp

        # 更新个体位置
        # x, y = self._state_to_xy(self.state)
        # x = self.state[0]
        # y = self.state[1]
        x, y = self.current_x, self.current_y
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        for x in range(self.n_width):
            for y in range(self.n_height):
                key = (x, y)
                if key in self.movable_types_dict:
                    self.translations_dict[key].set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)
                    self.translations_dict[key].set_scale(1, 1)
                else:
                    self.translations_dict[key].set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)
                    self.translations_dict[key].set_scale(0, 0)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close_render(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class AssignGoalEnv(GridWorldEnv):
    metadata = {
        'goal2xy.modes': ['horizontal', 'swatooth']
    }

    def __init__(self, goals, traj_length,
                 can_see_goal=True, get_goal_func=None,
                 seed=None, goal2xy_mode='swatooth',
                 end_reward=None,
                 *args, **kwargs):

        self.can_see_goal = can_see_goal
        self.get_goal_func = get_goal_func
        self.goal2xy_mode = goal2xy_mode
        self.end_reward = end_reward

        self.true_goal = None
        self.seen_goal = None

        # simulate B signals
        from simple_trajectory_test import TrajectoryGenerator
        self.tg = TrajectoryGenerator(length=traj_length, seed=seed)
        self.tg.set_style_features(style_feature=goals)

        super().__init__(*args, **kwargs)

    def _get_obs(self):
        obs = [self.current_x, self.current_y]
        if self.coords_scale is not None:
            obs = [x * self.coords_scale for x in obs]

        if self.can_see_goal:
            obs.extend(self.seen_goal)

        return np.array(obs)

    def update_goal(self):
        batch_data, batch_label = self.tg.sampling_trajectory(batch_size=1, fix_positive_rate=True)
        self.true_goal = np.sum(batch_data > 0)

        if self.get_goal_func is None:
            # self.seen_goal = [self.true_goal]
            # true x,y
            self.seen_goal = [*self.goal2xy(goal=self.true_goal, mode=self.goal2xy_mode)]
        else:
            self.seen_goal = self.get_goal_func.get_goal(batch_data)[0]

    @staticmethod
    def goal2xy(goal, mode):
        """
        change int goal to its x and y
        :return: tuple, (x, y)
        """
        x = goal

        if mode == 'horizontal':
            y = 0
        elif mode == 'swatooth':
            period = 3
            a = (x // period) % 2 == 0
            b = x % period

            y = a*b + (1-a)*(period-b-1)

        return x, y

    def reset(self):
        """
        """
        self.update_goal()
        x, y = self.goal2xy(goal=self.true_goal, mode=self.goal2xy_mode)

        if self.end_reward is not None:
            old_x, old_y = self.ends[0]
            self.rewards = [(old_x, old_y, self.default_reward), (x, y, self.end_reward)]
            self.refresh_setting()

        self.ends = [(x, y)]
        self.close_render()

        # TODO: self.state must be updated when modify grid state.
        super().reset()

        return self.state


class Character(object):
    def __init__(self, x, y, damage, attack_range, hp=100, max_hp=100, heal_buff=None, damage_increase=0):
        """

        Args:
            x: int, start x
            y: int, start y
            damage: float,
            attack_range: int,
            hp: float, current hp
            max_hp: float, maximum hp
            heal_buff: float, each step heal hp num.
        """
        self.x = x
        self.y = y
        self.hp = hp
        self.damage = damage
        self.attack_range = attack_range
        self.max_hp = max_hp
        self.heal_buff = heal_buff
        self.damage_increase = damage_increase

        self._update_attack_coords()

    def __str__(self):
        return "x:{} y:{} hp:{} max_hp:{} damage:{} attack_range:{} heal_buff:{} damage_increase:{}".format(
            self.x, self.y, self.hp, self.max_hp, self.damage, self.attack_range, self.heal_buff, self.damage_increase)

    def _update_attack_coords(self):
        if self.x is not None and self.y is not None:
            self.attack_coords = [(self.x - self.attack_range, self.x + self.attack_range),
                                  (self.y - self.attack_range, self.y + self.attack_range)]

    def get_hidden_attributes(self):
        # return [self.damage, self.attack_range]
        return [self.damage, self.attack_range, self.damage_increase]

    def get_coords(self):
        return self.x, self.y

    @staticmethod
    def a_in_b_attack_range(a, b):
        x1, x2 = b.attack_coords[0]
        y1, y2 = b.attack_coords[1]

        return x1 <= a.x <= x2 and y1 <= a.y <= y2

    def in_attack_range(self, b):
        """
        self in b attack range.
        :param b:
        :return:
        """
        x1, x2 = b.attack_coords[0]
        y1, y2 = b.attack_coords[1]

        return x1 <= self.x <= x2 and y1 <= self.y <= y2

    def get_attack_coords(self):
        return self.attack_coords

    def set_coords(self, x, y):
        self.x = x
        self.y = y
        self._update_attack_coords()

    def get_position_damage(self, x, y, ndigits=5):
        dis_x = np.abs(self.x - x)
        dis_y = np.abs(self.y - y)
        dis_coef = self.attack_range - max(dis_x, dis_y)
        if dis_coef < 0:
            damage = 0
        else:
            damage = self.damage + self.damage_increase * dis_coef

        return round(damage, ndigits)


class ShootEnv(GridWorldEnv):
    """
    nxn GridWorld, enemy in center, with an attack range. Random born player whose goal is destroy the enemy.
    """
    def __init__(self, n_height, n_width, end_reward, enemy_attribute_dict_list, player_attribute_dict,
                 can_see_enemy_attribute=False,
                 seed=None,
                 damage_reward_coef=0., heal_reward_coef=0., player_hurt_punish_coef=0.,
                 state_scale=None,
                 fix_player_born_position=False,
                 loop_enemy=False,
                 auto_shoot=True,
                 *args, **kwargs):
        """
        Shoot and escape game.
        An enemy which can not move and always shoot. An player which can move and choose to shoot. Enemy and player
        have its own attributes, which contains attack range, damage, hp and so on (see class Character for details).
        Player's attribute is fixed when initialize game. Enemy random select an attribute from default attributes for
        current episode.
        Player can heal itself when not in enemy's attack range. Enemy can not heal.
        So, according to player and enemy's attribute, player may need to escape to heal itself when dying. And return
        to shoot enemy in order to win the game again and again.
        This loop operation make agent very hard to explore out successful trajectory, still less learn to solve it.
        Args:
            n_height: int, grid height
            n_width: int, grid width
            end_reward: float, game end reward.
            action_num: int, action num.
            default_reward: float, default step reward.
            default_type: int, default grid type. Details see super class.
            max_step: int, max step num of a episode.
            seed:
            coords_scale: list, scale at coordinates. If not None, will divide coordinates by this.
                Note, this may conflict with state_scale.
            enemy_attribute_dict_list: list, each element is an dict, contains enemy attributes.
                When episode start, random select one attribute to initialize current enemy.
                See Character for detail attributes.
            player_attribute_dict: dict, See Character for detail attributes.
            can_see_enemy_attribute:
            damage_reward_coef: float, how much of the damage from player to enemy consider as reward.
                Zero means no damage reward, one means fully real damage num as reward.
            heal_reward_coef: float, how much of real heal hp num at player consider as reward.
                Zero means no, One means total.
            state_scale: list, state scale num of each dimension. state = state / state_scale.
            fix_player_born_position: bool, whether fix player born position.
            loop_enemy: bool, loop in enemies instead of random select.
            auto_shoot: bool, If True, shoot will be another action,
                else agent will automatic shoot at every step without any dissipation.
                Since the last action of manual shoot represent shoot, the action_num of manual shoot is one bigger.
                E.g. If action_num=4 when auto_shoot=False, then action_num will be 5 when auto_shoot=True.
            *args:
            **kwargs:
        """
        self.n_height = n_height
        self.n_width = n_width
        self.enemy_attribute_dict_list = enemy_attribute_dict_list
        self.player_attribute_dict = player_attribute_dict
        self.can_see_enemy_attribute = can_see_enemy_attribute
        self.end_reward = end_reward

        self.auto_shoot = auto_shoot
        self.player_shoot_func = self.player_auto_shoot_step if self.auto_shoot else self.player_manual_shoot_step

        self.fix_player_born_position = fix_player_born_position
        self.loop_enemy = loop_enemy
        self.current_enemy_index = 0
        self.total_enemy_num = len(self.enemy_attribute_dict_list)

        self.damage_reward_coef = damage_reward_coef
        self.heal_reward_coef = heal_reward_coef
        self.player_hurt_punish_coef = player_hurt_punish_coef

        self.state_scale = state_scale

        self.rng = np.random.RandomState(seed=seed)

        self.enemy = Character(**self.rng.choice(self.enemy_attribute_dict_list))
        self.player = Character(**self.player_attribute_dict)

        self._update_player_born_bound()

        super().__init__(n_height=self.n_height, n_width=self.n_width, *args, **kwargs)

        self.init_grid_attributes()

        # self.reset()

    def print_env_infos(self):
        print('damage_reward_coef:{} heal_reward_coef:{} default_r:{} final_r:{}'.format(
            self.damage_reward_coef, self.heal_reward_coef, self.default_reward, self.end_reward))
        print('player {}'.format(self.player))
        print('enemy {}, full {}'.format(self.enemy, self.enemy_attribute_dict_list))
        super(ShootEnv, self).print_env_infos()

    def get_random_pos(self, bounds):
        """
        Get random pos based on bounds
        :param bounds: list, each element is a tuple or list of upper and lower limit of x and y respectively.
            E.g. [[(0, 4),(0, 4)], [(0, 4), (5, 9),...]]
        :return: sx, sy, random position
        """
        bound = bounds[self.rng.choice(len(bounds))]
        x, y = self.rng.randint(*bound[0]), self.rng.randint(*bound[1])

        return x, y

    def init_grid_attributes(self):
        self.ends = []
        self.types = []

        self.grids.reset()

        # update enemy type
        xs, ys = self.get_character_related_coords(self.enemy)
        self.types.extend(list(zip(xs, ys, [2] * len(xs))))

        self.types.append((*self.enemy.get_coords(), 1))
        self.refresh_setting()

    def update_movable_types(self, show=True):
        if show:
            xs, ys = self.get_character_related_coords(self.player)
            self.movable_types_dict = dict(zip(zip(xs, ys), [3] * len(xs)))
        else:
            self.movable_types_dict = {}

    def get_character_related_coords(self, c):
        masks = np.zeros(shape=(self.n_width, self.n_height), dtype=np.bool)
        (x_min, x_max), (y_min, y_max) = c.get_attack_coords()
        masks[slice(max(x_min, 0), x_max + 1), slice(max(y_min, 0), y_max + 1)] = True
        xs, ys = np.where(masks == 1)

        return xs, ys

    def _update_player_born_bound(self):
        self.player_born_bounds = []
        xs, ys = self.enemy.attack_coords

        if xs[0] > 0:
            self.player_born_bounds.append([(0, xs[0]), (0, self.n_height)])
        if xs[1] + 1 < self.n_width:
            self.player_born_bounds.append([(xs[1] + 1, self.n_width), (0, self.n_height)])
        if ys[0] > 0:
            self.player_born_bounds.append([(xs[0], xs[1] + 1), (0, ys[0])])
        if ys[1] + 1 < self.n_height:
            self.player_born_bounds.append([(xs[0], xs[1] + 1), (ys[1] + 1, self.n_height)])

    def _get_obs(self):
        obs = [self.current_x, self.current_y, self.player.hp, self.enemy.hp]
        # obs = [self.current_x, self.player.hp, self.enemy.hp]

        if self.can_see_enemy_attribute:
            obs.extend(self.enemy.get_hidden_attributes())

        obs = np.array(obs)
        if self.state_scale is not None:
            obs = obs / self.state_scale

        return obs

    def step(self, action):
        step_reward, damage_reward, show_movable = self.player_shoot_func(action)

        # heal
        heal_reward = self.heal() * self.heal_reward_coef
        # print('heal_reward', heal_reward)

        # enemy attack
        # _, _, _, _ = self.attack(self.enemy, self.player)
        hurt_punish = -self.attack(self.enemy, self.player) * self.player_hurt_punish_coef

        state = self._get_obs()
        done = self._is_done(self.current_x, self.current_y)

        hp_done, hp_done_reward = self._hp_done()
        if hp_done:
            done = hp_done
            reward = hp_done_reward
        else:
            reward = step_reward + heal_reward + damage_reward + hurt_punish

        # need make max_step reward = - end_reward?
        # elif done:
        #     reward = -self.end_reward

        self.update_movable_types(show_movable)

        return state, reward, done, {'enemy_index': self.current_enemy_index, 'player_hp': self.player.hp,
                                     'enemy_hp': self.enemy.hp, 'win': hp_done and (hp_done_reward > 0)}

    def player_manual_shoot_step(self, action):
        # player action
        if action == self.action_num - 1:
            # player attack
            # state, reward, done, info = self.attack(self.player, self.enemy)
            damage_reward = self.attack(self.player, self.enemy) * self.damage_reward_coef
            show_movable = True

            step_reward = self.grids.get_reward(self.current_x, self.current_y)
            self.current_step += 1
        else:
            # player move
            state, step_reward, done, info = super(ShootEnv, self).step(action)
            self.player.set_coords(x=self.current_x, y=self.current_y)
            show_movable = False

            damage_reward = 0

        return step_reward, damage_reward, show_movable

    def player_auto_shoot_step(self, action):
        # auto attack, remove attack action
        # move
        state, step_reward, done, info = super(ShootEnv, self).step(action)
        self.player.set_coords(x=self.current_x, y=self.current_y)

        # player attack
        # state, reward, done, info = self.attack(self.player, self.enemy)
        damage_reward = self.attack(self.player, self.enemy) * self.damage_reward_coef
        show_movable = True

        return step_reward, damage_reward, show_movable

    def _hp_done(self):
        enemy_died = self.enemy.hp <= 0
        player_died = self.player.hp <= 0

        return enemy_died or player_died, self.end_reward if (enemy_died and not player_died) else -self.end_reward
        # return enemy_died or player_died, self.end_reward if enemy_died else 0

    def heal(self):
        # player heal
        heal_num = 0
        if self.player.heal_buff is not None and not self.player.in_attack_range(self.enemy):
            now_hp = min(self.player.hp + self.player.heal_buff, self.player.max_hp)
            heal_num = now_hp - self.player.hp
            self.player.hp = now_hp
            # print('heal:{}'.format(self.player.hp))

        return heal_num

    def attack(self, a, b):
        """
        a attack b. If anyone hp <= 0, terminal and return final reward. Else, return grid reward.
        :param a:
        :param b:
        :return:
        """
        real_damage = 0
        if b.in_attack_range(a):
            if a.damage_increase != 0:
                a_current_damage = a.get_position_damage(x=b.x, y=b.y)
            else:
                a_current_damage = a.damage
            # print('current_damage:{}'.format(a_current_damage))
            # current_hp = max(0, b.hp - a.damage)
            current_hp = max(0, b.hp - a_current_damage)
            real_damage = b.hp - current_hp

            b.hp = current_hp
            # reward = real_damage  # attack reward
            # reward = real_damage * self.damage_reward_coef  # attack reward

        return real_damage

    def reset(self):
        """
        """
        # debug
        print('reset env, current step {}/{} state:{} player_hp:{} enemy_hp:{}'.format(
            self.current_step, self.max_step, self._get_obs(), self.player.hp, self.enemy.hp))

        if self.loop_enemy:
            self.current_enemy_index = (self.current_enemy_index + 1) % self.total_enemy_num
        else:
            self.current_enemy_index = self.rng.randint(self.total_enemy_num)

        self.enemy = Character(**self.enemy_attribute_dict_list[self.current_enemy_index])
        # print('reset enemy:{}'.format(self.enemy))

        if self.fix_player_born_position:
            x, y = self.player_attribute_dict['x'], self.player_attribute_dict['y']
            self.player = Character(**self.player_attribute_dict)
        else:
            self._update_player_born_bound()
            x, y = self.get_random_pos(bounds=self.player_born_bounds)
            attribute_tmp = self.player_attribute_dict.copy()
            attribute_tmp.update({'x': x, 'y': y})
            self.player = Character(**attribute_tmp)

        self.set_start_point((x, y))

        # annotate this line to stop update types every time. May fast.
        self.update_movable_types(show=False)

        self.init_grid_attributes()

        self.close_render()

        # self.state must be updated when modify grid state.
        super().reset()

        return self.state

    @staticmethod
    def can_destroy_enemy(player_hp, enemy_damage, enemy_attack_range, enemy_hp, player_damage):
        """
        Calculate minimum step num to destroy enemy.
        Args:
            player_hp:
            enemy_damage:
            enemy_attack_range:
            enemy_hp:
            player_damage:

        Returns:

        """
        # TODO: only support 1D array and start at fix 0.
        total_steps = 9 - enemy_attack_range
        hp_left = 0

        def loop(player_hp, enemy_hp):
            current_steps = 0
            hp_left = 0

            player_max_be_attacked_num = np.ceil(player_hp / enemy_damage) - 1
            enemy_die_num = np.ceil(enemy_hp / player_damage)

            if player_max_be_attacked_num <= enemy_attack_range:
                each_time_max_attack_num = - 1
                total_steps = -1
                hp_left = 0
            else:
                start_attack_step = enemy_attack_range - 1
                P_hp = player_hp - start_attack_step * enemy_damage
                current_steps += start_attack_step

                if P_hp - enemy_die_num * enemy_damage > 0:
                    # always attack
                    current_steps += enemy_die_num
                    each_time_max_attack_num = enemy_die_num
                    hp_left = P_hp - enemy_die_num * enemy_damage
                else:
                    each_time_max_attack_num = player_max_be_attacked_num - 2 * enemy_attack_range + 1

                    if each_time_max_attack_num >= 0:
                        current_steps = current_steps + 1 + each_time_max_attack_num + enemy_attack_range
                        _, steps, hp_left = loop(player_hp, enemy_hp - (each_time_max_attack_num + 1) * player_damage)
                        current_steps += steps

            return each_time_max_attack_num, current_steps, hp_left

        each_time_max_attack_num, current_steps, hp_left = loop(player_hp, enemy_hp)

        return each_time_max_attack_num, total_steps + current_steps, hp_left


class DelayRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.history_rewards = 0.

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.history_rewards += reward

        if done:
            reward = self.history_rewards
            self.history_rewards = 0.
        else:
            reward = 0

        return state, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset()


class StochasticGrid(GridWorldEnv):
    def __init__(self, wall_start_x, teleport_prob, teleport_point=None, end_reward=None, *args, **kwargs):
        self.wall_start_x = wall_start_x
        self.teleport_prob = teleport_prob
        self.end_reward = end_reward
        super(StochasticGrid, self).__init__(*args, **kwargs)

        self._init_grid()

        self.teleports = {}  # {(x, y): probability}
        self.set_teleports()
        if teleport_point is None:
            self.teleport_point = self.start
        else:
            self.teleport_point = teleport_point

    def set_teleports(self):
        for x in range(self.wall_start_x, self.n_width - 1):
            self.teleports[(x, self.n_height // 2)] = self.teleport_prob

    def _init_grid(self):
        self.set_start_point((self.wall_start_x - 1, self.n_height // 2))
        self.ends = [(self.n_width - 1, self.n_height // 2)]
        if self.end_reward is not None:
            for e in self.ends:
                self.rewards.append((*e, self.end_reward))

        # set walls
        xs = list(range(self.wall_start_x, self.n_width)) * (self.n_height - 1)
        ys = []
        for i in range(self.n_height):
            if i != self.n_height // 2:
                ys.extend([i] * (self.n_width - self.wall_start_x))

        self.types.extend(list(zip(xs, ys, [1] * len(xs))))

        self.refresh_setting()

    def step(self, action):
        state, reward, done, info = super(StochasticGrid, self).step(action)

        xy = (self.current_x, self.current_y)
        if xy in self.teleports:
            tele_prob = self.teleports[xy]
            if self.rng.random_sample() < tele_prob:
                # transport to start point
                print('teleport')
                self.current_x, self.current_y = self.teleport_point
                state = self.state = self._get_obs()
                reward = self.grids.get_reward(self.current_x, self.current_y)
                done = self._is_done(self.current_x, self.current_y)

        return state, reward, done, info


class StochasticGridV2(GridWorldEnv):
    def __init__(self, wall_start_x, teleport_prob, teleport_point=None, end_reward=None, *args, **kwargs):
        self.wall_start_x = wall_start_x
        self.teleport_prob = teleport_prob
        self.end_reward = end_reward
        self.teleport_size = 1

        super(StochasticGridV2, self).__init__(*args, **kwargs)

        self._init_grid()

        self.teleports = {}  # {(x, y): probability}
        self.set_teleports()
        if teleport_point is None:
            self.teleport_point = self.start
        else:
            self.teleport_point = teleport_point

    def set_teleports(self):
        # for x in range(self.wall_start_x, self.n_width - 1):
        #     self.teleports[(x, self.n_height // 2)] = self.teleport_prob
        for x in range(self.wall_start_x+1, min(self.wall_start_x+1 + self.teleport_size, self.n_width)):
            for y in range(self.teleport_size):
                self.teleports[(x, y)] = self.teleport_prob

    def _init_grid(self):
        self.set_start_point((0, 0))
        # self.ends = [(self.n_width - 1, self.n_height - 1)]
        self.ends = [(self.wall_start_x + 3, 0)]
        if self.end_reward is not None:
            for e in self.ends:
                self.rewards.append((*e, self.end_reward))

        # set walls
        for y in range(1, self.n_height):
            self.types.append((self.wall_start_x, y, 1))

        self.refresh_setting()

    def step(self, action):
        state, reward, done, info = super(StochasticGridV2, self).step(action)

        xy = (self.current_x, self.current_y)
        if xy in self.teleports:
            tele_prob = self.teleports[xy]
            if self.rng.random_sample() < tele_prob:
                # transport to start point
                print('teleport')
                self.current_x, self.current_y = self.teleport_point
                state = self.state = self._get_obs()
                reward = self.grids.get_reward(self.current_x, self.current_y)
                done = self._is_done(self.current_x, self.current_y)

        return state, reward, done, info


def LargeGridWorld():
    '''10*10的一个格子世界环境，设置参照：
    http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
    '''
    env = GridWorldEnv(n_width=10,
                       n_height=10,
                       u_size=40,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.start = (0, 9)
    env.ends = [(5, 4)]
    env.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                 (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                 (8, 7, 1)]
    env.rewards = [(3, 2, -1), (3, 6, -1), (5, 2, -1), (6, 2, -1), (8, 3, -1),
                   (8, 4, -1), (5, 4, 1), (6, 4, -1), (5, 5, -1), (6, 5, -1)]
    env.refresh_setting()
    return env


def SimpleGridWorld():
    '''无风10*10的格子，设置参照： David Silver强化学习公开课视频 第3讲, used
    '''
    env = GridWorldEnv(n_width=10,
                       n_height=10,
                       action_num=8,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    # env.action_space = spaces.Discrete(8)
    env.start = (0, 0)
    env.ends = [(9, 9)]
    env.rewards = [(9, 9, 1)]
    env.refresh_setting()
    return env


def SimpleGridWorld2():
    '''无风10*7的格子，设置参照： David Silver强化学习公开课视频 第3讲
    '''
    env = GridWorldEnv(n_width=5,
                       n_height=3,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)
    env.start = (0, 1)
    env.ends = [(3, 1)]
    env.rewards = [(3, 1, 1)]
    env.refresh_setting()
    return env


def WindyGridWorld():
    '''有风10*7的格子，设置参照： David Silver强化学习公开课视频 第5讲
    '''
    env = GridWorldEnv(n_width=10,
                       n_height=7,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=True)
    env.start = (0, 3)
    env.ends = [(7, 3)]
    env.rewards = [(7, 3, 1)]

    env.refresh_setting()
    return env


def RandomWalk():
    '''随机行走示例环境
    '''
    env = GridWorldEnv(n_width=7,
                       n_height=1,
                       u_size=80,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(2)  # left or right
    env.start = (3, 0)
    env.ends = [(6, 0), (0, 0)]
    env.rewards = [(6, 0, 1)]
    env.refresh_setting()
    return env


def RandomWalk2(width):
    '''随机行走示例环境
    '''
    if width % 2 == 0:
        width = width + 1
    env = GridWorldEnv(n_width=width,
                       n_height=1,
                       u_size=80,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(2)  # left or right
    env.start = (int((width - 1) / 2), 0)
    env.ends = [((width - 1), 0), (0, 0)]
    env.rewards = [((width - 1), 0, 1)]
    env.refresh_setting()
    return env


def CliffWalk():
    '''悬崖行走格子世界环境
    '''
    env = GridWorldEnv(n_width=12,
                       n_height=4,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (0, 0)
    env.ends = [(11, 0)]
    # env.rewards=[]
    # env.types = [(5,1,1),(5,2,1)]
    for i in range(10):
        env.rewards.append((i + 1, 0, -100))
        env.ends.append((i + 1, 0))
    env.refresh_setting()
    return env


def DynaMaze():
    '''Example 8.1
    '''
    env = GridWorldEnv(n_width=9,
                       n_height=6,
                       u_size=60,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (0, 3)
    env.ends = [(8, 5)]
    env.rewards = [(8, 5, 1)]
    env.types = [(2, 2, 1), (2, 3, 1), (2, 4, 1), (5, 1, 1), (7, 3, 1), (7, 4, 1), (7, 5, 1)]
    env.refresh_setting()
    return env


def ChangingModel():
    '''Example 8.2/8.3
    '''
    env = GridWorldEnv(n_width=10,
                       n_height=7,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)

    env.start = (0, 0)
    env.ends = [(4, 4)]
    env.rewards = [(0, 6, 1)]
    env.refresh_setting()
    return env


def ChangingModel2():
    '''Example 8.2/8.3
    '''
    env = GridWorldEnv(n_width=9,
                       n_height=6,
                       u_size=60,
                       default_reward=0,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (3, 0)
    env.ends = [(8, 5)]
    env.rewards = [(8, 5, 1)]
    env.types = [(1, 2, 1), (2, 2, 1), (3, 2, 1), (4, 2, 1), (5, 2, 1), (6, 2, 1), (7, 2, 1)]
    env.refresh_setting()
    return env


def SkullAndTreasure():
    '''骷髅与钱币示例，解释随机策略的有效性 David Silver 强化学习公开课第六讲 策略梯度
    '''
    env = GridWorldEnv(n_width=5,
                       n_height=2,
                       u_size=60,
                       default_reward=-1,
                       default_type=0,
                       windy=False)
    env.action_space = spaces.Discrete(4)  # left or right
    env.start = (0, 1)
    env.ends = [(2, 0)]
    env.rewards = [(0, 0, -100), (2, 0, 100), (4, 0, -100)]
    env.types = [(1, 0, 1), (3, 0, 1)]
    env.refresh_setting()
    return env
