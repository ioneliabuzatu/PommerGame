from collections import deque
import numpy as np
from pommerman import make
from pommerman.agents import RandomAgent, SimpleAgent
import torch

# FrameStack implementation
# leaned on https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py

N_PLAYERS = 2
NUM_STACK = 5

NUM_ACTIONS = 6
'''
0 Stop
1 Up
2 Down
3 Left
4 Right
5 Bomb
'''

def rgb2grayscale(rgb_img_numpy):
  rgb_weights = [0.2989, 0.5870, 0.1140]
  grayscale_image = np.dot(rgb_img_numpy[...,:3], rgb_weights)
  return grayscale_image


class LazyFrames(object):
    r"""Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.

    .. note::

        This object should only be converted to numpy array just before forward pass.

    Args:
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    __slots__ = ('frame_shape', 'dtype', 'shape', 'lz4_compress', '_frames')

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress
            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack([self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0)

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress
            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(self.frame_shape)
        return frame


class FrameStack():
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

    """
    def __init__(self, num_stack, lz4_compress=False):
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

    def get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def append_frame(self, frame):
        self.frames.append(frame)
        return self.get_observation()

    def reset(self, initial_frame):
        [self.frames.append(initial_frame) for _ in range(self.num_stack)]
        return self.get_observation()


class PommerEnvWrapperFrameSkip2():
    """
    Pommerman environment wrapper, to save observations in a Framestack and skipping of every second frame.

    Args:
      num_stack: Number of frames that are stacked for the observation.
      start_pos: Starting position of player - either 0 for upper left or 1 for right corner
      opponent_actor: A opponent actor function can be defined, which gets the framestack of the opponent as input.
                      If no function is specified, the built in SimpleAgent is used as opponent.
      board: Defining the board size: 'GraphicOneVsOne-v0' for 8x8 or 'GraphicOVOCompact-v0' for 6x6 board
    """

    def __init__(self, num_stack, start_pos=0, opponent_actor=None,
                 board='GraphicOneVsOne-v0'):

        self.num_stack = num_stack
        self.start_pos = start_pos
        self.env = None
        self.frame_stack_even = FrameStack(num_stack)
        self.frame_stack_odd = FrameStack(num_stack)
        self.oppon_frame_stack_even = FrameStack(num_stack)
        self.oppon_frame_stack_odd = FrameStack(num_stack)
        self.board = board

        self.opponent_actor = opponent_actor

        if not opponent_actor:
            self.opponent = SimpleAgent()
        else:
            # Random agent serves as dummy; get the actions from opponent_actor function
            self.opponent = RandomAgent()

    def set_opponent_actor(self, opponent_actor):
        self.opponent_actor = opponent_actor

    def _recreate_env(self):
        if self.start_pos == -1:
            start_pos = np.random.randint(2)  # sample 0 or 1
        else:
            start_pos = self.start_pos

        if start_pos == 0:
            self.cur_start_pos = 0
            agent_list = [RandomAgent(), self.opponent]
        elif start_pos == 1:
            self.cur_start_pos = 1
            agent_list = [self.opponent, RandomAgent()]

        self.env = make(self.board, agent_list=agent_list,
                        render_mode='pixel_array')

    def print_cur_start_pos(self):
        if self.start_pos == -1:
            print(
                f"Random assignment of starting position - current start position: {self.cur_start_pos}")

    def step(self, action, opponent_action=None):
        # get opponent action
        if opponent_action is None:
            if not self.opponent_actor:
                raw_obs_list = self.env.get_last_step_raw()
                opponent_action = self.opponent.act(
                    raw_obs_list[1 - self.cur_start_pos],
                    NUM_ACTIONS)  # for SimpleAgent
            else:
                oppon_frame_stack = self.oppon_frame_stack_odd if self.next_is_even else self.oppon_frame_stack_even
                oppon_state = oppon_frame_stack.get_observation()
                oppon_state = torch.from_numpy(np.array(oppon_state)).float()

                device = next(self.opponent_actor.parameters()).device
                oppon_state = oppon_state.to(device)

                net_out = self.opponent_actor(oppon_state).detach().cpu().numpy()
                opponent_action = np.argmax(net_out).item()

        if self.cur_start_pos == 0:
            action_list = [action, opponent_action]
        else:
            action_list = [opponent_action, action]

        old_env_info = self.env.observations[self.cur_start_pos]

        # get observations
        observation_list, reward_list, done, info = self.env.step(action_list)

        new_env_info = self.env.observations[self.cur_start_pos]

        # for agent
        rgb_img = observation_list[self.cur_start_pos]
        self.last_state_rgb_img = rgb_img
        observation = rgb2grayscale(rgb_img)

        # for opponent
        oppon_obs = rgb2grayscale(observation_list[1 - self.cur_start_pos])

        # stack observations
        if self.next_is_even:
            observation_stack = self.frame_stack_even.append_frame(observation)
            oppon_obs_stack = self.oppon_frame_stack_even.append_frame(
                oppon_obs)
        else:
            observation_stack = self.frame_stack_odd.append_frame(observation)
            oppon_obs_stack = self.oppon_frame_stack_odd.append_frame(oppon_obs)

        self.next_is_even = not self.next_is_even

        agent_ret = (
        observation_stack, reward_list[self.cur_start_pos], done, info)
        opponent_ret = (
        oppon_obs_stack, reward_list[1 - self.cur_start_pos], done, info)

        return agent_ret, opponent_ret, old_env_info, new_env_info

    def reset(self, **kwargs):
        if self.env is None or self.start_pos == -1:
            self._recreate_env()

        observation_list = self.env.reset(**kwargs)

        self.last_state_rgb_img = observation_list[self.cur_start_pos]

        # observation for our player
        observation = rgb2grayscale(observation_list[self.cur_start_pos])
        self.frame_stack_odd.reset(observation)

        # observation for opponent player
        oppon_obs = rgb2grayscale(observation_list[1 - self.cur_start_pos])
        self.oppon_frame_stack_odd.reset(oppon_obs)

        self.next_is_even = True

        return self.frame_stack_even.reset(
            observation), self.oppon_frame_stack_even.reset(oppon_obs)

    def set_board_params(self, num_rigid=-1, num_wood=-1, num_items=-1):
        if self.env is None or self.start_pos == -1:
            self._recreate_env()
        self.env.set_num_board_params(num_rigid, num_wood, num_items)

    def get_rgb_img(self):
        return self.last_state_rgb_img

    def get_last_step_raw(self):
        return self.env.get_last_step_raw()

    def render_tiled_observations(self, obs):
        return self.env.render_tiled_observations(obs)