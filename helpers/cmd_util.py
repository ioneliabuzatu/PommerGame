"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from helpers import logger
from helpers import Monitor
from helpers import set_global_seeds
from helpers import make_atari, wrap_deepmind
from helpers import SubprocVecEnv

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)

    if reward_scale != 1.0:
        from helpers import RewardScaler
        env = RewardScaler(env, reward_scale)

    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6), 
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    for arg in args:
        assert arg.startswith('--')
        assert '=' in arg, 'cannot parse arg {}'.format(arg)
        key = arg.split('=')[0][2:]
        value = arg.split('=')[1]
        retval[key] = value

    return retval



