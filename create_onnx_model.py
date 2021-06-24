import numpy as np
import torch
from helpers.my_wrappers import PommerEnvWrapperFrameSkip2
from gym.spaces import Box, Discrete

from src.models.model_pomm import PommNet
from src.models.policy import Policy
from visualize_agent import make_video
import os, sys, argparse
from datetime import datetime
from helpers import pretrained_model

N_game = 50
NUM_ACTIONS = 6
RENDER = False
ENV_ID = "GraphicOVOCompact-v0"
ONNX_FILENAME = "second_stage_pommer_man.onnx"
USE_CUDA = True

if torch.cuda.is_available() and USE_CUDA:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

obs_space = Box(np.zeros(13440), np.ones(13440))
action_space = Discrete(6)
nn_kwargs = {
    "batch_norm": True,
    "recurrent": False,
    "hidden_size": 512,
    "cnn_config": "conv5",
}
actor_critic = Policy(
    PommNet(obs_shape=obs_space.shape, **nn_kwargs).eval(), action_space=action_space
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    type=str,
    help="path to .pt file",
)
parser.add_argument(
    "--name",
    type=str,
    help="name of resulting onnx file",
)
args = parser.parse_args()
print('-------------\nLoading model:', args.path, '\n-------------')

actor_critic.load_state_dict(torch.load(args.path)[0])
actor_critic = actor_critic.to(device)

input = torch.zeros((5, 56, 48)).to(device)
filename = ONNX_FILENAME if not args.name else args.name
torch.onnx.export(
    actor_critic,
    input.float(),
    f=filename,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
)
