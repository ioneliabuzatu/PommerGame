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
ONNX_FILENAME = "rename_checkpoint_pommer_man.onnx"
USE_CUDA = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    help="path to .pt file",
    default="GraphicOVOCompact-v0_recurrent_and_bombing.pt"
)
parser.add_argument(
    "--name",
    type=str,
    help="name of resulting onnx file",
)
args = parser.parse_args()

actor_critic = pretrained_model.load_pretrained(train=False, path=args.path, recurrent=True)

input = torch.zeros((5, 56, 48))
filename = ONNX_FILENAME if not args.name else args.name
torch.onnx.export(
    actor_critic,
    input.float(),
    f=filename,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
)