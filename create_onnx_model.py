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


def evaluate_model(model, opponent_actor=None, video_name=""):
    """
    the "Free-For-All" environment using the agent list env = PommerEnvWrapperFrameSkip2
    """
    print("evaluating...")
    model.eval()

    env = PommerEnvWrapperFrameSkip2(
        num_stack=5, start_pos=0, board=ENV_ID, opponent_actor=opponent_actor
    )
    win_cnt = 0
    lost_cnt = 0
    all_renders_img = []

    for i_episode in range(N_game):
        obs, opponent_obs = env.reset()
        done = False
        while not done:
            if i_episode < 4:
                rgb_img = np.array(env.get_rgb_img())
                all_renders_img.append(rgb_img)

            net_out = model(torch.tensor(obs).float().cuda())
            action = net_out.argmax(1).item()
            agent_step, opponent_step = env.step(action)
            obs, r, done, info = agent_step

        if r > 0:
            win_cnt += 1
        else:
            lost_cnt += 1

    print("win:", win_cnt, "lose_cnt:", lost_cnt)
    print("\n")
    time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    print("Saving video as:", video_name)
    make_video(all_renders_img, 'assets/' + video_name)
    return obs


#opponent_actor=None
opponent_actor=pretrained_model.load_pretrained(train=False)

video_name = args.path + ("__custom_opponent" if opponent_actor is not None else "__none_opponent")
video_name = video_name.split('/')[-1]

#sample_input = evaluate_model(actor_critic, opponent_actor=opponent_actor, video_name=video_name)
#input = torch.tensor(sample_input).to(device)

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
