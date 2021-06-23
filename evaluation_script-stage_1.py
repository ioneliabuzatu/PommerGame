# PyTorch imports
import numpy as np
import torch
import onnx
from onnx2pytorch import ConvertModel
import argparse
import sys
from gym import logger as gymlogger

# Environment import and set logger level to display error only
gymlogger.set_level(40)  # error only
# ignore prints to stdout of imports
save_stdout = sys.stdout
sys.stdout = open('trash', 'w')
import os

sys.stdout = save_stdout
from graphic_pomme_env import graphic_pomme_env
from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2
np.random.seed(147)
torch.manual_seed(147)

if __name__ == "__main__":
    N_EPISODES = 50

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, default="submission_model.onnx")
    args = parser.parse_args()
    model_file = args.submission

    # Network
    net = ConvertModel(onnx.load(model_file), experimental=True)
    net.eval()

    win_count = 0.0
    env = PommerEnvWrapperFrameSkip2(num_stack=5, start_pos=0, board='GraphicOVOCompact-v0')

    for i in range(N_EPISODES):

        done = False
        obs, opponent_obs = env.reset()
        while not done:
            obs = torch.from_numpy(np.array(obs)).float()
            net_out = net(obs).detach().cpu().numpy()
            action = np.argmax(net_out)

            agent_step, opponent_step = env.step(action)
            obs, r, done, info = agent_step

        if r > 0:
            win_count += 1

    print(win_count / N_EPISODES)