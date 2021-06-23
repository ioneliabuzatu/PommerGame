import numpy as np
import torch
import onnx
from onnx2pytorch import ConvertModel
import argparse
from gym import logger as gymlogger
import os
gymlogger.set_level(40)  # error only
#os.system("git clone https://github.com/MultiAgentLearning/playground ./pommer_setup")
#os.system("pip install -U ./pommer_setup")
#os.system('rm -rf ./pommer_setup')
#os.system("git clone https://github.com/RLCommunity/graphic_pomme_env ./graphic_pomme_env")
#os.system("pip install -U ./graphic_pomme_env")
#os.system('rm -rf ./graphic_pomme_env')

from graphic_pomme_env import graphic_pomme_env
from helpers.my_wrappers import PommerEnvWrapperFrameSkip2

np.random.seed(147)
torch.manual_seed(147)

if __name__ == "__main__":
    N_EPISODES = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, help="Path to onnx model of agent")
    parser.add_argument("--opponent", type=str, help="Path to onnx model of opponent")
    args = parser.parse_args()
    model_file = args.agent
    opponent_file = args.opponent

    # Agent Network
    print("\n------------\nagent file: ", model_file, '\n------------\n')
    agent = ConvertModel(onnx.load(model_file), experimental=True).cuda()
    agent.eval()
    # Opponent Network
    if opponent_file is not None:
        print("\n------------\nopponent file: ", opponent_file, '\n------------\n')
        opponent = ConvertModel(onnx.load(opponent_file), experimental=True).cuda()
        opponent.eval()
    else:
        print("\n------------\nNo opponent given, using SimpleAgent()...\n------------\n")

    win_count_player = 0.0
    draw_count = 0.0
    win_count_opponent = 0.0

    for i in range(N_EPISODES):
        if not i % 2:
            start_pos = 1
        else:
            start_pos = 0
        env = PommerEnvWrapperFrameSkip2(num_stack=5, start_pos=start_pos, board='GraphicOVOCompact-v0')
        done = False
        obs, opponent_obs = env.reset()
        observations = []
        n_steps = 0
        while not done:
            obs = torch.from_numpy(np.array(obs)).float().cuda()
            observations.extend(obs.cpu().detach().numpy().astype(np.uint8))
            net_out = agent(obs).cpu().detach().numpy()
            action = np.argmax(net_out)

            if opponent_file is not None:
                opponent_obs = torch.from_numpy(np.array(opponent_obs)).float().cuda()
                net_out = opponent(opponent_obs).cpu().detach().numpy()
                opponent_action = np.argmax(net_out)

                agent_step, opponent_step = env.step(action, opponent_action)
            else:
                agent_step, opponent_step = env.step(action)

            obs, r, done, info = agent_step
            opponent_obs, _, _, _ = opponent_step
            n_steps += 1
            if n_steps > 800:
                print("Draw")
                draw_count += 1
                r = 0
                break

        if r > 0:
            print("Win")
            win_count_player += 1
        elif r < 0:
            print("Loss")
            win_count_opponent += 1

    #print(f"Win ratio of agent: {win_count_player/N_EPISODES}")
    #print(f"Win ratio of opponent: {win_count_opponent / N_EPISODES}")
    print(f"Agent ({model_file}) had {win_count_player} wins, {draw_count} draws and {win_count_opponent} losses against the opponent ({opponent_file})")
