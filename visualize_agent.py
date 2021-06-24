import cv2
import numpy as np
import torch

from graphic_pomme_env import graphic_pomme_env
from helpers.my_wrappers import PommerEnvWrapperFrameSkip2

from gym.spaces import Box, Discrete

from src.models.model_pomm import PommNet
from src.models.policy import Policy

import argparse

def make_video(list_of_observations_of_a_player, prefix):
    images = list_of_observations_of_a_player
    height, width, layer = images[0].shape
    video_name = f'{prefix}-video.avi'
    video = cv2.VideoWriter(video_name, 0, 3, (width, height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def play(model, opponent_actor=None):
    """ Plays for 50 games and then a video is saved under `assets/*video.avi` """
    model.eval()
    all_renders_img = []
    black_img = np.zeros((56,48,3), dtype=np.uint8)

    opp = ('Simple' if not opponent_actor else 'Custom')
    text_pos = (-1, 13)
    text_img = cv2.putText(black_img.copy(), opp, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    text_pos = (-1, 28)
    text_img = cv2.putText(text_img, 'Opponent', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    all_renders_img += [text_img]*8

    win_count_player = 0
    win_count_opponent = 0
    for start_pos in [0,1]:
        print(f"\n -- Start position {start_pos}:\n")
        text_pos = (1, 13)
        text_img = cv2.putText(black_img.copy(), f'Pos. {start_pos}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        all_renders_img += [text_img]*5

        env = PommerEnvWrapperFrameSkip2(
            num_stack=5, start_pos=start_pos, board="GraphicOVOCompact-v0", opponent_actor=opponent_actor
        )

        for i_episode in range(3):
            obs, opponent_obs = env.reset()
            done = False

            renders_img = []
            rgb_img = np.array(env.get_rgb_img())
            renders_img.append(rgb_img)
            k = 0

            last_blast_str = 2
            last_max_ammo = 1 
            while not done:
                k += 1
                net_out = model(torch.tensor(obs).float())
                action = net_out.argmax(1).item()

                if opponent_actor is not None:
                    opponent_obs = torch.from_numpy(np.array(opponent_obs)).float()
                    net_out = opponent_actor(opponent_obs).cpu().detach().numpy()
                    opponent_action = np.argmax(net_out)

                    agent_step, opponent_step, blast_str, ammo = env.step(action, opponent_action)
                else:
                    agent_step, opponent_step, blast_str, ammo = env.step(action)

                obs, r, done, info = agent_step
                opponent_obs, _, _, _ = opponent_step

                if blast_str > last_blast_str:
                    print(" > Hooray! Blast Strength increased!")
                    last_blast_str = blast_str 

                if ammo > last_max_ammo:
                    print(" > Hooray! Ammo increased!")
                    last_max_ammo = ammo

                rgb_img = np.array(env.get_rgb_img())
                renders_img.append(rgb_img)

                if k >= 800:
                    r = 0
                    outcome = "DRAW"
                    break

            if r > 0:
                win_count_player += 1
                outcome = "WIN"
            elif r < 0:
                win_count_opponent += 1
                outcome = "LOSS"


            text_pos = (3, 30)
            text = f"{win_count_player}-{win_count_opponent}"
            end_img = cv2.putText(rgb_img.copy(), text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1)

            text_pos = (1, 13)
            text_img = cv2.putText(black_img.copy(), f'Game {i_episode+1}:', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            text_pos = (1, 25)
            print(outcome, f"after {k} steps")
            text_img = cv2.putText(text_img, outcome, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            if outcome == "DRAW":
                renders_img = [text_img]*5 + renders_img[:100] + [end_img] * 10 # draws can take up a lot of time, so cut off at 100 frames
            else:
                renders_img = [text_img]*5 + renders_img + [end_img] * 10

            all_renders_img += renders_img


    make_video(all_renders_img, "assets/latest_pommerman_games")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="path to .pt file of agent",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        help="path to .pt file of opponent",
    )
    args = parser.parse_args()
    print('-------------\nLoading model:', args.path, '\n-------------')

    obs_space = Box(np.zeros(13440), np.ones(13440))
    action_space = Discrete(6)
    nn_kwargs = {'batch_norm': True, 'recurrent': False, 'hidden_size': 512, 'cnn_config': 'conv5', }
    actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).eval(), action_space=action_space)

    actor_critic.load_state_dict(torch.load(args.path)[0])
    actor_critic = actor_critic

    if args.opponent:
        print('-------------\nLoading opponent:', args.opponent, '\n-------------')
        opponent_actor = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).eval(), action_space=action_space)

        opponent_actor.load_state_dict(torch.load(args.opponent)[0])
        opponent_actor = opponent_actor
    else:
        opponent_actor = None

    play(actor_critic, opponent_actor)
