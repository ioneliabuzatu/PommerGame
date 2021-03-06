import cv2
import numpy as np
import torch

from helpers import pretrained_model

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
            obs, _ = env.reset()
            done = False

            renders_img = []
            rgb_img = np.array(env.get_rgb_img())
            renders_img.append(rgb_img)
            k = 0

            while not done:
                k += 1

                net_out = model(torch.tensor(obs).float())
                action = net_out.argmax(1).item()

                agent_step, opponent_step, _, _ = env.step(action)

                obs, r, done, info = agent_step

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

    actor_critic = pretrained_model.load_pretrained(train=False, path=args.path)

    if args.opponent:
        opponent_actor = pretrained_model.load_pretrained(train=False, path=args.opponent)
    else:
        opponent_actor = None

    play(actor_critic, opponent_actor)

