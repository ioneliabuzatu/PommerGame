import cv2
import numpy as np
import torch
from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2
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
    text_pos = (0, 13)
    text_img = cv2.putText(black_img.copy(), opp, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    text_pos = (0, 28)
    text_img = cv2.putText(text_img, 'Opponent', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    all_renders_img += [text_img]*15

    for start_pos in [0,1]:
        text_pos = (1, 13)
        text_img = cv2.putText(black_img.copy(), f'Pos. {start_pos}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        all_renders_img += [text_img]*10

        env = PommerEnvWrapperFrameSkip2(
            num_stack=5, start_pos=start_pos, board="GraphicOVOCompact-v0", opponent_actor=opponent_actor
        )

        for i_episode in range(3):
            obs, opponent_obs = env.reset()
            done = False

            renders_img = []
            k = 0
            while not done:
                rgb_img = np.array(env.get_rgb_img())
                renders_img.append(rgb_img)

                net_out = model(torch.tensor(obs).float())
                action = net_out.argmax(1).item()

                if opponent_actor is None:
                    agent_step, opponent_step = env.step(action)
                else:
                    opponent_obs = torch.from_numpy(np.array(opponent_obs)).float()
                    net_out = opponent_actor(opponent_obs).cpu().detach().numpy()
                    opponent_action = np.argmax(net_out)

                    agent_step, opponent_step = env.step(action, opponent_action)

                obs, r, done, info = agent_step

                if k > 200:
                    r = -1
                    break

                k += 1

            text_pos = (1, 13)
            text_img = cv2.putText(black_img.copy(), f'Game {i_episode+1}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            text_pos = (1, 25)
            text = "WIN" * (r==1) + "DRAW" * (r==-1) + "LOSS" * (r==0)
            print(text)
            text_img = cv2.putText(text_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            renders_img = [text_img]*8 + renders_img

            all_renders_img += renders_img


    make_video(all_renders_img, "assets/three_games_pommerman_new")


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
