import cv2
import numpy as np
import torch

from graphic_pomme_env import graphic_pomme_env
from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2
from gym.spaces import Box, Discrete

from src.models.model_pomm import PommNet
from src.models.policy import Policy


def make_video(list_of_observations_of_a_player, prefix):
    images = list_of_observations_of_a_player
    height, width, layer = images[0].shape
    video_name = f'{prefix}-video.avi'
    video = cv2.VideoWriter(video_name, 0, 3, (width, height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def play(model, opponent_actor=None, game_score=None):
    """ Plays for 50 games and then a video is saved under `assets/*video.avi` """
    model.eval()
    all_renders_img = []
    black_img = np.zeros((56, 48, 3), dtype=np.uint8)

    opp = ('Simple' if not opponent_actor else 'Custom')
    text_pos = (-1, 13)
    text_img = cv2.putText(black_img.copy(), opp, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    text_pos = (-1, 28)
    text_img = cv2.putText(text_img, 'Opponent', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    all_renders_img += [text_img] * 8

    win_count_player = 0
    win_count_opponent = 0
    for start_pos in [0,1]:

        print(f"\n -- Start position {start_pos}:\n")
        text_pos = (1, 13)
        text_img = cv2.putText(black_img.copy(), f'Pos. {start_pos}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                               (255, 255, 255), 1)
        all_renders_img += [text_img] * 5

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
                net_out = model(torch.tensor(obs).float().cuda())
                action = net_out.argmax(1).item()

                agent_step, opponent_step = env.step(action)
                obs, r, done, info = agent_step

                rgb_img = np.array(env.get_rgb_img())
                renders_img.append(rgb_img)

                if k >= 800:
                    r = 0
                    game_score = "DRAW"
                    break

            if r > 0:
                win_count_player += 1
                game_score = "WIN"
            elif r < 0:
                win_count_opponent += 1
                game_score = "LOSS"

            text_pos = (3, 30)
            text = f"{win_count_player}-{win_count_opponent}"
            end_img = cv2.putText(rgb_img.copy(), text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

            text_pos = (1, 13)
            text_img = cv2.putText(black_img.copy(), f'Game {i_episode + 1}:', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                   (255, 255, 255), 1)
            text_pos = (1, 25)
            print(game_score, f"after {k} steps")
            text_img = cv2.putText(text_img, game_score, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if game_score == "DRAW":
                renders_img = [text_img] * 5 + renders_img[:100] + [
                    end_img] * 10  # draws can take up a lot of time, so cut off at 100 frames
            else:
                renders_img = [text_img] * 5 + renders_img + [end_img] * 10

            all_renders_img += renders_img

    make_video(all_renders_img, f"assets/three_games_pommerman_start_pos_{start_pos}")


if __name__ == "__main__":
    obs_space = Box(np.zeros(13440), np.ones(13440))
    action_space = Discrete(6)
    nn_kwargs = {'batch_norm': True, 'recurrent': False, 'hidden_size': 512, 'cnn_config': 'conv5', }
    actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).train(), action_space=action_space)
    state_dict, _ = torch.load("./checkpoints/stage_2.pt")
    actor_critic.load_state_dict(state_dict)
    actor_critic = actor_critic.cuda()
    play(actor_critic, opponent_actor=None)
