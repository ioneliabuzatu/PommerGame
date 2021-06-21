import cv2
import numpy as np
import torch
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


def play(model, opponent_actor=None, start_pos=0):
    """ Plays for 50 games and then a video is saved under `assets/*video.avi` """
    model.eval()

    env = PommerEnvWrapperFrameSkip2(
        num_stack=5, start_pos=start_pos, board="GraphicOVOCompact-v0", opponent_actor=opponent_actor
    )
    all_renders_img = []

    for i_episode in range(50):
        obs, opponent_obs = env.reset()
        done = False
        while not done:
            rgb_img = np.array(env.get_rgb_img())
            all_renders_img.append(rgb_img)

            net_out = model(torch.tensor(obs).float().cuda())
            action = net_out.argmax(1).item()
            agent_step, opponent_step = env.step(action)
            obs, r, done, info = agent_step

    make_video(all_renders_img, "assets/three_games_pommerman")


if __name__ == "__main__":
    obs_space = Box(np.zeros(13440), np.ones(13440))
    action_space = Discrete(6)
    nn_kwargs = {'batch_norm': True, 'recurrent': False, 'hidden_size': 512, 'cnn_config': 'conv5', }
    actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).train(), action_space=action_space)
    actor_critic.load_state_dict(torch.load("./checkpoints/stage_1.pt")[0])
    actor_critic = actor_critic.cuda()
    play(actor_critic)
