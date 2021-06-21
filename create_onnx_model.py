import numpy as np
import torch
from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2
from gym.spaces import Box, Discrete
from torch import nn
from src.models.model_pomm import PommNet
from src.models.policy import Policy
import cv2

N_game = 50
NUM_ACTIONS = 6
RENDER = False
ENV_ID = 'GraphicOVOCompact-v0'

obs_space = Box(np.zeros(13440), np.ones(13440))
action_space = Discrete(6)
nn_kwargs = {'batch_norm': True, 'recurrent': False, 'hidden_size': 512, 'cnn_config': 'conv5', }
actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).train(), action_space=action_space)
actor_critic.load_state_dict( torch.load("./checkpoints/stage_1.pt")[0])
actor_critic = actor_critic.cuda()


def make_video(list_of_observations_of_a_player, prefix):
    images = list_of_observations_of_a_player
    height, width, layer = images[0].shape
    video_name = f'{prefix}-video.avi'
    video = cv2.VideoWriter(video_name, 0, 3, (width, height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def evaluate_model(model, opponent_actor=None):
    print("evaluating...")
    model.eval()

    # Make the "Free-For-All" environment using the agent list
    env = PommerEnvWrapperFrameSkip2(num_stack=5, start_pos=1, board=ENV_ID, opponent_actor=opponent_actor)
    # Run the episodes just like OpenAI Gym
    win_cnt = 0
    draw_cnt = 0
    lost_cnt = 0
    all_renders_img = []

    for i_episode in range(N_game):
        obs, opponent_obs = env.reset()
        done = False
        step_cnt = 0
        while not done:
            if i_episode < 5:
                rgb_img = np.array(env.get_rgb_img())
                all_renders_img.append(rgb_img)

            net_out = model(torch.tensor(obs).float().cuda())
            action = net_out.argmax(1).item()
            agent_step, opponent_step = env.step(action)
            obs, r, done, info = agent_step
            step_cnt += 1

        if r > 0:
            win_cnt += 1
        elif step_cnt >= 800:
            draw_cnt += 1
        else:
            lost_cnt = lost_cnt + 1
        # print('Episode {} finished'.format(i_episode))
    print('win:', win_cnt, 'draw_cnt:', draw_cnt, 'lose_cnt:', lost_cnt)
    print('\n')
    make_video(all_renders_img, "assets/three_games_pommerman")
    return obs


sample_input = evaluate_model(actor_critic)
input = torch.tensor(sample_input).cuda()
torch.onnx.export(actor_critic, input.float(), f="submission_model.onnx", export_params=True, opset_version=12,
                  do_constant_folding=True)
