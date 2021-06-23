from gym.spaces import Discrete
from gym.spaces.box import Box
from src.models.model_pomm import PommNet
from src.models.policy import Policy
import numpy as np
import torch

def load_pretrained(train=True, path=None):
    obs_space = Box(np.zeros(13440), np.ones(13440), dtype=np.float32)
    action_space = Discrete(6)
    nn_kwargs = {'batch_norm': True, 'recurrent': False, 'hidden_size': 512, 'cnn_config': 'conv5', }

    if train:
        print("loading model for training...")
        actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).train(), action_space=action_space)
    else:
        print("loading model for evaluation...")
        actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).eval(), action_space=action_space)

    if path is None:
        actor_critic.load_state_dict(torch.load("./checkpoints/stage_2.pt")[0])
    else:
        actor_critic.load_state_dict(torch.load(path)[0])

    return actor_critic

