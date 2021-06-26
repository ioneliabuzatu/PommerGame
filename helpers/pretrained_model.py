from gym.spaces import Discrete
from gym.spaces.box import Box
from src.models.model_pomm import PommNet
from src.models.policy import Policy
import numpy as np
import torch

def load_pretrained(train=True, path=None, cuda=torch.cuda.is_available()):
    obs_space = Box(np.zeros(13440), np.ones(13440), dtype=np.float32)
    action_space = Discrete(6)
    nn_kwargs = {'batch_norm': True, 'recurrent': False, 'hidden_size': 512, 'cnn_config': 'conv5', }

    if path is None:
        path = "./checkpoints/stage_2.pt"

    if train:
        print(f"loading model for TRAINING: {path}")
        actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).train(), action_space=action_space)
    else:
        print(f"loading model for EVALUATION: {path}")
        actor_critic = Policy(PommNet(obs_shape=obs_space.shape, **nn_kwargs).eval(), action_space=action_space)

    if not cuda:
        actor_critic.load_state_dict(torch.load(path, map_location=torch.device('cpu') )[0])
    else:
        actor_critic.load_state_dict(torch.load(path)[0])

    return actor_critic

