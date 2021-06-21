import torch
import experiment_buddy

import torch.nn as nn

USE_BUDDY = True
use_cuda = True
host = ""

lr = 2.5e4
lr_schedule = 25000000
eps = 1e5
alpha = 0.99
gamma = 0.99
use_gae = True
tau = 0.95
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
seed = 1
num_processes = 16
num_steps = 5
num_stack = 1
log_interval = 10
save_interval = 100
eval_interval = 1000
num_frames = 5e7
env_name = 'GraphicOVOCompact-v0'
log_dir = '/tmp/gym/'
save_dir = './checkpoints/'
add_timestep = False  # add time_step to observations
recurrent_policy = False
no_norm = False  # disables normalization, no reward shaping
cuda = use_cuda and torch.cuda.is_available()
opponent_actor = None
starting_board_position = 0
random_start_position = False

if USE_BUDDY:
    experiment_buddy.register(locals())
    tensorboard = experiment_buddy.deploy(
        host,
        sweep_yaml="",
        proc_num=1,
        wandb_kwargs={"entity": "ionelia"}
    )
