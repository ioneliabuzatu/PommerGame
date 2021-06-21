import glob
import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import config
import src
from envs import make_vec_envs
from src.models.model_pomm import PommNet
from src.models.policy import Policy
from src.rollout_storage import RolloutStorage

update_factor = config.num_steps * config.num_processes
num_updates = int(config.num_frames) // update_factor
lr_update_schedule = None if config.lr_schedule is None else config.lr_schedule // update_factor

torch.manual_seed(config.seed)
if config.cuda:
    torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)

try:
    os.makedirs(config.log_dir)
except OSError:
    files = glob.glob(os.path.join(config.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = config.log_dir + "_eval"
try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.cuda else "cpu")
    print(f"Training model on device [{device}] starting now...")
    if config.USE_BUDDY:
        tensorboard = config.tensorboard
    else:
        tensorboard = SummaryWriter("results")

    tensorboard_x_data_points_counts = 0

    envs = make_vec_envs(
        config.env_name, config.seed, config.num_processes, config.gamma, config.no_norm, config.num_stack,
        config.log_dir, config.add_timestep, device, allow_early_resets=False
    )

    if config.eval_interval:
        eval_envs = make_vec_envs(
            config.env_name, config.seed + config.num_processes, config.num_processes, config.gamma,
            config.no_norm, config.num_stack, eval_log_dir, config.add_timestep, device,
            allow_early_resets=True, eval=True
        )

        if eval_envs.venv.__class__.__name__ == "VecNormalize":
            eval_envs.venv.ob_rms = envs.venv.ob_rms
    else:
        eval_envs = None

    nn_kwargs = {'batch_norm': True, 'recurrent': config.recurrent_policy, 'hidden_size': 512, 'cnn_config': 'conv5', }
    nn = PommNet(obs_shape=envs.observation_space.shape, **nn_kwargs)
    nn.train()

    actor_critic = Policy(nn, action_space=envs.action_space)
    state_dict, _ = torch.load("./checkpoints/stage_1.pt")
    actor_critic.load_state_dict(state_dict)
    actor_critic.to(device)

    agent = src.A2C_ACKTR(
        actor_critic, config.value_loss_coef,
        config.entropy_coef,
        lr=config.lr, lr_schedule=lr_update_schedule,
        eps=config.eps, alpha=config.alpha,
        max_grad_norm=config.max_grad_norm
    )

    rollouts = RolloutStorage(
        config.num_steps, config.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to_device(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):
        for step in range(config.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], device=device)
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, config.use_gae, config.gamma, config.tau)

        value_loss, action_loss, dist_entropy, other_metrics = agent.update(rollouts, j)

        rollouts.after_update()

        if j % config.save_interval == 0 and config.save_dir != "":
            save_path = os.path.join(config.save_dir)
            os.makedirs(save_path, exist_ok=True)
            state_dict = actor_critic.state_dict() if device.type == "cpu" else actor_critic.state_dict()
            save_model = [state_dict, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, config.env_name + ".pt"))

        total_num_steps = (j + 1) * update_factor

        if j % config.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()

            print(f"Updates={j}, timesteps={total_num_steps}, FPS {total_num_steps // (end - start)} "
                  f"last {len(episode_rewards)}, mean {np.mean(episode_rewards):.3f} "
                  f"value/action loss {value_loss:.3f}/{action_loss:.3f}, entropy {dist_entropy:.3f}")

            tensorboard.add_scalar("MeanReward", np.mean(episode_rewards), tensorboard_x_data_points_counts)
            tensorboard.add_scalar("Entropy", dist_entropy, tensorboard_x_data_points_counts)
            tensorboard.add_scalar("ValueLoss", value_loss, tensorboard_x_data_points_counts)
            tensorboard.add_scalar("ActionLoss", action_loss, tensorboard_x_data_points_counts)
            tensorboard_x_data_points_counts += 1

        if config.eval_interval and len(episode_rewards) > 1 and j > 0 and j % config.eval_interval == 0:
            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(config.num_processes, actor_critic.recurrent_hidden_state_size,
                                                       device=device)
            eval_masks = torch.zeros(config.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 50:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], device=device)
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            print(f"Games played: {len(eval_episode_rewards)}, mean reward {np.mean(eval_episode_rewards):.5f}\n")


if __name__ == "__main__":
    main()
