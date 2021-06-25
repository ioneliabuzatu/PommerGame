from gym.spaces import Discrete
from gym.spaces.box import Box

import glob
import os
import time
import re
import argparse
from collections import deque
import gc

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import config
import src
from envs import make_vec_envs
from src.models.model_pomm import PommNet
from src.models.policy import Policy
from src.rollout_storage import RolloutStorage
from helpers import pretrained_model

update_factor = config.num_steps * config.num_processes
# num_updates = int(config.num_frames) // update_factor
num_updates = 10000
lr_update_schedule = (
    None if config.lr_schedule is None else config.lr_schedule // update_factor
)

torch.manual_seed(config.seed)
if config.cuda:
    torch.cuda.manual_seed(config.seed)
np.random.seed(config.seed)

try:
    os.makedirs(config.log_dir)
except OSError:
    files = glob.glob(os.path.join(config.log_dir, "*.monitor.csv"))
    for f in files:
        os.remove(f)

eval_log_dir = config.log_dir + "_eval"
try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, "*.monitor.csv"))
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

    print(f"Using {config.num_processes} processes")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="path to .pt file",
    )
    args = parser.parse_args()
    start_update = 0
    if args.path:
        pattern = re.match(r".*GraphicOVOCompact-v0_(\d+).pt", args.path)
        if pattern:
            start_update = int(pattern.group(1))

        actor_critic = pretrained_model.load_pretrained(train=True, path=args.path)
    else:
        obs_space = Box(np.zeros(13440), np.ones(13440), dtype=np.float32)
        action_space = Discrete(6)
        nn_kwargs = {
            "batch_norm": True,
            "recurrent": False,
            "hidden_size": 512,
            "cnn_config": "conv5",
        }
        actor_critic = Policy(
            PommNet(obs_shape=obs_space.shape, **nn_kwargs).train(),
            action_space=action_space,
        )

    actor_critic.to(device)

    agent = src.A2C_ACKTR(
        actor_critic,
        config.value_loss_coef,
        config.entropy_coef,
        lr=config.lr,
        lr_schedule=lr_update_schedule,
        eps=config.eps,
        alpha=config.alpha,
        max_grad_norm=config.max_grad_norm,
    )

    print("\n-----------\nMain Loop\n-----------\n")
    while True:
        if "envs" in locals():
            print("\n-------------------------")
            print("Resetting environment...")
            print("-------------------------\n")
            del envs
            gc.collect()

        print("\nCreating Training Environments...\n")
        envs = make_vec_envs(
            config.env_name,
            config.seed,
            config.num_processes,
            config.gamma,
            config.no_norm,
            config.num_stack,
            config.log_dir,
            config.add_timestep,
            device,
            allow_early_resets=False,
        )

        if config.eval_interval:
            print("\nCreating Eval Environments...\n")
            eval_envs = make_vec_envs(
                config.env_name,
                config.seed + config.num_processes,
                config.num_processes,
                config.gamma,
                config.no_norm,
                config.num_stack,
                eval_log_dir,
                config.add_timestep,
                device,
                allow_early_resets=True,
                eval=True,
            )

            if eval_envs.venv.__class__.__name__ == "VecNormalize":
                eval_envs.venv.ob_rms = envs.venv.ob_rms
        else:
            eval_envs = None

        rollouts = RolloutStorage(
            config.num_steps,
            config.num_processes,
            envs.observation_space.shape,
            envs.action_space,
            actor_critic.recurrent_hidden_state_size,
        )

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to_device(device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        times_since_bomb = torch.zeros((config.num_processes, 1), dtype=torch.int32)
        times_since_done = torch.zeros((config.num_processes, 1), dtype=torch.int32)
        last_blast_str = torch.zeros((config.num_processes, 1), dtype=torch.int32) + 2
        last_max_ammo = torch.ones((config.num_processes, 1), dtype=torch.int32)
        for j in range(start_update, start_update + num_updates):
            for step in range(config.num_steps):
                # Sample actions
                with torch.no_grad():
                    (
                        value,
                        action,
                        action_log_prob,
                        recurrent_hidden_states,
                    ) = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )

                # Obser reward and next obs
                obs, reward, done, infos, blast_str, ammo = envs.step(action)

                for info in infos:
                    if "episode" in info.keys():
                        episode_rewards.append(info["episode"]["r"])

                blast_str = torch.as_tensor(blast_str, dtype=torch.int32)[:, None]
                ammo = torch.as_tensor(ammo, dtype=torch.int32)[:, None]

                #####################################
                # give reward for collecting power-up
                idx_blast_incr = blast_str > last_blast_str
                if idx_blast_incr.sum():
                    print(" > Hooray! Blast Strength increased!")
                reward[idx_blast_incr] += 0.5
                last_blast_str = blast_str

                ####################################
                # give reward for collecting ammo-up
                idx_ammo_incr = ammo > last_max_ammo
                if idx_ammo_incr.sum():
                    print(" > Hooray! Ammo increased!")
                reward[idx_ammo_incr] += 0.5
                last_max_ammo[idx_ammo_incr] = ammo[idx_ammo_incr]

                ########################################
                ## punish not laying bombs for too long
                #idx_nobomb_thr = times_since_bomb >= 25
                #reward[idx_nobomb_thr] -= 0.1
                ##if idx_nobomb_thr.sum():
                ##    print(" > Punishing passive behaviour...")

                #######################################
                # small random reward for placing bombs
                reward[action == 5] += np.random.choice((0.1,0), p=(0.1, 0.9))

                ###############
                ## Punish draws
                #reward[done * (times_since_done>=800)] -= 0.5

                #############################################
                ## punish repeating patterns (leads to draws) - very experimental
                #act_buf = rollouts.actions.squeeze()
                #pattern_len = 4
                #num_occ = 1

                #assert (
                #    pattern_len * (num_occ + 1) <= config.num_steps
                #), "Increase config.num_steps!"
                #patterns = act_buf[-pattern_len:, :]
                #history = act_buf[:-pattern_len, :]

                #z = [[(patterns[:, j] == history[i : (i + patterns.shape[0]), j]).all()
                #        for i in range(history.shape[0] - patterns.shape[0])] 
                #        for j in range(act_buf.shape[1])]

                #z = torch.as_tensor(z)

                #idx_repeats = (z.sum(1) >= num_occ)[:, None] * (
                #    times_since_done >= pattern_len * 3
                #)
                #if idx_repeats.sum():
                #    print(" > Punishing repeating behaviour...")
                #reward[idx_repeats] -= 0.1

                ################################################
                # If done then clean the history of observations
                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done], device=device
                )
                rollouts.insert(
                    obs,
                    recurrent_hidden_states,
                    action,
                    action_log_prob,
                    value,
                    reward,
                    masks,
                )

                times_since_bomb += 1
                times_since_bomb[action == 5] = 0
                times_since_done += 1
                times_since_done[done] = 0

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(
                next_value, config.use_gae, config.gamma, config.tau
            )

            value_loss, action_loss, dist_entropy, other_metrics = agent.update(
                rollouts, j
            )

            rollouts.after_update()

            if j % config.save_interval == 0 and config.save_dir != "":
                save_path = os.path.join(config.save_dir)
                os.makedirs(save_path, exist_ok=True)
                state_dict = actor_critic.state_dict()
                save_model = [
                    state_dict,
                    hasattr(envs.venv, "ob_rms") and envs.venv.ob_rms or None,
                ]

                torch.save(
                    save_model, os.path.join(save_path, config.env_name + f"_{j}.pt")
                )

            total_num_steps = (j + 1) * update_factor

            if j % config.log_interval == 0 and len(episode_rewards) > 1:
                end = time.time()

                print(
                    f"Updates={j}, timesteps={total_num_steps}, FPS {total_num_steps // (end - start)} "
                    f"last {len(episode_rewards)}, mean {np.mean(episode_rewards):.3f} "
                    f"value/action loss {value_loss:.3f}/{action_loss:.3f}, entropy {dist_entropy:.3f}"
                )

                tensorboard.add_scalar(
                    "MeanReward",
                    np.mean(episode_rewards),
                    tensorboard_x_data_points_counts,
                )
                tensorboard.add_scalar(
                    "Entropy", dist_entropy, tensorboard_x_data_points_counts
                )
                tensorboard.add_scalar(
                    "ValueLoss", value_loss, tensorboard_x_data_points_counts
                )
                tensorboard.add_scalar(
                    "ActionLoss", action_loss, tensorboard_x_data_points_counts
                )
                tensorboard_x_data_points_counts += 1

            if (
                config.eval_interval
                and len(episode_rewards) > 1
                and j > 0
                and j % config.eval_interval == 0
            ):
                eval_episode_rewards = []

                obs = eval_envs.reset()
                eval_recurrent_hidden_states = torch.zeros(
                    config.num_processes,
                    actor_critic.recurrent_hidden_state_size,
                    device=device,
                )
                eval_masks = torch.zeros(config.num_processes, 1, device=device)

                while len(eval_episode_rewards) < 50:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs,
                            eval_recurrent_hidden_states,
                            eval_masks,
                            deterministic=True,
                        )

                    # Obser reward and next obs
                    obs, reward, done, infos, _, _ = eval_envs.step(action)
                    eval_masks = torch.tensor(
                        [[0.0] if done_ else [1.0] for done_ in done], device=device
                    )
                    for info in infos:
                        if "episode" in info.keys():
                            eval_episode_rewards.append(info["episode"]["r"])

                print(
                    f"Games played: {len(eval_episode_rewards)}, mean reward {np.mean(eval_episode_rewards):.5f}\n"
                )

        start_update += num_updates


if __name__ == "__main__":
    main()
