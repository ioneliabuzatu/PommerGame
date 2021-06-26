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

VALUE_DICT = {'rigid':1, 'wood':2, 'bomb_incr':6, 
    'flame_incr':7, 'flame':4, 'bomb':3}

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.cuda else "cpu")
    print(f"Training model on device [{device}] starting now...")
    tensorboard = SummaryWriter("results")

    tensorboard_x_data_points_counts = 0

    print(f"Using {config.num_processes} processes")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="path to .pt file",
    )
    parser.add_argument(
        "--debug",
        help="Debug mode for exactly following what's happening in an environment",
        action="store_true",
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
                obs, reward, done, infos, old_env_info, new_env_info = envs.step(action)

                # env_info is a dict with the following keys:
                # 'alive', 'board', 'bomb_blast_strength', 'bomb_life', 
                # 'bomb_moving_direction', 'flame_life', 'game_type', 'game_env', 
                # 'position', 'blast_strength', 'can_kick', 'teammate', 'ammo', 
                # 'enemies', 'step_count', 'my_sprite'

                # from this we can get all kind of information:
                # blast strength, item positions, agent/opponent positions,...
                # this may be helpful in creating rewards for specific events!

                # merge list of dicts into dict of lists
                old_env_info = {k: [dic[k] for dic in old_env_info] for k in old_env_info[0]}
                new_env_info = {k: [dic[k] for dic in new_env_info] for k in new_env_info[0]}

                # mark agents position on board with value 111 and opponent 
                # position with 222
                for board_old, board_new, enemy_val in zip(old_env_info['board'], new_env_info['board'], old_env_info['enemies']):
                    agent_val = 10 if enemy_val[0].value==11 else 11
                    board_old[board_old==agent_val] = 111
                    board_old[board_old==enemy_val[0].value] = 222
                    board_new[board_new==agent_val] = 111
                    board_new[board_new==enemy_val[0].value] = 222

                ##############################
                # get positions of all objects
                # really messy but i don't care anymore
                pos_old = {key: [] for key in VALUE_DICT}
                pos_new = {key: [] for key in VALUE_DICT}

                pos_old['agent'] = torch.as_tensor(old_env_info['position'])
                pos_old['oppon'] = torch.zeros_like(pos_old['agent'])
                pos_new['agent'] = torch.as_tensor(new_env_info['position'])
                pos_new['oppon'] = torch.zeros_like(pos_new['agent'])

                for k, (b_old, b_new) in enumerate(zip(old_env_info['board'],new_env_info['board'])):
                    b_old, b_new = torch.as_tensor(b_old, dtype=torch.int32), torch.as_tensor(b_new, dtype=torch.int32)
                    if 222 in b_old:
                        pos_old['oppon'][k,:] = torch.as_tensor(torch.where(b_old==222))
                    if 222 in b_new:
                        pos_new['oppon'][k,:] = torch.as_tensor(torch.where(b_new==222))

                    for key in VALUE_DICT.keys():
                        pos_old[key].append(torch.as_tensor(np.where(b_old==VALUE_DICT[key])).T)
                        pos_new[key].append(torch.as_tensor(np.where(b_new==VALUE_DICT[key])).T)

                # direct distance to opponent
                # if dist_opp == 1, then the agent is right beside the opponent
                # if dist_opp == sqrt(2), then the agent is diagonally adjacent
                # if dist_opp > sqrt(2), then theres atleast 1 empty square between agent and opponent
                pos_old['dist_opp'] = ((pos_old['oppon']-pos_old['agent']).abs()**2).sum(1).sqrt()
                dist_opp_old_agent_new = ((pos_old['oppon']-pos_new['agent']).abs()**2).sum(1).sqrt()

                ##########################################################
                # give reward when placing bombs directly besides opponent
                # and give smaller reward when placing bombs diagonally adjacent to opponent
                idx = (action.cpu() == 5).squeeze()
                r1 = 0.3 * (pos_old['dist_opp'][idx]==1)[:,None] * (~done)[idx,None]
                reward[idx] += r1
                r2 = 0.15 * (pos_old['dist_opp'][idx]<2)[:,None] * (~done)[idx,None]
                reward[idx] += r2

                if args.debug and idx[0] and r1[0]:
                    print(" -> 0.3 reward for placing bomb directly beside opponent!")

                if args.debug and idx[0] and r2[0]:
                    print(" -> 0.15 reward for placing bomb diagonally beside opponent!")

                ##############################################
                # give small reward for going towards opponent
                idx = (pos_old['dist_opp'] > dist_opp_old_agent_new)
                reward[idx] += 0.05 * torch.from_numpy(~done)[idx,None]

                if args.debug and idx[0]:
                    print(" -> 0.05 reward for TRYING TO move towards opponent!")

                ###########################################
                # give reward for placing bomb becides wood
                dist_wood = [((pos_old['agent'][i]-w).abs()**2).sum(1).sqrt() for i, w in enumerate(pos_old['wood'])]
                num_wood_beside = torch.as_tensor([(el==1).sum() for el in dist_wood])
                idx = (num_wood_beside>=1)[:,None] * (action.cpu()==5)
                reward[idx] += 0.25 * torch.from_numpy(~done)[idx.squeeze()]

                if args.debug and idx[0]:
                    print(" -> 0.25 reward for placing bomb beside wood!")

                #################################
                # give reward for getting an item
                for k, (pos_a, pos_b, pos_f) in enumerate(zip(pos_new['agent'], pos_old['bomb_incr'], pos_old['flame_incr'])):
                    for p_b in pos_b:
                        r = (pos_a == p_b).all().item() * 0.5 * (~done)[k]
                        reward[k,0] += r
                        if args.debug and k==0 and r:
                            print(" -> 0.5 bomb increase item!")
                    for p_f in pos_f:
                        r = (pos_a == p_f).all().item() * 0.5 * (~done)[k]
                        reward[k,0] += r
                        if args.debug and k==0 and r:
                            print(" > Got flame increase item!")
 

                ##############################################
                ## Debugging: follow what happens on the board
                if args.debug:
                    k=0
                    print("\nold board:\n",old_env_info['board'][k],
                            "\n\nnew board:\n",new_env_info['board'][k],"\n\naction:", 
                            action[k].cpu().item(),"\nreward", reward[k].item())
                    time.sleep(2)
                    print("--------------------------------------------")

                ###############
                ## punish draws
                #idx_draw = torch.as_tensor(done)[:,None] * (reward == 0)
                #reward[idx_draw] -= 0.1

                if args.debug and done[0]:
                    print("############\nGame finished. Last reward:", reward[done].flatten(), "\n############")

                for info in infos:
                    if "episode" in info.keys():
                        episode_rewards.append(info["episode"]["r"])

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
