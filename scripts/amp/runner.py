import os
import psutil

import math
import time
import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import random
from datetime import datetime
from copy import deepcopy

from ruamel.yaml import YAML, dump, RoundTripDumper

from raisim_gym_torch.env.bin.amp import RaisimGymEnv
from raisim_gym_torch.env.bin.amp import NormalSampler

import modules

from raisim_gym_torch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisim_gym_torch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher, RewardLogger

from raisim_gym_torch.algo.ppo.ppo import PPO
from raisim_gym_torch.algo.amp.amp import AMP


def main():
    # task specification
    task_name = "amp"

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
    parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
    args = parser.parse_args()
    mode = args.mode
    weight_path = args.weight

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directories
    home_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../../')
    task_path = home_path + "/gym_envs/" + task_name

    # config
    cfg = YAML().load(open(task_path + "/cfg_amp.yaml", 'r'))

    if cfg['environment']['num_threads'] == 'auto':
        cfg['environment']['num_threads'] = psutil.cpu_count(logical=False)

    # create environment from the configuration file
    env = VecEnv(RaisimGymEnv(home_path + "/resources", dump(cfg['environment'], Dumper=RoundTripDumper)))

    # load expert dataset
    disc_half_ob_dim = 45
    expert_dataset = np.loadtxt(home_path + "/resources/expert_data/expert_data_processed_w_feet.csv", delimiter=",")
    expert_dataset = expert_dataset[:, :disc_half_ob_dim]  # remove initialisation data

    # load style coeff from config
    style_coeff = cfg['environment']['reward']['style']['coeff']

    # Set seed
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    env.seed(cfg['seed'])

    # Reward logger to compute the means and stds of individual reward terms
    reward_logger = RewardLogger(env, cfg, episodes_window=100)

    # Training
    n_steps = cfg['algorithm']['update_steps']
    total_steps = n_steps * env.num_envs

    last_ppo_log_iter = -cfg['environment']['log_interval']

    _actor_critic_module = modules.get_actor_critic_module_from_config(cfg, env, NormalSampler, device)
    actor_critic_module_eval = modules.get_actor_critic_module_from_config(cfg, env, NormalSampler, device)

    _discriminator_module = modules.get_discriminator_from_config(cfg, env, device)

    saver = ConfigurationSaver(log_dir=home_path + "/data/" + task_name,
                               save_items=[task_path + "/cfg_amp.yaml", task_path + "/Environment.hpp"])

    # rick server ip: 129.67.95.58
    tensorboard_launcher(saver.data_dir)  # press refresh (F5) after the first ppo update
    log_dir = os.path.join(saver.data_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    # Discount factor
    rl_gamma = np.exp(cfg['environment']['control_dt'] * np.log(0.5) / cfg['algorithm']['gamma_half_life_duration'])
    learning_rate_decay_gamma = np.exp(np.log(
        cfg['algorithm']['learning_rate']['final'] / cfg['algorithm']['learning_rate']['initial']
    ) / cfg['algorithm']['learning_rate']['decay_steps'])  # x_t = x_0 * gamma ^ t

    if cfg['algorithm']['learning_rate']['mode'] == 'constant':
        learning_rate_decay_gamma = 1.

    ppo = PPO(
        actor_critic_module=_actor_critic_module,
        num_envs=cfg['environment']['num_envs'],
        num_transitions_per_env=n_steps,
        num_learning_epochs=4,
        gamma=rl_gamma,
        lam=0.95,
        num_mini_batches=4,
        writer=writer,
        device=device,
        log_dir=saver.data_dir,
        learning_rate=cfg['algorithm']['learning_rate']['initial'],
        entropy_coef=0.0,
        learning_rate_schedule=cfg['algorithm']['learning_rate']['mode'],
        learning_rate_min=cfg['algorithm']['learning_rate']['min'],
        decay_gamma=learning_rate_decay_gamma
    )

    amp = AMP(
        discriminator=_discriminator_module,
        ppo=ppo,
        num_envs=cfg['environment']['num_envs'],
        num_transitions_per_env=n_steps,
        num_learning_epochs=1,
        mini_batch_size=256,
        expert_dataset=expert_dataset,
        writer=writer,
        device=device
    )

    if mode == 'retrain':
        load_param(weight_path, env, ppo.actor_critic_module, ppo.optimizer, saver.data_dir)

    for update in range(20500):
        start = time.time()

        # If true, only those environments which meet a condition are reset - for example, if max episode length
        # is not reached, the environment will not be reset.
        reset_indices = env.reset(conditional_reset=update != 0)
        ppo.actor_critic_module.reset()

        env.set_max_episode_length(cfg['environment']['max_time'])
        env.enable_early_termination()

        reward_ll_sum = 0
        done_sum = 0

        obs_mean = np.zeros(env.num_obs, dtype=np.float32)
        obs_var = np.zeros(env.num_obs, dtype=np.float32)
        obs_count = 0.0

        visualizable_iteration = False

        if update % cfg['environment']['save_every_n'] == 0:
            print('Storing Actor Critic Module Parameters')
            parameters_save_dict = {'optimizer_state_dict': ppo.optimizer.state_dict()}
            ppo.actor_critic_module.save_parameters(saver.data_dir + "/full_" + str(update) + '.pt',
                                                    parameters_save_dict)
            env.save_scaling(saver.data_dir, str(update))

        if update % cfg['environment']['eval_every_n'] == 0:
            if cfg['environment']['render']:
                visualizable_iteration = True
                print('Visualizing and Evaluating the Current Policy')

                # we create another graph just to demonstrate the save/load method
                actor_critic_module_eval.load_parameters(saver.data_dir + "/full_" + str(update) + '.pt')

                env.turn_on_visualization()

                if cfg['record_video']:
                    env.start_video_recording(saver.data_dir + "/policy_" + str(update) + '.mp4')

                for step in range(math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])):
                    with torch.no_grad():
                        frame_start = time.time()
                        obs = env.observe(False)

                        action_ll = actor_critic_module_eval.generate_action(
                            torch.from_numpy(obs).to(device)).cpu().detach().numpy()

                        reward_ll, dones = env.step(action_ll)
                        actor_critic_module_eval.update_dones(dones)
                        frame_end = time.time()
                        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
                        if wait_time > 0.:
                            time.sleep(wait_time)

                if cfg['record_video']:
                    env.stop_video_recording()

                env.turn_off_visualization()

                env.reset()
                ppo.actor_critic_module.reset()

        # actual training
        for step in range(n_steps):
            obs = env.observe()
            obs_1 = deepcopy(obs)  # make a deepcopy to prevent obs_1 and obs_2 pointing to the same object

            action = ppo.act(obs_1)
            reward, dones = env.step(action)

            # collect second observation
            obs_2 = env.observe(False)

            # compute style rewards
            style_reward = style_coeff * amp.step(obs_1, obs_2)
            reward += style_reward

            ppo.step(rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)

            # Store the rewards for this step
            reward_logger.step(style_reward)

        # take st step to get value obs
        obs = env.observe()

        log_this_iteration = update % cfg['environment']['log_interval'] == 0

        if not visualizable_iteration:
            if update - last_ppo_log_iter > cfg['environment']['log_interval']:
                log_this_iteration = True

            if log_this_iteration:
                last_ppo_log_iter = update

            env.wrapper.getObStatistics(obs_mean, obs_var, obs_count)
            amp.update(log_this_iteration=update % 10 == 0, update=update,
                       obs_mean=obs_mean[:disc_half_ob_dim], obs_var=obs_var[:disc_half_ob_dim])
            ppo.update(obs=obs, log_this_iteration=log_this_iteration, update=update)

            ppo.actor_critic_module.update()
            ppo.actor_critic_module.distribution.enforce_minimum_std((torch.ones(12) * 0.2).to(device))

            # curriculum update. Implement it in Environment.hpp
            env.curriculum_callback()

        ppo.storage.clear()

        average_ll_performance = reward_ll_sum / total_steps
        average_dones = done_sum / total_steps

        # Add to tensorboard
        if log_this_iteration and not visualizable_iteration:
            ppo.writer.add_scalar('Rewards/Episodic/average_ll', average_ll_performance, update)
            reward_logger.log_to_tensorboard(ppo.writer, update)

        # Clear the episodic reward storage buffer
        reward_logger.episodic_reset()

        end = time.time()

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                           * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')


if __name__ == '__main__':
    main()
