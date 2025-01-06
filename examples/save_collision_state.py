import os
import csv
import math
import torch
import numpy as np
import gymnasium as gym
import time

from utils import *
from acoustorl import MADDPG
from acoustic_levitation_environment_v2.particles import particle_slim, target_slim


# Change from divide_global_path_8.py


if __name__ == "__main__":

    n_particles = 6
    global_model_dir_1 = '/home/william/Projects/acoustic_levitation_environment_v2/examples/experiments/experiment_202'
    global_model_dir_2 = '/home/william/Projects/acoustic_levitation_environment_v2/examples/experiments/experiment_203'
    global_model_dir_3 = '/home/william/Projects/acoustic_levitation_environment_v2/examples/experiments/experiment_204'
    global_model_dir_4 = '/home/william/Projects/acoustic_levitation_environment_v2/examples/experiments/experiment_205'
    best_model_number_1 = 1000
    best_model_number_2 = 1000

    env_name_1 = "acoustic_levitation_environment_v2/GlobalPlanner-v0"
    delta_time_1 = 1.0/10
    max_timesteps_1 = 20
    global_env = gym.make(env_name_1, n_particles=n_particles, delta_time=delta_time_1, max_timesteps=max_timesteps_1)

    env_name_2 = "acoustic_levitation_environment_v2/GlobalRePlanner-v0"
    delta_time_2 = 1.0/10
    max_timesteps_2 = 20
    replan_env = gym.make(env_name_2, n_particles=n_particles, delta_time=delta_time_2, max_timesteps=max_timesteps_2)

    state_dim = global_env.observation_space[0].shape[0]
    action_dim = global_env.action_space[0].shape[0]
    min_action = global_env.action_space[0].low[0]
    max_action = global_env.action_space[0].high[0]  # 动作最大值

    hidden_dim = 64

    main_agent = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    main_agent.load(best_model_number_1, global_model_dir_1)

    replan_agent = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    replan_agent.load(best_model_number_2, global_model_dir_2)

    replan_agent_1 = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    replan_agent_1.load(best_model_number_2, global_model_dir_3)

    replan_agent_2 = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    replan_agent_2.load(best_model_number_2, global_model_dir_4)

    delta_time_3 = 0.1 * math.sqrt(3)
    success_num = 100
    non_collision = 100
    times = []

    collision_state = []

    for n in range(100):  
        start_time = time.time()  # 开始计时
        print(f'-----------------------The {n} th set of paths-----------------------')  

        key_points, failure1 = generate_global_paths(global_env, main_agent, n_particles, max_timesteps_1)
        paths = key_points

        if failure1:
            re_plan_paths, failure2 = generate_replan_paths(replan_env, replan_agent, n_particles, max_timesteps_2, key_points)
            paths = re_plan_paths

            if failure2:
                re_plan_paths_1, failure3 = generate_replan_paths(replan_env, replan_agent_1, n_particles, max_timesteps_2, key_points)
                paths = re_plan_paths_1

                if failure3:
                    re_plan_paths_2, failure4 = generate_replan_paths(replan_env, replan_agent_2, n_particles, max_timesteps_2, key_points)
                    paths = re_plan_paths_2

                    if failure4:
                        success_num -= 1
                        non_collision -= 1

                        collision_state.append(paths[-2, :])
                        continue

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算单次运行时间
        times.append(elapsed_time)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"平均时间: {mean_time:.6f} 秒")
    print(f"标准差: {std_time:.6f} 秒")

    print(f'The success number: {success_num}')
    print(f'The real success number: {non_collision}')

    np.save('./collision_state/collision_state.npy', collision_state)