import os
import csv
import math
import torch
import numpy as np
import gymnasium as gym

from examples.general_utils import *
from acoustorl import MADDPG
from acousticlevitationenvironment.utils import general_utils


if __name__ == "__main__":

    global_model_dir_1 = './experiments/experiment_20'
    global_model_dir_2 = './experiments/experiment_19'
    local_model_dir_1 = './experiments/experiment_14'
    local_model_dir_2 = './experiments/experiment_30'
    best_model_number_1 = 1000
    best_model_number_2 = 1000
    best_model_number_3 = 1000
    best_model_number_4 = 1000

    n_particles = 8
    env_name_1 = "acousticlevitationgym/GlobalPlanner-v0"
    delta_time_1 = 1.0/10
    max_timesteps_1 = 20
    global_env = gym.make(env_name_1, n_particles=n_particles, delta_time=delta_time_1, max_timesteps=max_timesteps_1)

    env_name_2 = "acousticlevitationgym/GlobalRePlanner-v0"
    delta_time_2 = 1.0/10
    max_timesteps_2 = 20
    replan_env = gym.make(env_name_2, n_particles=n_particles, delta_time=delta_time_2, max_timesteps=max_timesteps_2)

    env_name_3 = "acousticlevitationgym/LocalPlanner-v0"
    delta_time_3 = 1.0/50
    max_timesteps_3 = 10
    local_env = gym.make(env_name_3, n_particles=n_particles, delta_time=delta_time_2, max_timesteps=max_timesteps_2)

    state_dim = global_env.observation_space[0].shape[0]
    action_dim = global_env.action_space[0].shape[0]
    min_action = global_env.action_space[0].low[0]
    max_action = global_env.action_space[0].high[0]  # 动作最大值

    hidden_dim = 64

    global_agent = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    global_agent.load(best_model_number_1, global_model_dir_1)

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

    success_num = 1000
    non_collision = 1000
    for n in range(success_num):  
        print(f'-----------------------The {n} th set of paths-----------------------')  

        print(f'-----------------------Global Planning----------------------')  

        key_points, failure1 = generate_global_paths(global_env, global_agent, n_particles, max_timesteps_1)

        if failure1:
            #success_num -= 1
            #non_collision -= 1
            print(f'-----------------------Re-Planning-----------------------')

            re_plan_paths, failure2 = generate_replan_paths(replan_env, replan_agent, n_particles, max_timesteps_1, key_points)

            if failure2:
                success_num -= 1
                non_collision -= 1
            else:            
                for k in range(len(re_plan_paths)-1):
                    segment = re_plan_paths[k:(k+2)]
                    interpolated_coords = interpolate_positions(segment)

                    for i in range(10):
                        collision = safety_area(n_particles, interpolated_coords[i])
                        if not np.all(collision == 0):
                            non_collision -= 1
                            break
                    if not np.all(collision == 0):
                        break

        else:            
            for k in range(len(key_points)-1):
                segment = key_points[k:(k+2)]
                interpolated_coords = interpolate_positions(segment)

                for i in range(10):
                    collision = safety_area(n_particles, interpolated_coords[i])
                    if not np.all(collision == 0):
                        non_collision -= 1
                        break
                if not np.all(collision == 0):
                    break


    print(f'The success number: {success_num}')
    print(f'The real success number: {non_collision}')