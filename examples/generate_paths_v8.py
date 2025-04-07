import os
import math
import gymnasium as gym
from acoustorl import MADDPG

from examples.utils.general_utils import *
from examples.utils.top_bottom_setup import top_bottom_setup

# Change from generate_path_2.py and evaluate_planner_v1.py

if __name__ == "__main__":
    n_particles = 8
    model_dirs = [
        './experiments/experiment_20',
        './experiments/experiment_19',
        './experiments/experiment_98',
        './experiments/experiment_99'
    ]
    best_model_number = 1000

    read_dir = os.path.join(model_dirs[0], '20_19_98_99/planner_v2')
    save_dir = os.path.join(model_dirs[0], '20_19_98_99/planner_v11')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = 'path'

    env_name_1 = "acousticlevitationenvironment/PlannerAPF-v0"
    delta_time_1 = 0.096
    max_timesteps_1 = 20
    global_env = gym.make(env_name_1, n_particles=n_particles, delta_time=delta_time_1, max_timesteps=max_timesteps_1)

    env_name_2 = "acousticlevitationenvironment/RePlannerAPF-v0"
    delta_time_2 = 0.096
    max_timesteps_2 = 20
    replan_env = gym.make(env_name_2, n_particles=n_particles, delta_time=delta_time_2, max_timesteps=max_timesteps_2)

    state_dim = global_env.observation_space[0].shape[0]
    action_dim = global_env.action_space[0].shape[0]
    min_action = global_env.action_space[0].low[0]
    max_action = global_env.action_space[0].high[0]  # 动作最大值
    hidden_dim = 64

    agents = []
    for i in range(4):
        agent = MADDPG(
            num_agents = n_particles, 
            state_dims = state_dim, 
            action_dims = action_dim, 
            min_action = min_action,
            max_action = max_action,         
            critic_input_dim = 9*n_particles, 
            hidden_dim = hidden_dim,
        )
        agent.load(best_model_number, model_dirs[i])
        agents.append(agent)

    success_num = 200
    for n in range(200):  
        print(f'-----------------------The {n} th set of paths-----------------------')  
        csv_file = os.path.join(read_dir, f'{file_name}_{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue
        data_numpy, _ = read_paths(csv_data)
        # 每个粒子的轨迹长度相同
        paths_length = int(csv_data[1][1])
        # split_data 的形状为(n_keypoints, n_particles, 3)
        split_data = data_numpy.reshape(-1, paths_length, 5)[:, :, 2:]
        split_data = np.transpose(split_data, [1, 0, 2])

        original_paths, failure = generate_replan_paths(replan_env, agents[0], n_particles, max_timesteps_1, split_data)

        if failure:
            print('1st planner failure!')
            original_paths, failure = generate_replan_paths(replan_env, agents[1], n_particles, max_timesteps_2, original_paths)

            if failure:
                print('2nd planner failure!')
                original_paths, failure = generate_replan_paths(replan_env, agents[2], n_particles, max_timesteps_2, original_paths)

                if failure:
                    print('3rd planner failure!')
                    original_paths, failure = generate_replan_paths(replan_env, agents[3], n_particles, max_timesteps_2, original_paths)

                    if failure:
                        print('4th planner failure!')
                        print('Faliure!/n')
                        success_num -= 1
                        continue

        # 计算时间序列，要求每个片段的最大速度不超过最大速度（0.1m/s）
        # paths: (num_particles, paths_length, 3)
        paths = np.transpose(original_paths, (1, 0, 2))
        # 每个时间段的最大位移
        max_displacements = max_displacement_v2(paths)
        diff_time = max_displacements / 0.1
        # 向上取整为 32.0/10000 的整数倍
        step = 32.0 / 10000
        rounded_diff_time = np.ceil(diff_time / step) * step
        # 计算累计时间并保存
        total_time = np.insert(np.cumsum(rounded_diff_time), 0, 0.0)
        # (paths_length,) -> (num_particles, paths_length, 1)
        total_time_broadcast = np.tile(total_time, (n_particles, 1))[:, :, np.newaxis]
        # 合并时间和路径
        trajectories = np.concatenate((total_time_broadcast, paths), axis=2)
        # 保存修改后的轨迹
        if not failure:
            file_path = os.path.join(save_dir, f'{file_name}_{str(n)}.csv')
        else:
            file_path = os.path.join(save_dir, f'failure_path_{str(n)}.csv')
        save_path_v2(file_path, n_particles, trajectories)   

    print(f'The success number: {success_num}')