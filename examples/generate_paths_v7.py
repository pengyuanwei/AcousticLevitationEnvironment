import os
import math
import gymnasium as gym
from acoustorl import MADDPG

from examples.utils.general_utils_v2 import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing_2 import *
from examples.utils.top_bottom_setup import top_bottom_setup

'''
Change from generate_path_v6.py: 输入起点终点
planner_v5: delta t 0.192 且 search domain 全简化为position为圆心的区域内
planner_v6: max Gorkov 超过阈值则重新搜索
planner_v7: 同planner_v5, condidate_solutions 变为100，delta t 变为 0.32
planner_v8: 无Gorkov优化
planner_v9: 同planner_v5, condidate_solutions 变为100，delta t 变为 0.192
'''

if __name__ == "__main__":
    n_particles = 8
    global_model_dir_1 = 'D:/PythonProjects/AcousticLevitationEnvironment/examples/experiments/experiment_20'
    global_model_dir_2 = 'D:/PythonProjects/AcousticLevitationEnvironment/examples/experiments/experiment_19'
    global_model_dir_3 = 'D:/PythonProjects/AcousticLevitationEnvironment/examples/experiments/experiment_98'
    global_model_dir_4 = 'D:/PythonProjects/AcousticLevitationEnvironment/examples/experiments/experiment_99'
    best_model_number_1 = 1000
    best_model_number_2 = 1000

    read_dir = os.path.join(global_model_dir_1, '20_19_98_99/planner_v3')
    save_dir = os.path.join(global_model_dir_1, '20_19_98_99/planner_v9')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = 'path'

    # TWGS, iterations=5
    levitator = top_bottom_setup(n_particles, algorithm='TWGS', iterations=5)

    env_name_1 = "acousticlevitationenvironment/PlannerAPFGorkov-v0"
    delta_time = 0.192
    max_timesteps = 15
    global_env = gym.make(env_name_1, n_particles=n_particles, delta_time=delta_time, max_timesteps=max_timesteps, levitator=levitator)

    env_name_2 = "acousticlevitationenvironment/RePlannerAPFGorkov-v0"
    replan_env = gym.make(env_name_2, n_particles=n_particles, delta_time=delta_time, max_timesteps=max_timesteps, levitator=levitator)

    env_name_3 = "acousticlevitationenvironment/RePlannerAPF-v2"
    nonAcoustic_replan_env = gym.make(env_name_3, n_particles=n_particles, delta_time=delta_time, max_timesteps=max_timesteps)

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
        agent.load(best_model_number_1, global_model_dir_1)
        agents.append(agent)

    debug = False
    num_file = 100
    success_num = 200
    for n in range(80):
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


        original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(replan_env, agents[0], n_particles, max_timesteps, split_data, levitator)     

        if failure:
            print('1st planner failure!')
            original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(replan_env, agents[1], n_particles, max_timesteps, original_paths, levitator)

            if failure:
                print('2nd planner failure!')
                original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(replan_env, agents[2], n_particles, max_timesteps, original_paths, levitator)

                if failure:
                    print('3rd planner failure!')
                    original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(replan_env, agents[3], n_particles, max_timesteps, original_paths, levitator)

                    if failure:
                        print('4th planner failure!')
                        original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(nonAcoustic_replan_env, agents[0], n_particles, max_timesteps, original_paths)

                        if failure:
                            print('5th planner failure!')
                            original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(nonAcoustic_replan_env, agents[1], n_particles, max_timesteps, original_paths)

                            if failure:
                                print('6th planner failure!')
                                original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(nonAcoustic_replan_env, agents[2], n_particles, max_timesteps, original_paths)

                                if failure:
                                    print('7th planner failure!')
                                    original_paths, last_unique_indexs, fixed_locations, failure = generate_replan_paths_smoothing(nonAcoustic_replan_env, agents[3], n_particles, max_timesteps, original_paths)

                                    if failure:
                                        print('8th planner failure!')
                                        success_num -= 1

        if not failure:
            # 计算时间序列，要求每个片段的最大速度不超过最大速度（0.1m/s）
            # (num_particles, paths_length, 3)
            paths = np.transpose(original_paths, (1, 0, 2))
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
            file_path = os.path.join(save_dir, f'{file_name}_{str(n)}.csv')

        # 保存失败轨迹
        else:
            # 计算时间序列，要求每个片段的最大速度不超过最大速度（0.1m/s）
            # (num_particles, paths_length, 3)
            paths = np.transpose(original_paths, (1, 0, 2))
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
            file_path = os.path.join(save_dir, f'failure_path_{str(n)}.csv')
        
        save_path_v2(file_path, n_particles, trajectories)            

    print(f'\nThe success number: {success_num}')