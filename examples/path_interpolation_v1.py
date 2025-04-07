import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing_2 import *
from examples.utils.path_smoothing_3 import *

# 将S2M2轨迹按delta t = 0.0032 进行插值，TWGS计算，评估turing points的acoustic trapping quality

if __name__ == '__main__':
    n_particles = 8
    model_dir_1 = './experiments/experiment_20'
    model_name_1 = '20_19_98_99/planner_v2/S2M2'
    num_file = 30
    file_name_0 = 'S2M2_optimized_path'
    file_name_1 = 'path_S2M2'
    save_dir = os.path.join(model_dir_1, model_name_1)

    # TWGS, iterations=5
    levitator_TWGS = top_bottom_setup(n_particles, algorithm='Naive', iterations=5)
    gorkov_1 = []
    gorkov_2 = []

    for n in range(10):
        print(f'\n-----------------------The paths {n}-----------------------')

        csv_file = os.path.join(model_dir_1, model_name_1, f'{file_name_0}_{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue

        data_numpy, _ = read_paths(csv_data)

        # 每个粒子的轨迹长度相同
        paths_length = int(csv_data[1][1])
        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # When axis=2: keypoints_id, time, x, y, z
        split_data = data_numpy.reshape(-1, paths_length, 5)

        # 找出turning points
        turning_points = find_turning_points(split_data[:, :, 1:])
        turning_index_1d = sorted(set(j for (_, j) in turning_points))
        print(turning_points)
        print(turning_index_1d)

        # 计算时间序列，要求每个片段的最大速度不超过最大速度（0.1m/s）
        step = 0.0032
        max_speed = 0.1
        # paths: (num_particles, paths_length, 3)
        paths = split_data[:, :, 2:]
        # 每个时间段的最大位移
        max_displacements = max_displacement_v2(paths)
        # 计算每段路径在 max_speed 下的时间差
        diff_time = max_displacements / max_speed
        # 向上取整为 step 的整数倍
        rounded_diff_time = np.ceil(diff_time / step) * step
        # 计算累计时间并保存
        total_time = np.insert(np.cumsum(rounded_diff_time), 0, 0.0)
        # 精确控制每个时间点为step的整数倍，消除累积误差
        total_time = np.round(total_time / step) * step
        turning_points_time = total_time[turning_index_1d]
        print(total_time)
        print(turning_points_time)

        # (paths_length,) -> (num_particles, paths_length, 1)
        total_time_broadcast = np.tile(total_time, (n_particles, 1))[:, :, np.newaxis]
        # 合并时间和路径
        trajectories = np.concatenate((total_time_broadcast, paths), axis=2)
        print("Original solution:", trajectories.shape)
            
        # 线性插值
        interpolated_trajectories = interpolate_trajectories(trajectories, step)
        print("Interpolated solution:", interpolated_trajectories.shape)

        # 提取时间维度：shape (N, M)
        time_stamps = interpolated_trajectories[0, :, 0]
        # 找到最接近每个 turning_points_time 的 M 维索引
        # 结果索引 shape 应该为 (N, k)
        closest_indices = np.argmin(np.abs(time_stamps[:, None] - turning_points_time[None, :]), axis=0)
        print(closest_indices)

        # 映射原转向点索引 j 到新索引
        j_to_new = dict(zip(turning_index_1d, closest_indices))
        # 给每个 agent 的 j 找到对应的新索引
        agent_new_indices = [(agent_id, j_to_new[j]) for agent_id, j in turning_points]
        # 按 agent_id 排序
        agent_new_indices_sorted = np.array(sorted(agent_new_indices, key=lambda x: x[0]))
        print(agent_new_indices_sorted)

        # Calculate Gorkov
        gorkov_TWGS_1 = levitator_TWGS.calculate_gorkov(interpolated_trajectories[:, :, 1:]).T
        print(gorkov_TWGS_1.shape)
        for i in range(agent_new_indices_sorted.shape[0]):
            print(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]]) 
            gorkov_1.append(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]])


        
        print(f'\n-----------------------Original path---------------------')
        csv_file = os.path.join(model_dir_1, model_name_1, f'{file_name_1}_{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue

        data_numpy, _ = read_paths(csv_data)
        split_data, _ = process_data(data_numpy)
        split_data = np.concatenate((split_data[:, :, 0:1], split_data), axis=2)

        # 找出turning points
        turning_points = find_turning_points(split_data[:, :, 1:])
        turning_index_1d = sorted(set(j for (_, j) in turning_points))
        print(turning_points)
        print(turning_index_1d)

        # 计算时间序列，要求每个片段的最大速度不超过最大速度（0.1m/s）
        step = 0.0032
        max_speed = 0.1
        # paths: (num_particles, paths_length, 3)
        paths = split_data[:, :, 2:]
        # 每个时间段的最大位移
        max_displacements = max_displacement_v2(paths)
        # 计算每段路径在 max_speed 下的时间差
        diff_time = max_displacements / max_speed
        # 向上取整为 step 的整数倍
        rounded_diff_time = np.ceil(diff_time / step) * step
        # 计算累计时间并保存
        total_time = np.insert(np.cumsum(rounded_diff_time), 0, 0.0)
        # 精确控制每个时间点为step的整数倍，消除累积误差
        total_time = np.round(total_time / step) * step
        turning_points_time = total_time[turning_index_1d]
        print(total_time)
        print(turning_points_time)

        # (paths_length,) -> (num_particles, paths_length, 1)
        total_time_broadcast = np.tile(total_time, (n_particles, 1))[:, :, np.newaxis]
        # 合并时间和路径
        trajectories = np.concatenate((total_time_broadcast, paths), axis=2)
        # 线性插值
        interpolated_trajectories = interpolate_trajectories(trajectories, step)

        # 提取时间维度：shape (N, M)
        time_stamps = interpolated_trajectories[0, :, 0]
        # 找到最接近每个 turning_points_time 的 M 维索引
        # 结果索引 shape 应该为 (N, k)
        closest_indices = np.argmin(np.abs(time_stamps[:, None] - turning_points_time[None, :]), axis=0)
        print(closest_indices)

        # 映射原转向点索引 j 到新索引
        j_to_new = dict(zip(turning_index_1d, closest_indices))
        # 给每个 agent 的 j 找到对应的新索引
        agent_new_indices = [(agent_id, j_to_new[j]) for agent_id, j in turning_points]
        # 按 agent_id 排序
        agent_new_indices_sorted = np.array(sorted(agent_new_indices, key=lambda x: x[0]))
        print(agent_new_indices_sorted)

        # Calculate Gorkov
        gorkov_TWGS_1 = levitator_TWGS.calculate_gorkov(interpolated_trajectories[:, :, 1:]).T
        print(gorkov_TWGS_1.shape)
        for i in range(agent_new_indices_sorted.shape[0]):
            print(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]]) 
            gorkov_2.append(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]])

    gorkov_mean_1 = np.mean(gorkov_1)
    gorkov_mean_2 = np.mean(gorkov_2)
    gorkov_std_1 = np.std(gorkov_1)
    gorkov_std_2 = np.std(gorkov_2)    
    
    print(f"优化后的平均gorkov: {gorkov_mean_1}")
    print(f"优化后的gorkov标准差: {gorkov_std_1}")
    print(f"原平均gorkov: {gorkov_mean_2}")
    print(f"原gorkov标准差: {gorkov_std_2}") 