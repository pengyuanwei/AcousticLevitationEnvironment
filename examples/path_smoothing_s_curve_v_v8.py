import os
import numpy as np
import matplotlib.pyplot as plt

from examples.utils.general_utils_v2 import *
from examples.utils.path_smoothing_3 import *


# Modified based on the path_smoothing_s_curve_v_v7.py: 只优化两端


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v2/S2M2'
    num_file = 200
    file_name_0 = 'S2M2_optimized_path'
    file_name_1 = 'S2M2_optimized_path_v2'

    for n in range(100):
        print(f'\n-----------------------The paths {n}-----------------------')

        csv_file = os.path.join(global_model_dir_1, model_name, f'{file_name_0}_{str(n)}.csv')
        #csv_file = 'F:\Desktop\Projects\AcousticLevitationGym\examples\experiments\S2M2_8_experiments\data0.csv'
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

        # 计算时间变化量（差分）
        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        delta_time = np.diff(split_data[0, :, 1], axis=0)
        print(delta_time)
        
        # 计算时间序列，要求每个片段的最大速度不超过最大速度（0.1m/s）
        # 每个时间段的最大位移
        max_displacements = max_displacement_v2(split_data[:, :, 2:])
        diff_time = max_displacements / 0.1
        max_v = max_displacements / delta_time
        print('Original max v:', max_v)

        # # 可视化
        # _, _, sum_t, sum_traj = kinodynamics_analysis_v2(n_particles, split_data, delta_time, save=True)

        clipped_max_v = np.clip(max_v, None, 0.1)
        print('Clipped max v:', clipped_max_v)
        delta_time = max_displacements / clipped_max_v

        # # 可视化
        # _, _, sum_t, sum_traj = kinodynamics_analysis_v2(n_particles, split_data, delta_time, save=True)


        # 计算累计时间并保存
        total_time = np.insert(np.cumsum(delta_time), 0, 0.0)
        # (paths_length,) -> (num_particles, paths_length, 1)
        total_time_broadcast = np.tile(total_time, (n_particles, 1))[:, :, np.newaxis]
        # 合并时间和路径
        trajectories = np.concatenate((total_time_broadcast, split_data[:, :, 2:]), axis=2)

        file_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        save_path_v2(file_path, n_particles, trajectories)  