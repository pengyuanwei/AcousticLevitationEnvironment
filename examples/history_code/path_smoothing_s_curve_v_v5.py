import os
import numpy as np

from examples.utils.general_utils import *
from examples.utils.path_smoothing_3 import *


# Modified based on the path_smoothing_s_curve_v_v4.py: 消除速度突变: 根据sum max a优化


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v2'
    num_file = 30
    file_name_0 = 'path'
    file_name_1 = 'new_smoothed_path'

    computation_time = []
    for n in range(num_file):
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
        
        # 平滑路径
        # 修改第一段和最后一段的delta_time: 每个segme nt的dt为确保所有粒子速度小于等于0.1m/s的最大时间
        # 匀加速直线运动，可知：v_max = 2 * s / t
        # 轨迹原为匀速直线运动，有：s = v_max * dt
        # 则有：dt_new =  2 * s / v_max = 20 * s = 20 * (v_max * dt) = 2 * dt
        delta_time[0] *= 2
        delta_time[-1] *= 2

        original_max_a = kinodynamics_analysis(n_particles, split_data, delta_time)
        print(np.sum(original_max_a))

        time_factor = [1.25, 1.5, 1.75, 2.0]
        sum_max_a = np.zeros((len(time_factor), delta_time.shape[0]-2))
        delta_v = np.zeros((len(time_factor), delta_time.shape[0]-2, 2))
        for i in range(len(time_factor)):
            for j in range(1, delta_time.shape[0]-1):
                delta_time[j] *= time_factor[i]
                max_a = kinodynamics_analysis(n_particles, split_data, delta_time)
                sum_max_a[i][j-1] = np.sum(max_a)
                diff_a = max_a-original_max_a
                non_zeros_index = np.nonzero(abs(diff_a) > 1e-6)
                delta_v[i][j-1] = diff_a[non_zeros_index]
                delta_time[j] /= time_factor[i]

        # 找出最小元素的索引
        min_index = np.unravel_index(np.argmin(sum_max_a), sum_max_a.shape)    
        if np.all(delta_v[min_index] < 0.0):
            print('The new sum acceleration:', sum_max_a[min_index])
            delta_time[min_index[1]+1] *= time_factor[min_index[0]]
            max_a, sum_t, sum_traj = kinodynamics_analysis(n_particles, split_data, delta_time, save=True)
            print('The new sum acceleration:', np.sum(max_a))

            final_traj = np.zeros((sum_traj.shape[0], sum_traj.shape[1], 5))
            final_traj[:, :, 0] = np.arange(sum_traj.shape[1])
            final_traj[:, :, 1] = sum_t
            final_traj[:, :, 2:] = sum_traj

            # 保存修改后的轨迹
            file_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
            save_path_v2(file_path, n_particles, final_traj)

        else:
            print(f'Path {n} cannot be further optimized!')