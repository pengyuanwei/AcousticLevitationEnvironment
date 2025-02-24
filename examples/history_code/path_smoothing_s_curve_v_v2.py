import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing_2 import *


# Modified based on the path_smoothing_s_curve_v_v1.py: 从现实的离散轨迹入手，实现开头结尾的S曲线速度插值


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v2'
    num_file = 1
    file_name_0 = 'path'
    file_name_1 = 'smoothed_path'

    levitator = top_bottom_setup(n_particles)

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

        displacements = np.zeros((split_data.shape[0], split_data.shape[1]))
        displacements[:, 1:] = np.linalg.norm(split_data[:, 1:, 2:] - split_data[:, :-1, 2:], axis=2)  # (N,) 每个粒子的总路径长度
        visualize_lengths(split_data[0, :, 1], displacements)

        new_time, new_trajectories = uniform_dt_interpolation(split_data[:, :, 2:], delta_time)
        displacements = np.zeros((new_trajectories.shape[0], new_trajectories.shape[1]))
        displacements[:, 1:] = np.linalg.norm(new_trajectories[:, 1:, :] - new_trajectories[:, :-1, :], axis=2)  # (N,) 每个粒子的总路径长度
        visualize_lengths(new_time, displacements)