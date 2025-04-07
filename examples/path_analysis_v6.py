import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing_2 import *
from examples.utils.path_smoothing_3 import *

# Modified based on the path_analysis_v5.py: 评估turing points的acoustic trapping quality

if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name_1 = '20_19_98_99/planner_v7'
    model_name_2 = '20_19_98_99/planner_v9'
    num_file = 30
    file_name_0 = 'smoothed_path'
    file_name_1 = 'new_smoothed_path'

    # TWGS, iterations=5
    levitator_TWGS = top_bottom_setup(n_particles, algorithm='TWGS', iterations=5)
    levitator_WGS = top_bottom_setup(n_particles, algorithm='Naive', iterations=5)


    computation_time = []
    for n in range(10):
        print(f'\n-----------------------The paths {n}-----------------------')

        csv_file = os.path.join(global_model_dir_1, model_name_1, f'{file_name_0}_{str(n)}.csv')
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
        print(split_data.shape)

        # 计算时间变化量（差分）
        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        delta_time = np.diff(split_data[0, :, 1], axis=0)

        velocities, accelerations = calculate_v_a(split_data[:, :, 2:])
        print(velocities.shape, accelerations.shape)


        # # 由于垂直方向上的F_max是水平方向的大约6倍，给xy方向的加速度6倍的权重以提高其影响
        # accelerations[:, :, :2] *= 6.0
        # sum_a = np.linalg.norm(accelerations, axis=2)
        # print(sum_a.shape)

        # # 在时间序列尾部添加一个累计时间以匹配速度序列的长度
        # accumulative_time = split_data[0, :, 1]
        # last_time = accumulative_time[-1] + 0.0032
        # accumulative_time_1 = np.append(accumulative_time, last_time)
        # # 可视化速度
        # # visualize_all_particles_v3(accumulative_time, sum_a)

        # # Calculate Gorkov
        # # 给paths末尾添加一个frame以匹配sum_a
        # paths = np.concatenate((split_data[:, :, 2:], split_data[:, -1:, 2:]), axis=1)
        # gorkov_TWGS_1 = levitator_TWGS.calculate_gorkov(paths)


        # csv_file = os.path.join(global_model_dir_1, model_name_2, f'{file_name_0}_{str(n)}.csv')
        # #csv_file = 'F:\Desktop\Projects\AcousticLevitationGym\examples\experiments\S2M2_8_experiments\data0.csv'
        # csv_data = read_csv_file(csv_file)
        # if csv_data == None:
        #     print(f"Skipping file due to read failure: {csv_file}")
        #     continue

        # data_numpy, _ = read_paths(csv_data)

        # # 每个粒子的轨迹长度相同
        # paths_length = int(csv_data[1][1])
        # # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # # When axis=2: keypoints_id, time, x, y, z
        # split_data = data_numpy.reshape(-1, paths_length, 5)
        # print(split_data.shape)

        # # 计算时间变化量（差分）
        # # split_data_numpy[:,:,1] 是时间累加值（时间列）
        # delta_time = np.diff(split_data[0, :, 1], axis=0)

        # velocities, accelerations = calculate_v_a(split_data[:, :, 2:])
        # # 由于垂直方向上的F_max是水平方向的大约6倍，给xy方向的加速度6倍的权重以提高其影响
        # accelerations[:, :, :2] *= 6.0
        # sum_a = np.linalg.norm(accelerations, axis=2)
        # print(sum_a.shape)

        # # 在时间序列尾部添加一个累计时间以匹配速度序列的长度
        # accumulative_time = split_data[0, :, 1]
        # last_time = accumulative_time[-1] + 0.0032
        # accumulative_time_2 = np.append(accumulative_time, last_time)
        # # 可视化速度
        # # visualize_all_particles_v3(accumulative_time, sum_a)

        # # Calculate Gorkov
        # # 给paths末尾添加一个frame以匹配sum_a
        # paths = np.concatenate((split_data[:, :, 2:], split_data[:, -1:, 2:]), axis=1)
        # gorkov_TWGS_2 = levitator_TWGS.calculate_gorkov(paths)

        # # 可视化速度
        # visualize_all_particles_v4(accumulative_time_1, gorkov_TWGS_1.T, accumulative_time_2, gorkov_TWGS_2.T)