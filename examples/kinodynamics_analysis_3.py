import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.s_curve import *


# Modified based on the kinodynamics_analysis_2.py: integrate velocity S-curve


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99'
    num_file = 1
    file_name_0 = 'optimised_data'
    file_name_1 = 'optimised_1_data'

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
        # split_data_numpy的形状为(n_particles, paths_length, 5)
        # When axis=2: keypoints_id, time, x, y, z
        split_data = data_numpy.reshape(-1, paths_length, 5)

        # 计算时间变化量（差分）
        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        delta_time = np.diff(split_data[0, :, 1], axis=0)

        # 对第一段和最后一段进行插值处理
        # 修改第一段和最后一段的dt: 每个segment的dt为确保所有粒子速度小于等于0.1m/s的最大时间
        # 变加速直线运动，已知：dt_new = 20 * s, 则有：dt_new = 20 * (0.1 * dt) = 2 * dt
        delta_time[0] *= 2
        delta_time[-1] *= 2

        # 计算速度S曲线
        t, accelerations, velocities, trajectories = smooth_acceleration_trajectories(
            split_data[:, 0, 2:], split_data[:, 1, 2:], delta_time[0], dt=32.0/10000
        )
        # 可视化所有粒子
        visualize_all_particles(t, accelerations, velocities, trajectories)


        # # # 保存修改后的轨迹
        # # file_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        # # save_path_v2(file_path, n_particles, new_split_data_numpy)