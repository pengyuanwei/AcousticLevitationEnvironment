import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing import *


# Modified based on the kinodynamics_analysis_4.py: keypoints之间进行匀速直线运动，除最后一段的匀速直线


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
        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # When axis=2: keypoints_id, time, x, y, z
        split_data = data_numpy.reshape(-1, paths_length, 5)

        # 计算时间变化量（差分）
        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        delta_time = np.diff(split_data[0, :, 1], axis=0)

        # 初始化
        t = []
        accelerations = []
        velocities = []
        trajectories = []

        dt = 32.0/10000
        sub_initial_t = 0.0
        sub_initial_v = np.zeros((8,))

        # 计算速度S曲线
        for i in range(split_data.shape[1]-1):
            sub_t, sub_accelerations, sub_velocities, sub_trajectories = uniform_velocity_interpolation(
                start=split_data[:, i, 2:], end=split_data[:, i+1, 2:], total_time=delta_time[i], dt=dt, velocities=sub_initial_v
            )

            sub_t += sub_initial_t
            sub_initial_t = sub_t[-1] + dt
            sub_initial_v = sub_velocities[:, -1]

            t.append(sub_t)
            accelerations.append(sub_accelerations)
            velocities.append(sub_velocities)
            trajectories.append(sub_trajectories)            

        # 将所有子数组沿 axis=1 拼接成一个总数组
        sum_t = np.concatenate(t, axis=0)
        sum_a = np.concatenate(accelerations, axis=1)
        sum_v = np.concatenate(velocities, axis=1)
        sum_traj = np.concatenate(trajectories, axis=1)

        print(sum_t.shape)
        print(sum_a.shape)

        sum_jerk = calculate_jerk(sum_t, sum_a)
        zero_array = np.zeros((sum_jerk.shape[0], 1))
        print(zero_array.shape)
        sum_jerk = np.concatenate([zero_array, sum_jerk], axis=1)
        print(sum_jerk.shape)


        # 可视化所有粒子
        visualize_all_particles(sum_t, sum_a, sum_v, sum_traj, jerks=sum_jerk)


        # # 保存修改后的轨迹
        # file_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        # save_path_v2(file_path, n_particles, new_split_data_numpy)