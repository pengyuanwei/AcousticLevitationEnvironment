import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing import *


# Modified based on the kinodynamics_analysis_8.py: 任意初始速度的匀加速直线运动计算，zoom时间序列，以确保最高速度<=0.1m/s


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
        
        # 平滑路径
        # 修改第一段的delta_time: 每个segment的dt为确保所有粒子速度小于等于0.1m/s的最大时间
        # 匀加速直线运动，可知：v_max = 2 * s / t
        # 轨迹原为匀速直线运动，有：s = v_max * dt
        # 则有：dt_new =  2 * s / v_max = 20 * s = 20 * (v_max * dt) = 2 * dt
        delta_time[0] *= 2

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
            sub_t, sub_accelerations, sub_velocities, sub_trajectories = uniformly_accelerated_with_arbitrary_initial_velocity(
                split_data[:, i, 2:], split_data[:, i+1, 2:], delta_time[i], dt=32.0/10000, velocities=sub_initial_v
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