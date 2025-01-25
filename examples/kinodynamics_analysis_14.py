import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing_2 import *


# Modified based on the kinodynamics_analysis_13.py: 从现实的离散轨迹入手，实现开头结尾的S曲线速度插值


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
        # 修改第一段和最后一段的delta_time: 每个segment的dt为确保所有粒子速度小于等于0.1m/s的最大时间
        # 匀加速直线运动，可知：v_max = 2 * s / t
        # 轨迹原为匀速直线运动，有：s = v_max * dt
        # 则有：dt_new =  2 * s / v_max = 20 * s = 20 * (v_max * dt) = 2 * dt
        delta_time[0] *= 2
        delta_time[-1] *= 2

        # 初始化
        t = []
        accelerations = []
        velocities = []
        trajectories = []

        dt = 32.0/10000
        sub_initial_t = 0.0
        sub_initial_v = np.zeros((n_particles,))


        # 第一段匀加速
        sub_t, _, _, sub_trajectories = smooth_trajectories_arbitrary_initial_velocity(
            split_data[:, 0, 2:], split_data[:, 1, 2:], delta_time[0], dt=dt, velocities=sub_initial_v
        )

        sub_t += sub_initial_t
        sub_initial_t = sub_t[-1] + dt

        t.append(sub_t)
        trajectories.append(sub_trajectories)  

        # 中间段匀速直线
        for i in range(1, split_data.shape[1]-2):
            sub_t, sub_trajectories, sub_initial_v = uniform_velocity_interpolation_simple(
                start=split_data[:, i, 2:], end=split_data[:, i+1, 2:], total_time=delta_time[i], dt=dt, velocities=sub_initial_v
            )

            sub_t += sub_initial_t
            sub_initial_t = sub_t[-1] + dt

            t.append(sub_t)
            trajectories.append(sub_trajectories)  

        # 最后一段匀减速
        sub_t, _, _, sub_trajectories = s_curve_smoothing_with_zero_end_velocity(
            split_data[:, -2, 2:], split_data[:, -1, 2:], delta_time[0], dt=dt, velocities=sub_initial_v
        )
        
        sub_t += sub_initial_t
        sub_initial_t = sub_t[-1] + dt

        t.append(sub_t)
        trajectories.append(sub_trajectories)

        # 将所有子数组沿 axis=1 拼接成一个总数组
        sum_t = np.concatenate(t, axis=0)
        sum_traj = np.concatenate(trajectories, axis=1)

        displacements = np.zeros((sum_traj.shape[0], sum_traj.shape[1]))
        displacements[:, 1:] = np.linalg.norm(sum_traj[:, 1:, :] - sum_traj[:, :-1, :], axis=2)  # (N,) 每个粒子的总路径长度

        visualize_lengths(sum_t, displacements)


        # # final_traj = np.zeros((sum_traj.shape[0], sum_traj.shape[1], 5))
        # # final_traj[:, :, 0] = np.arange(sum_traj.shape[1])
        # # final_traj[:, :, 1] = sum_t
        # # final_traj[:, :, 2:] = sum_traj

        # # # 保存修改后的轨迹
        # # file_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        # # save_path_v2(file_path, n_particles, final_traj)