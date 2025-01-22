import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing import *


# Modified based on the kinodynamics_analysis_3.py: 任意初始速度的S曲线速度计算


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

        # # 计算每个粒子每一段的位移和速度
        # displacements = calculate_displacements(split_data[:, :, 2:])  # (num_particle, n_keypoints - 1)
        # dx = np.transpose(displacements, (1, 0))  # (n_keypoints - 1, n_particles)
        # velocities_set = calculate_velocities(dx, delta_time)


        # 处理第一段直线路径 ################################################################################
        # 修改第一段的delta_time: 确保所有粒子的速度均<=0.1m/s
        # S曲线变加速直线运动，可知：v_max = 2 * s / t
        # 轨迹原为匀速直线运动，有：s = v_max * dt
        # 因此：dt_new = 2 * s / v_max = 20 * s = 20 * (v_max * dt) = 2 * dt
        delta_time[0] *= 2

        # 计算速度S曲线
        t, accelerations, velocities, trajectories = smooth_acceleration_trajectories(
            split_data[:, 0, 2:], split_data[:, 1, 2:], delta_time[0], dt=32.0/10000
        )


        # 处理第二段直线路径 ################################################################################
        start = split_data[:, 1, 2:]
        end = split_data[:, 2, 2:]
        total_time = delta_time[1]
        dt = 32.0/10000
        # 粒子数量
        N = start.shape[0]

        # 计算路径长度和方向
        L = np.linalg.norm(end - start, axis=1)  # (N,) 每个粒子的总路径长度
        direction = (end - start) / L[:, np.newaxis]  # (N, 3) 单位方向向量

        # 初速度
        v_0 = velocities[:, -1]
        # 最大加速度和末速度
        a_max = 4 * (L - v_0 * total_time) / total_time ** 2  # (N,)
        v_2 = v_0 + 2 * (L - v_0 * total_time) / total_time  # (N,)

        # 时间数组
        t = np.arange(0, total_time, dt)
        num_steps = len(t)

        # 初始化结果数组
        accelerations = np.zeros((N, num_steps))
        velocities = np.zeros((N, num_steps))
        positions = np.zeros((N, num_steps))

        # 时间点对应的加速度、速度和位移
        for i, ti in enumerate(t):
            if ti <= (total_time / 2):
                # 加速阶段前半部分
                a = (2 * a_max * ti) / total_time  # (N,)
                v = v_0 + (a_max * ti**2) / total_time  # (N,)
                s = v_0 * ti + (1/3) * (a_max * ti**3) / total_time  # (N,)
            elif ti <= total_time:
                # 加速阶段后半部分
                a = (-2 * a_max * ti) / total_time + 2 * a_max  # (N,)
                v = v_0 - (a_max * ti**2) / total_time + 2 * a_max * ti - (1/2) * a_max * total_time  # (N,)
                s = v_0 * ti - (1/3) * (a_max * ti**3) / total_time + a_max * ti**2 - (1/2) * a_max * total_time * ti + (1/12) * a_max * total_time**2  # (N,)
            else:
                a = np.zeros(N)  # 全部置为 0
                v = v_2
                s = L  # 终点

            accelerations[:, i] = a
            velocities[:, i] = v
            positions[:, i] = s

        # 计算三维轨迹
        trajectories = positions[:, :, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

        # 可视化所有粒子
        visualize_all_particles(t, accelerations, velocities, trajectories)


        # # 保存修改后的轨迹
        # file_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        # save_path_v2(file_path, n_particles, new_split_data_numpy)