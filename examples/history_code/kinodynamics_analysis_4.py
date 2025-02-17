import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing import *


# Modified based on the kinodynamics_analysis_2.py: keypoints之间进行匀速直线运动，第一段的匀速直线


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

        # 处理第一个片段：对第一段进行插值处理
        start = split_data[:, 0, 2:]
        end = split_data[:, 1, 2:]
        dt = 32.0/10000

        # 修改第一段的delta_time
        # 在理想模型中，存在一个瞬间的脉冲加速度。然而，在现实中，速度变化需要一个有限时间dt。
        # 理想: s = v_max * T
        # 现实: s = 0.5 * a * dt**2 + v_max * (T' - dt)
        # 则有: T' = T + 0.5 * dt
        delta_time[0] += (1/2) * dt
        total_time = delta_time[0]

        # 粒子数量
        N = start.shape[0]

        # 计算路径长度和方向
        L = np.linalg.norm(end - start, axis=1)  # (N,) 每个粒子的总路径长度
        direction = (end - start) / L[:, np.newaxis]  # (N, 3) 单位方向向量

        # 末速度，瞬时加速度，加速阶段位移
        v_max = L / (total_time - (1/2) * dt)  # (N,)
        a_max = L / ((total_time - (1/2) * dt) * dt)  # (N,)
        s_1 = (1/2) * a_max * dt**2

        # 时间数组
        t = np.arange(0, total_time, dt)
        num_steps = len(t)

        # 初始化结果数组
        accelerations = np.zeros((N, num_steps))
        velocities = np.zeros((N, num_steps))
        positions = np.zeros((N, num_steps))

        # 时间点对应的加速度、速度和位移
        for i, ti in enumerate(t):
            if i == 0:
                # 静止阶段
                v = 0.0
                a = 0.0
                s = 0.0
            elif i == 1:
                # 瞬时加速
                v = v_max
                a = a_max
                s = s_1             
            else:
                # 匀速阶段
                v = v_max  # (N,)
                a = 0.0
                s = s_1 + v_max * (ti - (1/2) * dt)  # (N,)

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