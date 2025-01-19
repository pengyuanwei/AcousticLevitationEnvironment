import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.s_curve import *


# Modified based on the kinodynamics_analysis_1.py: 优化代码结构，提高可读性和效率


def calculate_kinematic_quantities(segment, dt_set):
    '''
    输入：
        segment: N个粒子的等长轨迹 (N, lengths, 3)
        dt_set: keypoints 与前一个 keypoint 的时间步长 (lengths, )
    输出：
        t: 时间数组
        velocities: (N, len(t)) 每个粒子随时间的速度
        accelerations: (N, len(t)) 每个粒子随时间的加速度
        trajectories: (N, len(t), 3) 每个粒子的轨迹
    '''
    # 累积时间步长以获取时间数组
    t = np.cumsum(dt_set)

    # 计算轨迹
    trajectories = segment  # 轨迹已经由输入提供，直接赋值

    # 计算速度
    # 使用中央差分法计算速度：v[i] = (x[i+1] - x[i-1]) / (2 * dt)
    velocities = np.zeros((segment.shape[0], segment.shape[1], 3))
    for i in range(segment.shape[0]):
        for j in range(1, segment.shape[1] - 1):
            dt = dt_set[j] + dt_set[j - 1]
            velocities[i, j] = (segment[i, j + 1] - segment[i, j - 1]) / dt
        # 处理边界情况
        velocities[i, 0] = (segment[i, 1] - segment[i, 0]) / dt_set[0]
        velocities[i, -1] = (segment[i, -1] - segment[i, -2]) / dt_set[-1]

    # 计算加速度
    # 使用中央差分法计算加速度：a[i] = (v[i+1] - v[i-1]) / (2 * dt)
    accelerations = np.zeros((segment.shape[0], segment.shape[1], 3))
    for i in range(segment.shape[0]):
        for j in range(1, segment.shape[1] - 1):
            dt = dt_set[j] + dt_set[j - 1]
            accelerations[i, j] = (velocities[i, j + 1] - velocities[i, j - 1]) / dt
        # 处理边界情况
        accelerations[i, 0] = (velocities[i, 1] - velocities[i, 0]) / dt_set[0]
        accelerations[i, -1] = (velocities[i, -1] - velocities[i, -2]) / dt_set[-1]

    return t, velocities, accelerations, trajectories


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
        
        # 对第一段和最后一段进行插值处理
        # 修改第一段和最后一段的dt: 每个segment的dt为确保所有粒子速度小于等于0.1m/s的最大时间
        # 匀加速直线运动，可知：v_max = 2 * s / t
        # 轨迹原为匀速直线运动，有：s = v_max * dt
        # 则有：dt_new =  2 * s / v_max = 20 * s = 20 * (v_max * dt) = 2 * dt
        delta_time[0] *= 2
        delta_time[-1] *= 2

        # 处理第一个片段
        start = split_data[:, 0, 2:]
        end = split_data[:, 1, 2:]
        total_time = delta_time[0]
        dt = 32.0/10000

        # 粒子数量
        N = start.shape[0]

        # 计算路径长度和方向
        L = np.linalg.norm(end - start, axis=1)  # (N,) 每个粒子的总路径长度
        direction = (end - start) / L[:, np.newaxis]  # (N, 3) 单位方向向量

        # 加速度和最大速度
        a = 2 * L / total_time ** 2  # (N,)
        v_max = 2 * L / total_time  # (N,)

        # 时间数组
        t = np.arange(0, total_time, dt)
        num_steps = len(t)

        # 初始化结果数组
        accelerations = np.zeros((N, num_steps))
        velocities = np.zeros((N, num_steps))
        positions = np.zeros((N, num_steps))

        # 时间点对应的加速度、速度和位移
        for i, ti in enumerate(t):
            if ti <= total_time:
                # 匀加速阶段
                v = a * ti  # (N,)
                s = (1/2) * a * ti**2  # (N,)
            else:
                v = v_max  # 最大速度保持
                s = L  # 终点

            accelerations[:, i] = a
            velocities[:, i] = v
            positions[:, i] = s

        # 计算三维轨迹
        trajectories = positions[:, :, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

        # 可视化所有粒子
        visualize_all_particles(t, accelerations, velocities, trajectories)


        # # 保留初始时间点为0，补齐成与原数据相同的形状
        # t_set = np.concatenate([[0.0], delta_time], axis=0)

        # new_t_set = np.zeros(paths_length + 18)
        # # 前10个元素设置为 t_set[1] / 10
        # new_t_set[1:11] = t_set[1] / 10
        # # 后10个元素设置为 t_set[-1] / 10
        # new_t_set[-10:] = t_set[-1] / 10
        # # 中间部分赋值为 t_set 的第2个到倒数第2个元素
        # new_t_set[11:-10] = t_set[2:-1]
        # #print(new_t_set)


        # # 对坐标key_points进行插值处理
        # key_points = np.transpose(split_data[:, :, 2:], (1, 0, 2))
        # new_key_points = np.zeros((paths_length + 18, n_particles, 3))
        # new_key_points[0] = key_points[0]
        # new_key_points[11:-10] = key_points[2:-1]


        # coordinate_changes_0 = (key_points[1] - key_points[0]) / 100
        # for i in range(1, 11):
        #     new_key_points[i] = key_points[0] + coordinate_changes_0 * (i**2)

        # coordinate_changes_1 = (key_points[-1] - key_points[-2]) / 100
        # for i in range(1, 11):
        #     new_key_points[-11+i] = key_points[-2] + coordinate_changes_1 * (20 * i - i**2)


        # new_split_data_numpy = np.zeros((n_particles, paths_length + 18, 5))
        # new_split_data_numpy[:, :, 0] = np.arange(paths_length + 18)
        # new_split_data_numpy[:, :, 1] = new_t_set
        # new_split_data_numpy[:, :, 2:] = np.transpose(new_key_points, (1, 0, 2))


        # # 计算速度和加速度曲线
        # t, velocities, accelerations, trajectories = calculate_kinematic_quantities(
        #     new_split_data_numpy[:, :10, 2:], new_t_set[:10]
        # )
        # # 可视化所有粒子
        # visualize_all_particles(t, accelerations, velocities, trajectories)


        # # # 保存修改后的轨迹
        # # file_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        # # save_path_v2(file_path, n_particles, new_split_data_numpy)