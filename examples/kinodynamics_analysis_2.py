import os
import math
import torch
import numpy as np

from top_bottom_setup import top_bottom_setup
from examples.general_utils import *
from examples.acoustic_utils import *
from examples.optimizer_utils import *


# Modified based on the kinodynamics_analysis_1.py


if __name__ == '__main__':
    n_particles = 4
    global_model_dir_1 = './experiments/experiment_100'
    model_name = '100_101'
    num_file = 50
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

        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        # 计算时间变化量（差分）
        delta_time = np.diff(split_data[:, :, 1], axis=1)
        # 将时间累加值替换为时间变化量
        split_data[:, 1:, 1] = delta_time 


        t_set = np.copy(split_data[0, :, 1])
        print(t_set)

        key_points = np.copy(split_data[:, :, 2:])
        key_points = np.transpose(key_points, (1, 0, 2))
        print(key_points.shape)

        dx = calculate_dx(key_points)
        #print(dx, '\n')

        v_mean = calculate_mean_v(dx, t_set)
        #print(v_mean)


        # 对第一段和最后一段进行插值处理
        # 先修改第一段和最后一段的dt
        t_set[1] *= 2
        t_set[-1] *= 2
        print(t_set)

        new_t_set = np.zeros(t_set.shape[0] + 18)
        # 前10个元素设置为 t_set[1] / 10
        new_t_set[1:11] = t_set[1] / 10
        # 后10个元素设置为 t_set[-1] / 10
        new_t_set[-10:] = t_set[-1] / 10
        # 中间部分赋值为 t_set 的第2个到倒数第2个元素
        new_t_set[11:-10] = t_set[2:-1]
        print(new_t_set)


        # 对坐标key_points进行插值处理
        new_key_points = np.zeros((key_points.shape[0] + 18, n_particles, 3))
        new_key_points[0] = key_points[0]
        new_key_points[11:-10] = key_points[2:-1]

        #print(new_key_points)

        coordinate_changes_0 = (key_points[1] - key_points[0]) / 100
        for i in range(1, 11):
            new_key_points[i] = key_points[0] + coordinate_changes_0 * (i + i * (i - 1))

        coordinate_changes_1 = (key_points[-1] - key_points[-2]) / 100
        for i in range(1, 11):
            new_key_points[-11+i] = key_points[-2] + coordinate_changes_1 * (19 * i - i * (i - 1))

        #print(new_key_points)

        #new_dx = calculate_dx(new_key_points)
        #print(dx, '\n')

        #v_mean = calculate_mean_v(new_dx, new_t_set)
        #print(v_mean, '\n')

        #accelerations = calculate_accelerations(v_mean, new_t_set)
        #print(accelerations)

        new_split_data_numpy = np.zeros((n_particles, new_key_points.shape[0], 5))
        for i in range(new_key_points.shape[0]):
            new_split_data_numpy[:, i, 0] = i

        for i in range(n_particles):
            new_split_data_numpy[i, :, 1] = new_t_set

        new_split_data_numpy[:, :, 2:] = np.transpose(new_key_points, (1, 0, 2))


        # # 保存修改后的轨迹
        # save_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        # file_instance = open(save_path, "w", encoding="UTF8", newline='')
        # csv_writer = csv.writer(file_instance)

        # for i in range(n_particles):
        #     header = ['Agent ID', i]
        #     row_1 = ['Number of', new_split_data_numpy.shape[1]]

        #     csv_writer.writerow(header)
        #     csv_writer.writerow(row_1)

        #     rows = []
        #     path_time = 0.0
        #     for j in range(new_split_data_numpy.shape[1]):
        #         path_time += new_split_data_numpy[i][j][1]
        #         rows = [j, path_time, new_split_data_numpy[i][j][2], new_split_data_numpy[i][j][3], new_split_data_numpy[i][j][4]]
        #         csv_writer.writerow(rows)

        # file_instance.close()  