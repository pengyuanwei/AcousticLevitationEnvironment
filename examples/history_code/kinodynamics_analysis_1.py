import os
import math
import torch
import numpy as np

from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *


# Modified based on the trajectory_optimization_1.py
# Do random search to find better Gorkov points for finded paths(keypoints)
# Transform the time series into integer multiples of 32/1000s


def calculate_dx(key_points):
    # key_points的形状(n_keypoints, n_particles, 3): [时刻, 粒子, 坐标]

    # 初始化存储位移的列表
    displacements_set = np.zeros((key_points.shape[0] - 1, key_points.shape[1]))

    # 遍历所有时间段 t1->t2
    for t in range(key_points.shape[0] - 1):
        # 计算8个粒子在时间段t到t+1之间的位移
        displacements = np.sqrt(
            (key_points[t+1, :, 0] - key_points[t, :, 0])**2 +  # x坐标差
            (key_points[t+1, :, 1] - key_points[t, :, 1])**2 +  # y坐标差
            (key_points[t+1, :, 2] - key_points[t, :, 2])**2    # z坐标差
        )
        
        # 保存当前时间段的所有位移
        displacements_set[t] = displacements

    # 输出每个时间段的所有位移
    return displacements_set


def calculate_mean_v(dx, t_set):
    # dx    的形状 (n_segments, n_particles)
    # t_set 的形状 (n_keypoints, ) 应该有 n_segments+1 个 keypoints 对应 n_segments 个时间间隔

    # 初始化存储平均速度的数组
    n_segments, n_particles = dx.shape
    velocities_set = np.zeros((n_segments, n_particles))

    # 遍历所有时间段 t1 -> t2
    for t in range(n_segments):
        # 计算粒子在时间段 t 到 t+1 之间的平均速度
        velocities = dx[t] / t_set[t+1]

        # 保存当前时间段的所有平均速度
        velocities_set[t] = velocities

    # 输出每个时间段的所有平均速度
    return velocities_set


def calculate_accelerations(v_mean, t_set):
    # v_mean 的形状 (n_segments, n_particles)
    # t_set  的形状 (n_keypoints, ) 应该有 n_segments+1 个 keypoints 对应 n_segments 个时间间隔

    # 初始化存储加速度的数组
    n_segments, n_particles = v_mean.shape
    accelerations_set = np.zeros((n_segments-1, n_particles))

    # 遍历所有segments
    for t in range(1, n_segments):
        # 计算粒子从segment 1 到 segment 2 的加速度
        accelerations = (v_mean[t] - v_mean[t-1]) / t_set[t]

        # 保存所有加速度
        accelerations_set[t-1] = accelerations

    # 输出所有加速度
    return accelerations_set


if __name__ == '__main__':
    n_particles = 4
    global_model_dir_1 = './experiments/experiment_100'
    model_name = '100_101'
    num_file = 50
    file_name_0 = 'optimised_data'
    file_name_1 = 'optimised_1_data'

    # Setup gorkov
    l=.00865
    delta = l/32
    density_0=1.2
    speed_0=343
    density_p=1052
    speed_p=1150
    radius=.001

    w=2*np.pi*(speed_0/l)
    volume=4*np.pi*radius**3/3
    k1=volume/4*(1/(density_0*speed_0**2)-1/(density_p*speed_p**2))
    k2=3*volume/4*(density_0-density_p)/(w**2*density_0*(2*density_p+density_0))

    transducer = torch.cat((examples.utils.Gorkov_new.create_board(17,-.24/2),examples.utils.Gorkov_new.create_board(17,.24/2)),axis=0)
    num_transducer = transducer.shape[0]
    m = n_particles
    b = torch.ones(m,1) +1j*torch.zeros(m,1)
    b = b.to(torch.complex64)


    computation_time = []
    for n in range(num_file):
        # start_time = time.time()  # 记录当前循环的开始时间

        print(f'-----------------------The paths {n}-----------------------')
        include_NaN = False

        csv_file = os.path.join(global_model_dir_1, model_name, f'{file_name_0}_{str(n)}.csv')
        #csv_file = 'F:\Desktop\Projects\AcousticLevitationGym\examples\experiments\S2M2_8_experiments\data0.csv'
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            continue

        max_length = np.zeros(n_particles)
        which_particle = 0

        csv_data_float = []
        for j in range(len(csv_data)):
            sub_data_list = []
            if csv_data[j] and len(csv_data[j]) == 5:
                # 检测是否为NaN值
                if any(value == '-nan(ind)' or math.isnan(float(value)) for value in csv_data[j]):
                    include_NaN = True
                    break
                if include_NaN == True:
                    break
                sub_data_list = [float(element) for element in csv_data[j]]
                csv_data_float.append(sub_data_list)
                if sub_data_list[0] >= max_length[which_particle]:
                    max_length[which_particle] = sub_data_list[0]
                else:
                    which_particle += 1

        if np.max(max_length) == 0.0 or include_NaN == True:
            continue

        max_length_int = max_length.astype(int)
        max_length_int += 1
        #print(max_length_int)

        data_numpy = np.array(csv_data_float)


        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # When axis=2: keypoints_id, time, x, y, z
        split_data_numpy = np.zeros((n_particles, np.max(max_length_int), 5))

        for j in range(len(split_data_numpy)):
            split_data_numpy[j, :max_length_int[j]] = data_numpy[:max_length_int[j]]

            if max_length_int[j] < np.max(max_length_int):
                last_particle_position = data_numpy[max_length_int[j]-1]
                split_data_numpy[j, -(np.max(max_length_int)-max_length_int[j]):] = last_particle_position

            data_numpy = data_numpy[max_length_int[j]:]


        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        # 计算时间变化量（差分）
        delta_time = np.diff(split_data_numpy[:, :, 1], axis=1)

        # 保留初始时间点为0，补齐成与原数据相同的形状
        delta_time = np.concatenate([np.zeros((split_data_numpy.shape[0], 1)), delta_time], axis=1)

        # 将时间累加值替换为时间变化量
        split_data_numpy[:, :, 1] = delta_time 


        print(split_data_numpy.shape)   

        t_set = np.copy(split_data_numpy[0, :, 1])
        print(t_set)

        key_points = np.copy(split_data_numpy[:, :, 2:])
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


        # 保存修改后的轨迹
        save_path = os.path.join(global_model_dir_1, model_name, f'{file_name_1}_{str(n)}.csv')
        file_instance = open(save_path, "w", encoding="UTF8", newline='')
        csv_writer = csv.writer(file_instance)

        for i in range(n_particles):
            header = ['Agent ID', i]
            row_1 = ['Number of', new_split_data_numpy.shape[1]]

            csv_writer.writerow(header)
            csv_writer.writerow(row_1)

            rows = []
            path_time = 0.0
            for j in range(new_split_data_numpy.shape[1]):
                path_time += new_split_data_numpy[i][j][1]
                rows = [j, path_time, new_split_data_numpy[i][j][2], new_split_data_numpy[i][j][3], new_split_data_numpy[i][j][4]]
                csv_writer.writerow(rows)

        file_instance.close()  