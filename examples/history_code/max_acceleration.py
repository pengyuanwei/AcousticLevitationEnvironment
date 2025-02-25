import os
import csv
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def read_csv_file(file_path):
    if not os.path.exists(file_path):
        return None
    
    data_list = []
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data_list.append(row)
    return data_list


def extract_trajectories(n_particles, csv_data):
    max_length = np.zeros(n_particles)
    particle_index = 0
    csv_data_float = []
    invalid_points = 0
    for j in range(len(csv_data)):
        sub_data_list = []
        if csv_data[j] and len(csv_data[j]) == 5:
            if csv_data[j][2] == '-nan(ind)':
                invalid_points += 1
                continue
            sub_data_list = [float(element) for element in csv_data[j]]
            csv_data_float.append(sub_data_list)
            if sub_data_list[0] >= max_length[particle_index]:
                max_length[particle_index] = sub_data_list[0]
            else:
                max_length[particle_index] -= invalid_points
                particle_index += 1
                invalid_points = 0

    if np.max(max_length) == 0.0:
        raise ValueError("Max length of paths equal to zero!")
    
    return csv_data_float, max_length
        

def interpolate_positions(data, dt=32/10000):
    """
    在给定的时间-位置信息之间插入新的时间-位置信息.
    
    参数:
    - data: numpy.ndarray, shape (n, 4), 存储time, x, y, z
    - dt: float, 插值的时间间隔，默认值为 32/10000 秒
    
    返回:
    - interpolated_data: numpy.ndarray, 新的时间-位置信息数组
    """
    # 初始化新的时间-位置信息列表
    interpolated_data = []

    # 遍历每两个相邻的点
    for i in range(len(data) - 1):
        t0, x0, y0, z0 = data[i]
        t1, x1, y1, z1 = data[i + 1]
        
        # 计算插入点的数量
        num_points = int(np.ceil((t1 - t0) / dt))
        
        # 插入新的时间-位置信息
        for j in range(num_points):
            t_new = t0 + j * dt
            if t_new < t1:
                x_new = x0 + (x1 - x0) * (t_new - t0) / (t1 - t0)
                y_new = y0 + (y1 - y0) * (t_new - t0) / (t1 - t0)
                z_new = z0 + (z1 - z0) * (t_new - t0) / (t1 - t0)
                interpolated_data.append([t_new, x_new, y_new, z_new])
        
    # 添加最后的时间点
    interpolated_data.append([t1, x1, y1, z1])
    
    # 转换为numpy数组
    interpolated_data = np.array(interpolated_data)
    
    return interpolated_data


if __name__ == "__main__":
    n_particles = 8
    insert_t = 32/10000

    max_acceleration_set = []

    for n in range(100):
        # read the start and end points
        #csv_file = 'F:/Desktop/Levitator/experimental_data/evaluating_topbottom_8_taskAssignment/path' + str(n) + '.csv'
        csv_file = './experiments/experiment_20/paths/path' + str(n) + '.csv'
        csv_data = read_csv_file(csv_file)
        if not csv_data:
            continue

        csv_data_float, original_max_length = extract_trajectories(n_particles, csv_data)

        original_max_length_int = original_max_length.astype(int)
        original_max_length_int += 1
        
        data_numpy = np.array(csv_data_float)

        #print(data_numpy)
        #print(original_max_length_int)

        # time, x, y, z
        split_data_numpy = [[] for i in range(n_particles)]
        max_length = np.zeros(n_particles).astype(int)

        index = 0
        for j in range(n_particles):
            original_path = data_numpy[index:(index + original_max_length_int[j]), 1:]
            interpolated_path = interpolate_positions(original_path, insert_t)
            split_data_numpy[j] = interpolated_path
            max_length[j] = len(interpolated_path)
            index += original_max_length_int[j]

        velocities = [[] for i in range(n_particles)]
        for i in range(n_particles):
            velocities[i].append(np.zeros(3))
            for j in range(1, max_length[i]):
                dt = split_data_numpy[i][j, :1] - split_data_numpy[i][j - 1, :1]
                dx = split_data_numpy[i][j, 1:] - split_data_numpy[i][j - 1, 1:]

                velocity = dx / dt
                velocities[i].append(velocity)
            velocities[i].append(np.zeros(3))
        
        accelerations = []
        for i in range(n_particles):
            dt = split_data_numpy[i][1, 0]/2.0
            dv = velocities[i][1] - velocities[i][0]
            acceleration = np.linalg.norm(dv) / dt
            accelerations.append(acceleration)

            for j in range(2, max_length[i]-1):
                dt1 = (split_data_numpy[i][j - 1, 0] - split_data_numpy[i][j - 2, 0])/2.0
                dt2 = (split_data_numpy[i][j, 0] - split_data_numpy[i][j - 1, 0])/2.0
                dt = dt1 + dt2
                dv = velocities[i][j] - velocities[i][j - 1]
                acceleration = np.linalg.norm(dv) / dt
                accelerations.append(acceleration)

            dt = (split_data_numpy[i][max_length[i]-1, 0] - split_data_numpy[i][max_length[i]-2, 0])/2.0
            dv = velocities[i][max_length[i]-1] - velocities[i][max_length[i]-2]
            acceleration = np.linalg.norm(dv) / dt
            accelerations.append(acceleration)

        # 找到最大加速度
        max_acceleration = np.max(accelerations)

        print(max_acceleration)

        max_acceleration_set.append(max_acceleration)

    max_acceleration_set = np.array(max_acceleration_set)

    mean = np.mean(max_acceleration_set)
    std = np.std(max_acceleration_set)

    print(f"均值: {mean}")
    print(f"标准差: {std}")