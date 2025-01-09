import os
import csv
import math
import torch
import numpy as np
import gymnasium as gym

from acoustorl import MADDPG
from acousticlevitationenvironment.utils import general_utils


def save_path(path, save_dir, n_particles, delta_time, num, file_name='path'):
    paths_transpose = np.transpose(path, (1, 0, 2))

    save_path = os.path.join(save_dir, f'{file_name}{str(num)}.csv')
    file_instance = open(save_path, "w", encoding="UTF8", newline='')
    csv_writer = csv.writer(file_instance)

    for i in range(n_particles):
        header = ['Agent ID', i]
        row_1 = ['Number of', len(paths_transpose[i])]

        csv_writer.writerow(header)
        csv_writer.writerow(row_1)

        rows = []
        path_time = 0.0
        for j in range(len(paths_transpose[i])):
            rows = [j, path_time, paths_transpose[i][j][0], paths_transpose[i][j][1], paths_transpose[i][j][2]]
            path_time += delta_time
            csv_writer.writerow(rows)

    file_instance.close()  


def read_csv_file(file_path):
    data_list = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            # 过滤掉空字符串
            filtered_row = [item for item in row if item.strip()]
            data_list.append(filtered_row)
    return data_list


def read_paths(csv_data):
    """
    读取等长路径并转换为浮点型数组，同时检测是否包含 NaN 值。

    参数:
        csv_data (list of list): 输入的 CSV 数据，每行应该包含 5 个值。
    
    返回:
        tuple: 包含两部分 (numpy array of floats, bool indicating if NaN is present)
    """
    try:
        # 转换为浮点数组，并过滤长度不是5的行
        csv_data_float = [
            [float(element) for element in row] 
            for row in csv_data if len(row) == 5
        ]
        
        # 转换为 numpy 数组
        csv_array = np.array(csv_data_float)
        
        # 检查是否存在 NaN 值
        if np.isnan(csv_array).any():
            return np.array([]), True

        return csv_array, False

    except ValueError:
        # 捕获不能转换为浮点数的异常
        return np.array([]), True
    

def process_paths(data_numpy, paths_length):
    # split_data_numpy的形状为(n_particles, n_keypoints, 5)
    # Axis 2: keypoints_idx, 时间累加值（时间列）, x, y, z
    split_data_numpy = data_numpy.reshape(-1, paths_length, 5)

    # 时间变化量：dt不变，不需要差分
    delta_time = split_data_numpy[0][1][1]
    # 将时间累加值替换为时间变化量
    split_data_numpy[:, 2:, 1] = delta_time 

    return split_data_numpy


def interpolate_positions(coords, delta_time_original=0.1, delta_time_new=32/10000):
    num_interpolations = int(delta_time_original / delta_time_new) - 1
    interpolated_coords = []

    for i in range(coords.shape[1]):
        start = coords[0, i]
        end = coords[1, i]
        
        # Calculate step for each dimension
        step = (end - start) / (num_interpolations + 1)
        
        # Generate interpolated positions
        positions = [start + j * step for j in range(num_interpolations + 2)]
        
        interpolated_coords.append(positions)
    
    # Convert list to numpy array
    interpolated_coords = np.array(interpolated_coords)
    
    # Reshape to match the required format (2 * (N * 10) / 10, n_particles, 3)
    interpolated_coords = interpolated_coords.transpose(1, 0, 2).reshape(-1, coords.shape[1], 3)
    
    return interpolated_coords


def safety_area(n_particles, coords):
    collision = np.zeros(n_particles)
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]

    for i in range(n_particles):
        x, y, z = [coords[i][0], coords[i][1], coords[i][2]]
        if not (x_min < x < x_max and y_min < y < y_max and z_min < z < z_max):
            collision[i] = 1.0
            
        for j in range(i+1, n_particles):
            dist = math.sqrt((x - coords[j][0])**2/0.014**2 + 
                                (y - coords[j][1])**2/0.014**2 + 
                                (z - coords[j][2])**2/0.03**2)
            if dist <= 1.0:
                collision[i] = 1.0
                collision[j] = 1.0

    return collision


def euclidean_distance_3d(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)


def calculate_3d_distances(coordinate_list):
    distances = []
    for sublist in coordinate_list:
        sublist_distances = []
        for i in range(len(sublist) - 1):
            dist = euclidean_distance_3d(sublist[i], sublist[i+1])
            sublist_distances.append(dist)
        distances.append(sublist_distances)
    return distances


def generate_global_paths_0(env, agent, n_particles, max_timesteps):
    paths = [[] for _ in range(n_particles)]
    collision_happen = False
        
    state, _ = env.reset()
    terminated, truncated = False, False

    # 更新 paths
    for i in range(n_particles):
        paths[i].append(state[i, :3])

    #print('The target positions are:')
    final_points = np.zeros((n_particles, 3))
    final_points = state[:, :3] + state[:, 6:9]

    for _ in range(max_timesteps):
        action = agent.take_action(state, explore=False)  

        next_state, _, terminated, truncated, _ = env.step(action)
                
        state = next_state
        for i in range(n_particles):
            paths[i].append(state[i, :3])

        if terminated or (truncated == 1):
            for i in range(n_particles):
                paths[i].append(final_points[i])
            break
        elif truncated == 2:
            collision_happen = True

    paths_array = np.array(paths)
    paths_transpose = np.transpose(paths_array, (1, 0, 2))
    #print(f'The key points shape: {paths_transpose.shape} \n')

    return paths_transpose, truncated, collision_happen


def generate_global_paths(env, agent, n_particles: int, max_timesteps: int):
    paths = [[] for _ in range(n_particles)]
        
    state, _ = env.reset()
    terminated, truncated = False, False

    # 更新 paths
    for i in range(n_particles):
        paths[i].append(state[i, :3])

    #print('The target positions are:')
    final_points = state[:, :3] + state[:, 6:9]

    for _ in range(max_timesteps):
        action = agent.take_action(state, explore=False)  

        next_state, _, terminated, truncated, _ = env.step(action)
                
        state = next_state
        for i in range(n_particles):
            paths[i].append(state[i, :3])

        if terminated or truncated:
            for i in range(n_particles):
                paths[i].append(final_points[i])
            break

    paths_array = np.array(paths)
    paths_transpose = np.transpose(paths_array, (1, 0, 2))
    #print(f'The key points shape: {paths_transpose.shape} \n')

    return paths_transpose, truncated


def generate_global_paths_combination(env, agent1, agent2, n_particles: int, max_timesteps: int):
    # agent 1: target arriving
    # agent 2: gorkov minimization
    max_timesteps *= 2

    paths = [[] for _ in range(n_particles)]
        
    state, _ = env.reset()
    terminated, truncated = False, False

    # 更新 paths
    for i in range(n_particles):
        paths[i].append(state[i, :3])

    #print('The target positions are:')
    final_points = state[:, :3] + state[:, 6:9]

    for i in range(max_timesteps):
        if i < 5 or i >= 10:
            action = agent1.take_action(state, explore=False)  
        else:
            print('Gorkov optimization!')
            action = agent2.take_action(state, explore=False)  

        next_state, _, terminated, truncated, _ = env.step(action)
                
        state = next_state
        for i in range(n_particles):
            paths[i].append(state[i, :3])

        if terminated or truncated:
            for i in range(n_particles):
                paths[i].append(final_points[i])
            break

    paths_array = np.array(paths)
    paths_transpose = np.transpose(paths_array, (1, 0, 2))
    #print(f'The key points shape: {paths_transpose.shape} \n')

    return paths_transpose, truncated


def generate_global_paths_input(env, agent, n_particles, max_timesteps, start_points, target_points):
    paths = [[] for _ in range(n_particles)]
        
    env.input_start_end_points(start_points, target_points)
    state, _ = env.reset()
    terminated, truncated = False, False

    # 更新 paths
    for i in range(n_particles):
        paths[i].append(state[i, :3])

    #print('The target positions are:')
    final_points = np.zeros((n_particles, 3))
    final_points = state[:, :3] + state[:, 6:9]

    for _ in range(max_timesteps):
        action = agent.take_action(state, explore=False)  

        next_state, _, terminated, truncated, _ = env.step(action)
                
        state = next_state
        for i in range(n_particles):
            paths[i].append(state[i, :3])

        if terminated or truncated:
            for i in range(n_particles):
                paths[i].append(final_points[i])
            break

    paths_array = np.array(paths)
    paths_transpose = np.transpose(paths_array, (1, 0, 2))
    #print(f'The key points shape: {paths_transpose.shape} \n')

    return paths_transpose, truncated


def generate_replan_paths(env, agent, n_particles, max_timesteps, points):
    paths = [[] for _ in range(n_particles)]

    start_points = points[0]
    target_points = points[-1]
    
    env.unwrapped.input_start_end_points(start_points, target_points)
    state, _ = env.reset()
    terminated, truncated = False, False

    # 更新 paths
    for i in range(n_particles):
        paths[i].append(state[i, :3])

    #print('The target positions are:')
    final_points = state[:, :3] + state[:, 6:9]

    for _ in range(max_timesteps):
        action = agent.take_action(state, explore=False)  

        next_state, _, terminated, truncated, _ = env.step(action)
                
        state = next_state
        for i in range(n_particles):
            paths[i].append(state[i, :3])

        if terminated or truncated:
            for i in range(n_particles):
                paths[i].append(final_points[i])
            break

    paths_array = np.array(paths)
    paths_transpose = np.transpose(paths_array, (1, 0, 2))
    #print(f'The key points shape: {paths_transpose.shape} \n')

    return paths_transpose, truncated