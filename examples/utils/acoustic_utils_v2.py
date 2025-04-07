import csv
import numpy as np
import torch
import os
from scipy.spatial.distance import cdist

from examples.utils.general_utils_v2 import *
import examples.utils.phase_retrieval as phase_retrieval


# Modified based on the calculate_gorkov_v2.py
# Calculate Gorkov for S2M2 generated trajectories: save all gorkov values


def calculate_distances(positions):
    # 假设有n个粒子，m个时刻，三维坐标
    # positions的形状应该是 (n, m, 3)，其中n是粒子数，m是时刻数，3是三维坐标
    # 计算每个粒子每个时刻与上一个时刻之间的位移
    # 位移是两个连续时刻的坐标差的欧几里得距离
    displacements = np.linalg.norm(np.diff(positions, axis=1), axis=2)
    
    # 找到所有位移中的最大值
    min_displacement = np.min(displacements)
    
    return min_displacement


def calculate_distance(segment, n):
    # 假设 segment 已经是 (3, 8, 3) 的形状
    # segment[时刻, 粒子, 坐标]

    # 初始化存储位移的列表
    displacements_set = np.zeros((2, n))

    # 遍历两个时间段 t1->t2 和 t2->t3
    for t in range(2):  # 两个时间段
        # 计算8个粒子在时间段t到t+1之间的位移
        displacements = np.sqrt(
            (segment[t+1, :, 0] - segment[t, :, 0])**2 +  # x坐标差
            (segment[t+1, :, 1] - segment[t, :, 1])**2 +  # y坐标差
            (segment[t+1, :, 2] - segment[t, :, 2])**2    # z坐标差
        )
        
        # 保存当前时间段的所有位移
        displacements_set[t] = displacements

    # 输出每个时间段的最小位移
    return np.min(displacements_set, axis=0)


def create_constrained_points(N, particles, cube_size, max_attempts=1000):
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]

    solutions = []
    attempts_1 = 0
    while attempts_1 < 100:
        points = []
        attempts_2 = 0
        for i in range(N):
            # 在[-cube_size/2, cube_size/2]范围内随机生成一个点
            movement = np.random.uniform(-cube_size[i]/2, cube_size[i]/2, 3)

            point = np.array([min(max(particles[i][0] + movement[0], x_min), x_max), 
                              min(max(particles[i][1] + movement[1], y_min), y_max), 
                              min(max(particles[i][2] + movement[2], z_min), z_max)])
            
            # 检查与已生成点之间的椭球体距离
            if all(np.linalg.norm((point - p) / np.array([0.015, 0.015, 0.03])) > 1 for p in points):
                points.append(point)
            
            attempts_2 += 1
            if attempts_2 == max_attempts:
                break

        attempts_1 += 1

        if len(points) == N:
            solutions.append(points)
    
    return np.array(solutions)


def create_constrained_points_1(N, cur_positions, last_positions, next_positions, max_attempts=1000):
    '''
    下个状态已知；不能固定某些粒子
    '''
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    cube_size = np.linalg.norm((next_positions - last_positions), axis=1)
    search_area_center = (next_positions + last_positions) / 2.0

    solutions = []
    attempts_1 = 0
    while attempts_1 < 100:
        points = []
        attempts_2 = 0
        for i in range(N):
            # 在search area内随机生成一个点（确保两个segment形成的夹角大于135度
            movement = np.random.uniform(-cube_size[i]/(2*2.42), cube_size[i]/(2*2.42), 3)

            point = np.array([min(max(search_area_center[i][0] + movement[0], x_min), x_max), 
                              min(max(search_area_center[i][1] + movement[1], y_min), y_max), 
                              min(max(search_area_center[i][2] + movement[2], z_min), z_max)])
            
            # 检查与已生成点之间的椭球体距离
            if all(np.linalg.norm((point - p) / np.array([0.015, 0.015, 0.03])) > 1 for p in points):
                points.append(point)
            
            attempts_2 += 1
            if attempts_2 == max_attempts:
                break

        attempts_1 += 1

        if len(points) == N:
            solutions.append(points)
    
    return np.array(solutions)


def create_constrained_points_v2(N, cur_positions, last_positions, next_positions, max_attempts=1000):
    '''
    下个状态已知；不能固定某些粒子
    output:
        - np.transpose(np.array(solutions), (1, 0, 2)): (n_particles, num_solutions, 3)
    '''    
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    cube_size = np.linalg.norm((next_positions - last_positions), axis=1)
    search_area_center = (next_positions + last_positions) / 2.0

    solutions = []
    attempts_1 = 0
    while attempts_1 < 100:
        points = []
        attempts_2 = 0
        for i in range(N):
            # 在search area内随机生成一个点（确保两个segment形成的夹角大于135度
            movement = np.random.uniform(-cube_size[i]/(2*2.42), cube_size[i]/(2*2.42), 3)

            point = np.array([min(max(search_area_center[i][0] + movement[0], x_min), x_max), 
                              min(max(search_area_center[i][1] + movement[1], y_min), y_max), 
                              min(max(search_area_center[i][2] + movement[2], z_min), z_max)])
            
            # 检查与已生成点之间的椭球体距离
            if all(np.linalg.norm((point - p) / np.array([0.015, 0.015, 0.03])) > 1 for p in points):
                points.append(point)
            
            attempts_2 += 1
            if attempts_2 == max_attempts:
                break

        attempts_1 += 1

        if len(points) == N:
            solutions.append(points)
    
    return np.transpose(np.array(solutions), (1, 0, 2))


def create_constrained_points_5(
        n_particles: int, 
        last_positions: np.array, 
        current_positions: np.array,
        next_positions: np.array, 
        reach_index: np.array, 
        num_solutions: int=10,
        search_factor: float=10.0
    ):
    '''
    下个状态已知；允许固定一些粒子
    input:
        - reach_index: (num_particles, fixed_or_not)
    output:
        - np.array(solutions): (num_solutions, n_particles, 3)
    '''
    fixed_points = {}
    for i in range(n_particles):
        if reach_index[i]:
            fixed_points[i] = np.array([current_positions[i][0], current_positions[i][1], current_positions[i][2]])

    cube_size = np.linalg.norm((next_positions - last_positions), axis=1)
    search_area_center = (next_positions + last_positions) / 2.0
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    solutions = []
    attempts = 0
    while len(solutions) < num_solutions:
        searched_points = {}
        for i in range(n_particles): 
            if not reach_index[i]:
                # 在search area内随机生成一个点
                movement = np.random.uniform(-cube_size[i]/search_factor, cube_size[i]/search_factor, 3)
                point = np.array([min(max(search_area_center[i][0] + movement[0], x_min), x_max), 
                                min(max(search_area_center[i][1] + movement[1], y_min), y_max), 
                                min(max(search_area_center[i][2] + movement[2], z_min), z_max)])
                
                # 检查与已生成点之间的椭球体距离约束
                valid_fixed = is_valid_distance(point, fixed_points.values())
                valid_searched = is_valid_distance(point, searched_points.values())
                if valid_fixed and valid_searched:
                    searched_points[i] = point
                
        if (len(fixed_points) + len(searched_points)) == n_particles:
            new_points = fixed_points | searched_points
            sorted_keys = sorted(new_points.keys())
            points_array = np.array([new_points[key] for key in sorted_keys])
            solutions.append(points_array)
        
        attempts += 1
        if attempts >= num_solutions*100:
            print(f"Gorkov solutions: 达到最大迭代次数, solution 数量: {len(solutions)}。")
            break

    if len(solutions) > 0:
        return np.array(solutions)
    else:
        return None


def create_constrained_points_6(
        n_particles: int, 
        last_positions: np.array, 
        current_positions: np.array,
        next_positions: np.array, 
        reach_index: np.array, 
        num_solutions: int=10,
        search_factor: float=10.0
    ):
    '''
    下个状态已知；允许固定一些粒子。在原坐标的周围生成
    input:
        - reach_index: (num_particles, fixed_or_not), 未到达终点为1
    output:
        - np.array(solutions): (num_solutions, n_particles, 3)
    '''
    fixed_points = {}
    for i in range(n_particles):
        if not reach_index[i]:
            fixed_points[i] = np.array([current_positions[i][0], current_positions[i][1], current_positions[i][2]])

    cube_size = np.linalg.norm((next_positions - last_positions), axis=1)
    search_area_center = current_positions
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    solutions = []
    attempts = 0
    while len(solutions) < num_solutions:
        searched_points = {}
        for i in range(n_particles): 
            if reach_index[i]:
                # 在search area内随机生成一个点
                movement = np.random.uniform(-cube_size[i]/search_factor, cube_size[i]/search_factor, 3)
                point = np.array([min(max(search_area_center[i][0] + movement[0], x_min), x_max), 
                                min(max(search_area_center[i][1] + movement[1], y_min), y_max), 
                                min(max(search_area_center[i][2] + movement[2], z_min), z_max)])
                
                # 检查与已生成点之间的椭球体距离约束
                valid_fixed = is_valid_distance(point, fixed_points.values())
                valid_searched = is_valid_distance(point, searched_points.values())
                if valid_fixed and valid_searched:
                    searched_points[i] = point
                
        if (len(fixed_points) + len(searched_points)) == n_particles:
            new_points = fixed_points | searched_points
            sorted_keys = sorted(new_points.keys())
            points_array = np.array([new_points[key] for key in sorted_keys])
            solutions.append(points_array)
        
        attempts += 1
        if attempts >= num_solutions*100:
            print(f"Gorkov solutions: 达到最大迭代次数, solution 数量: {len(solutions)}。")
            break

    if len(solutions) > 0:
        return np.array(solutions)
    else:
        return None
    

def create_constrained_points_single_frame(
        n_particles: int, 
        last_positions: np.array, 
        positions: np.array, 
        last_displacements: np.array, 
        displacements: np.array, 
        reach_index: np.array, 
        num_solutions: int=10
    ):
    '''
    基于前一个位移和新位移，生成一个搜索区域，来生成一些candidate solutions
    output:
        - np.transpose(np.array(solutions), (1, 0, 2)): (n_particles, num_solutions, 3)
    '''
    fixed_points = {}
    for i in range(n_particles):
        if reach_index[i]:
            fixed_points[i] = np.array([positions[i][0], positions[i][1], positions[i][2]])

    search_domain = displacements - last_displacements
    guess_positions = last_positions + last_displacements
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    solutions = []
    attempts = 0
    while len(solutions) < num_solutions:
        searched_points = {}
        for i in range(n_particles): 
            if not reach_index[i]:
                # 在search area内随机生成一个点
                movement = np.zeros((3, ))
                movement[0] = np.random.uniform(min(0.0, search_domain[i][0]), max(0.0, search_domain[i][0]))
                movement[1] = np.random.uniform(min(0.0, search_domain[i][1]), max(0.0, search_domain[i][1]))
                movement[2] = np.random.uniform(min(0.0, search_domain[i][2]), max(0.0, search_domain[i][2]))
                point = np.array([min(max(guess_positions[i][0] + movement[0], x_min), x_max), 
                                  min(max(guess_positions[i][1] + movement[1], y_min), y_max), 
                                  min(max(guess_positions[i][2] + movement[2], z_min), z_max)])
                
                # 检查与已生成点之间的椭球体距离约束
                valid_fixed = is_valid_distance(point, fixed_points.values())
                valid_searched = is_valid_distance(point, searched_points.values())
                if valid_fixed and valid_searched:
                    searched_points[i] = point

        if (len(fixed_points) + len(searched_points)) == n_particles:
            new_points = fixed_points | searched_points
            sorted_keys = sorted(new_points.keys())
            points_array = np.array([new_points[key] for key in sorted_keys])
            solutions.append(points_array)

        attempts += 1
        if attempts >= num_solutions*100:
            print(f"Gorkov solutions: 达到最大迭代次数, solution 数量: {len(solutions)}。")
            break

    if len(solutions) > 0:
        return np.transpose(np.array(solutions), (1, 0, 2))
    else:
        return None


def create_constrained_points_single_frame_v2(
        n_particles: int, 
        previous_frame: np.array, 
        current_frame: np.array, 
        reach_index: np.array, 
        num_solutions: int=10,
        search_factor: float=5.0
    ):
    '''
    为当前时刻生成solutions；下个状态未知；允许固定一些粒子
    output:
        - np.transpose(np.array(solutions), (1, 0, 2)): (n_particles, num_solutions, 3)
    '''
    fixed_points = {}
    for i in range(n_particles):
        if reach_index[i]:
            fixed_points[i] = np.array([current_frame[i][0], current_frame[i][1], current_frame[i][2]])

    cube_size = np.linalg.norm((current_frame - previous_frame), axis=1)
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    solutions = []
    attempts = 0
    while len(solutions) < num_solutions:
        searched_points = {}
        for i in range(n_particles): 
            if not reach_index[i]:
                # 在search area内随机生成一个点
                movement = np.random.uniform(-cube_size[i]/search_factor, cube_size[i]/search_factor, 3)
                point = np.array([min(max(current_frame[i][0] + movement[0], x_min), x_max), 
                                  min(max(current_frame[i][1] + movement[1], y_min), y_max), 
                                  min(max(current_frame[i][2] + movement[2], z_min), z_max)])
                
                # 检查与已生成点之间的椭球体距离约束
                valid_fixed = is_valid_distance(point, fixed_points.values())
                valid_searched = is_valid_distance(point, searched_points.values())
                if valid_fixed and valid_searched:
                    searched_points[i] = point

        if (len(fixed_points) + len(searched_points)) == n_particles:
            new_points = fixed_points | searched_points
            sorted_keys = sorted(new_points.keys())
            points_array = np.array([new_points[key] for key in sorted_keys])
            solutions.append(points_array)

        attempts += 1
        if attempts >= num_solutions*100:
            print(f"Gorkov solutions: 达到最大迭代次数, solution 数量: {len(solutions)}。")
            break

    if len(solutions) > 0:
        return np.transpose(np.array(solutions), (1, 0, 2))
    else:
        return None
    

def is_valid_distance(point, other_points, scale=np.array([0.015, 0.015, 0.03])):
    """检查 point 与 other_points 中所有点的归一化欧几里得距离平方和是否大于 1"""
    return all(np.sum(((point - p) / scale) ** 2) > 1 for p in other_points)


def positions_check(
        cur_positions, 
        last_positions, 
        next_positions
    ):
    cube_size = np.linalg.norm((next_positions - last_positions), axis=1)
    search_area_center = (next_positions + last_positions) / 2.0

    relative_dist = np.linalg.norm((cur_positions - search_area_center), axis=1)

    # 使用向量化比较
    return np.any(relative_dist > (cube_size/(2*2.42)))


def calculate_gorkov(key_points, n_particles, transducer, delta, b, num_transducer, k1, k2):

    transformed_coordinate = key_points.copy()
    transformed_coordinate[:, :, 2] -= 0.12

    gorkov_all_timestep = np.zeros((key_points.shape[1], n_particles))

    for i in range(key_points.shape[1]):
        points = np.zeros((n_particles, 3))
        for j in range(n_particles):
            points[j] = [transformed_coordinate[j][i][0], transformed_coordinate[j][i][1], transformed_coordinate[j][i][2]]

        points1 = torch.tensor(points)
        Ax2, Ay2, Az2 = phase_retrieval.surround_points(transducer, points1, delta)
        Ax2 = Ax2.to(torch.complex64)
        Ay2 = Ay2.to(torch.complex64)
        Az2 = Az2.to(torch.complex64)
        H = phase_retrieval.piston_model(transducer, points1).to(torch.complex64)
        gorkov = phase_retrieval.wgs_new(H, Ax2, Ay2, Az2, b, num_transducer, k1, k2, 1)

        gorkov_numpy = gorkov.numpy()
        
        gorkov_numpy_transpose = gorkov_numpy.T

        gorkov_all_timestep[i:i+1, :] = gorkov_numpy_transpose

    return gorkov_all_timestep


def calculate_max_gorkov(gorkov):
    max_gorkov = np.max(gorkov, axis=1)
    max_gorkov[[0, -1]] = -1.0
    max_index = np.argmax(max_gorkov)
    return max_index, max_gorkov

def calculate_max_gorkov_v2(gorkov):
    max_gorkov = np.max(gorkov, axis=1)
    max_gorkov[[0, -1]] = -1.0
    max_index = np.argmax(max_gorkov)
    # 根据 Gorkov 对 candidate_solutions 从小到大排序
    sorted_indices = np.argsort(max_gorkov)
    return max_index, max_gorkov, sorted_indices

def read_csv_file(file_path):
    if not os.path.exists(file_path):
        return None
    
    data_list = []
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data_list.append(row)
    return data_list