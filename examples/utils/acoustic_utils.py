import csv
import numpy as np
import torch
import os
from scipy.spatial.distance import cdist

from examples.utils.general_utils import *
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


def create_constrained_points_2(N, cur_positions, last_positions, next_positions, max_total_attempts = 1000, max_attempts=1000):
    # The solutions are generated from a circle plane:
    # The center of the circle is the midpoint of the front-back connection line,
    # and the circle is perpendicular to the front-back connection line.

    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]

    search_center = (next_positions + last_positions) / 2.0
    vs, us, ws = unit_vectors_of_search_circle(last_positions, next_positions)
    front_back_connection_line = np.linalg.norm(vs, axis=1)[:, np.newaxis]
    search_radius = front_back_connection_line / (2*2.415*2)

    solutions = []
    total_attempts = 0
    while len(solutions) < 100 and total_attempts < max_total_attempts:
        # 用无效值初始化 sample_points
        sample_points = np.full((N, 3), np.nan)
        attempts = np.zeros(N, dtype=int)
        while np.any(np.isnan(sample_points)) and np.all(attempts < max_attempts):
            # 识别需要重新采样的粒子
            # shape: (N, )
            to_resample = np.isnan(sample_points[:, 0])
            # 存储to_resample中True的索引
            to_resample_indices = np.where(to_resample)[0]
            
            # shape: (len(to_resample中True的数量)， 1)
            sample_radius = np.random.uniform(0.0, search_radius[to_resample])
            # np.sum(to_resample): to_resample中True的数量
            # shape: (len(to_resample中True的数量)， 1)
            sample_angle = np.random.uniform(0.0, 2 * np.pi, len(to_resample_indices))[:, np.newaxis]
            new_points = circle_points(
                search_center[to_resample],
                us[to_resample],
                ws[to_resample],
                sample_radius,
                sample_angle
            )

            # 判断粒子是否在x_min, x_max, y_min, y_max, z_min, z_max的范围中
            valid_bounds = (
                (new_points[:, 0] >= x_min) & (new_points[:, 0] <= x_max) &
                (new_points[:, 1] >= y_min) & (new_points[:, 1] <= y_max) &
                (new_points[:, 2] >= z_min) & (new_points[:, 2] <= z_max)
            )

            # 获取有效的新点的索引
            valid_new_points_indices = np.where(valid_bounds)[0]
            valid_indices = to_resample_indices[valid_new_points_indices]

            # 将有效的新点放入 sample_points 中
            sample_points[valid_indices] = new_points[valid_bounds]

            # 更新尝试次数
            attempts[to_resample_indices] += 1

            # 对已采样的点检查距离约束
            # ~ 是逻辑非运算符
            indices = np.where(~np.isnan(sample_points[:, 0]))[0]
            if len(indices) > 1:
                # 提取有效的点
                valid_points = sample_points[indices]

                # 计算点之间的距离（椭球距离）
                diff = valid_points[:, np.newaxis, :] - valid_points[np.newaxis, :, :]
                diff[:, :, 0] /= 0.014  # X 轴缩放
                diff[:, :, 1] /= 0.014  # Y 轴缩放
                diff[:, :, 2] /= 0.03   # Z 轴缩放
                
                # 计算距离的平方矩阵，避免使用平方根以提高效率
                dist_squared = np.sum(diff**2, axis=-1)
                
                # 标记距离小于等于 1.0 的点对（对应距离的平方 <= 1.0）
                violating_mask = np.triu(dist_squared <= 1.0, k=1)  # 只选取上三角部分，避免重复计算
                
                # 找到违反约束的点索引
                violating_indices = np.where(np.any(violating_mask, axis=0))[0]
                
                # 重新标记违法点以重新采样
                sample_points[indices[violating_indices]] = np.nan


        if np.all(~np.isnan(sample_points)):
            solutions.append(sample_points)
            # print("New solution has been added!")
        total_attempts += 1

    if len(solutions) < 100:
        print("无法在最大尝试次数内生成足够的有效采样点。")

    return solutions


def unit_vectors_of_search_circle(front_points, back_points):    
    # 计算从P1到P2的所有向量, 形状为 (N, 3)
    vs = back_points - front_points

    # 选择k向量，形状为 (N,)
    condition = vs[:, 2] == 0
    # 初始化 k_replacements 数组，形状为 (N, 3)
    k_replacements = np.zeros_like(vs)
    # 对于满足条件的行，赋值为 [0, 0, 1]
    k_replacements[condition] = [0, 0, 1]
    # 对于不满足条件的行，赋值为 [1, 0, 0]
    k_replacements[~condition] = [1, 0, 0]

    # 计算垂直向量u和w
    us = np.cross(vs, k_replacements)
    ws = np.cross(vs, us)
    
    # 标准化向量
    us = us / np.linalg.norm(us, axis=1)[:, np.newaxis]
    ws = ws / np.linalg.norm(ws, axis=1)[:, np.newaxis]
        
    return vs, us, ws


def circle_points(Ms, us, ws, rs, ts):
    # 计算多个圆上的点
    # Ms: 多个中点的数组，形状为 (n, 3)
    # us: 对应每个中点的单位向量u的数组，形状为 (n, 3)
    # ws: 对应每个中点的单位向量w的数组，形状为 (n, 3)
    # rs: 每个圆的半径的数组，已经是形状为 (n, 1)
    # ts: 参数t的数组，已经是形状为 (n, 1)
    
    # 计算圆的参数方程的结果
    cos_t = np.cos(ts)  # 已经是列向量，用于广播
    sin_t = np.sin(ts)  # 已经是列向量，用于广播
    points = Ms + rs * (cos_t * us + sin_t * ws)
    
    return points


def create_constrained_points_3(N, cur_positions, last_positions, next_positions, max_total_attempts = 1000, max_attempts=1000):
    # The solutions are generated from a circle plane:
    # The center of the circle is the midpoint of the front-back connection line,
    # and the circle is perpendicular to the front-back connection line.

    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]

    search_center = (next_positions + last_positions) / 2.0
    us, ws = unit_vectors_of_search_circle(last_positions, next_positions)
    front_back_connection_line = np.linalg.norm((next_positions - last_positions), axis=1)[:, np.newaxis]
    search_radius = front_back_connection_line / (2*2.415)

    solutions = []
    total_attempts = 0
    while len(solutions) < 100 and total_attempts < max_total_attempts:
        # 用无效值初始化 sample_points
        sample_points = np.full((N, 3), np.nan)
        attempts = np.zeros(N, dtype=int)
        while np.any(np.isnan(sample_points)) and np.all(attempts < max_attempts):
            # 识别需要重新采样的粒子
            # shape: (N, )
            to_resample = np.isnan(sample_points[:, 0])
            # 存储to_resample中True的索引
            to_resample_indices = np.where(to_resample)[0]
            
            # shape: (len(to_resample中True的数量)， 1)
            sample_radius = np.random.uniform(0.0, search_radius[to_resample])
            # np.sum(to_resample): to_resample中True的数量
            # shape: (len(to_resample中True的数量)， 1)
            sample_angle = np.random.uniform(0.0, 2 * np.pi, len(to_resample_indices))[:, np.newaxis]
            new_points = circle_points(
                search_center[to_resample],
                us[to_resample],
                ws[to_resample],
                sample_radius,
                sample_angle
            )

            # 判断粒子是否在x_min, x_max, y_min, y_max, z_min, z_max的范围中
            valid_bounds = (
                (new_points[:, 0] >= x_min) & (new_points[:, 0] <= x_max) &
                (new_points[:, 1] >= y_min) & (new_points[:, 1] <= y_max) &
                (new_points[:, 2] >= z_min) & (new_points[:, 2] <= z_max)
            )

            # 获取有效的新点的索引
            valid_new_points_indices = np.where(valid_bounds)[0]
            valid_indices = to_resample_indices[valid_new_points_indices]

            # 将有效的新点放入 sample_points 中
            sample_points[valid_indices] = new_points[valid_bounds]

            # 更新尝试次数
            attempts[to_resample_indices] += 1
            print(attempts)

            # 对已采样的点检查距离约束
            # ~ 是逻辑非运算符
            indices = np.where(~np.isnan(sample_points[:, 0]))[0]
            if len(indices) > 1:
                # 提取采样的粒子的 XY 坐标，计算这些粒子之间的欧氏距离，得到一个距离矩阵。
                distances = cdist(sample_points[indices, 0:2], sample_points[indices, 0:2])
                # 将距离矩阵的对角线（即粒子自身与自身的距离）设置为无穷大，防止在后续检查中误判。
                np.fill_diagonal(distances, np.inf)
                print(distances, '\n')
                # 找出距离小于 0.015 的粒子对，即违反最小距离约束的粒子对。
                violating_pairs = np.where(distances < 0.015)

                # 识别违反距离约束的粒子索引
                violating_indices = np.unique(np.concatenate((indices[violating_pairs[0]], indices[violating_pairs[1]])))
                sample_points[violating_indices] = np.nan  # 标记它们以重新采样

        if not np.any(np.isnan(sample_points)):
            solutions.append(sample_points)
            print("New solution has been added!")
        total_attempts += 1

    if len(solutions) < 100:
        print("无法在最大尝试次数内生成足够的有效采样点。")

    return solutions


#############################################################################################################################################################
def angle_between(v1, v2):
    """
    计算两个向量之间的角度（以弧度为单位）。
    """
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        # 如果其中一个向量为零向量，角度定义为0.0
        return 0.0

    # 计算v1和v2之间的余弦值
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)

    # 处理可能将cos_theta推到[-1, 1]之外的数值误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算角度（以弧度为单位）
    angle = np.arccos(cos_theta)
    return angle


def compute_angles_and_distances(trajectories, degrees=False):
    """
    计算沿每个轨迹的各段之间的角度变化和相邻关键点之间的距离。

    参数：
    - trajectories: 轨迹数组，形状为 (N, M, 3)，其中N是轨迹数量，M是每个轨迹的关键点数量。
    - degrees: 如果为True，返回角度以度为单位；否则以弧度为单位。

    返回：
    - angles_array: 形状为 (N, M - 2) 的 NumPy 数组，包含每个轨迹的角度变化。
    - distances_array: 形状为 (N, M - 1) 的 NumPy 数组，包含每个轨迹的相邻关键点之间的距离。
    """
    N, M, _ = trajectories.shape

    # 初始化角度和距离数组，使用 NaN 填充，以处理可能的缺失值
    angles_array = np.full((N, M - 2), np.nan)
    distances_array = np.full((N, M - 1), np.nan)

    for idx in range(N):
        keypoints = trajectories[idx]  # 形状为 (M, 3)

        # 计算关键点之间的段，形状为 (M - 1, 3)
        segments = keypoints[1:] - keypoints[:-1]

        # 计算相邻关键点之间的距离，长度为 M - 1
        distances = np.linalg.norm(segments, axis=1)
        # 将可能的 nan 值替换为 0.0
        distances = np.nan_to_num(distances, nan=0.0)
        distances_array[idx, :] = distances

        # 计算连续段之间的角度，长度为 M - 2
        for i in range(len(segments) - 1):
            v1 = segments[i]
            v2 = segments[i + 1]
            angle = angle_between(v1, v2)
            if degrees:
                angle = np.degrees(angle)
            # 如果角度为 nan，替换为 0.0
            if np.isnan(angle):
                angle = 0.0
            angles_array[idx, i] = angle

    # 将所有 NaN 值替换为 0.0
    angles_array = np.nan_to_num(angles_array, nan=0.0)
    distances_array = np.nan_to_num(distances_array, nan=0.0)

    return angles_array, distances_array


def preprocess_trajectories(trajectories, threshold=0.002):
    """
    将距离终点小于指定阈值的点替换为终点坐标，并生成标记数组。

    参数：
    - trajectories: 轨迹数组，形状为 (N, M, 3)
    - threshold: 距离阈值，小于该值的点将被替换为终点

    返回：
    - processed_trajectories: 预处理后的轨迹数组
    - endpoint_flags: 布尔数组，形状与 trajectories 一致，标记哪些关键点已到达终点
    """
    processed_trajectories = trajectories.copy()
    N = processed_trajectories.shape[0]
    endpoint_flags = []

    for idx in range(N):
        keypoints = processed_trajectories[idx]
        M = keypoints.shape[0]
        end_point = keypoints[-1]
        distances_to_end = np.linalg.norm(keypoints - end_point, axis=1)
        close_to_end = distances_to_end < threshold
        keypoints[close_to_end] = end_point
        processed_trajectories[idx] = keypoints
        # 创建布尔数组，标记哪些关键点已到达终点
        flags = close_to_end
        endpoint_flags.append(flags)

    endpoint_flags = np.array(endpoint_flags)
    return processed_trajectories, endpoint_flags


def create_constrained_points_4(N, cur_positions, last_positions, next_positions, tan_coefficient, max_total_attempts = 1000, max_attempts=1000):
    # The solutions are generated from a circle plane:
    # The center of the circle is the midpoint of the front-back connection line,
    # and the circle is perpendicular to the front-back connection line.

    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]

    search_center = (next_positions + last_positions) / 2.0
    vs, us, ws = unit_vectors_of_search_circle(last_positions, next_positions)
    front_back_connection_line = np.linalg.norm(vs, axis=1)[:, np.newaxis]
    search_radius = front_back_connection_line / (2*tan_coefficient)

    solutions = []
    total_attempts = 0
    while len(solutions) < 100 and total_attempts < max_total_attempts:
        # 用无效值初始化 sample_points
        sample_points = np.full((N, 3), np.nan)
        attempts = np.zeros(N, dtype=int)
        while np.any(np.isnan(sample_points)) and np.all(attempts < max_attempts):
            # 识别需要重新采样的粒子
            # shape: (N, )
            to_resample = np.isnan(sample_points[:, 0])
            # 存储to_resample中True的索引
            to_resample_indices = np.where(to_resample)[0]
            
            # shape: (len(to_resample中True的数量)， 1)
            sample_radius = np.random.uniform(0.0, search_radius[to_resample])
            # np.sum(to_resample): to_resample中True的数量
            # shape: (len(to_resample中True的数量)， 1)
            sample_angle = np.random.uniform(0.0, 2 * np.pi, len(to_resample_indices))[:, np.newaxis]
            new_points = circle_points(
                search_center[to_resample],
                us[to_resample],
                ws[to_resample],
                sample_radius,
                sample_angle
            )

            # 判断粒子是否在x_min, x_max, y_min, y_max, z_min, z_max的范围中
            valid_bounds = (
                (new_points[:, 0] >= x_min) & (new_points[:, 0] <= x_max) &
                (new_points[:, 1] >= y_min) & (new_points[:, 1] <= y_max) &
                (new_points[:, 2] >= z_min) & (new_points[:, 2] <= z_max)
            )

            # 获取有效的新点的索引
            valid_new_points_indices = np.where(valid_bounds)[0]
            valid_indices = to_resample_indices[valid_new_points_indices]

            # 将有效的新点放入 sample_points 中
            sample_points[valid_indices] = new_points[valid_bounds]

            # 更新尝试次数
            attempts[to_resample_indices] += 1

            # 对已采样的点检查距离约束
            # ~ 是逻辑非运算符
            indices = np.where(~np.isnan(sample_points[:, 0]))[0]
            if len(indices) > 1:
                # 提取有效的点
                valid_points = sample_points[indices]

                # 计算点之间的距离（椭球距离）
                diff = valid_points[:, np.newaxis, :] - valid_points[np.newaxis, :, :]
                diff[:, :, 0] /= 0.014  # X 轴缩放
                diff[:, :, 1] /= 0.014  # Y 轴缩放
                diff[:, :, 2] /= 0.03   # Z 轴缩放
                
                # 计算距离的平方矩阵，避免使用平方根以提高效率
                dist_squared = np.sum(diff**2, axis=-1)
                
                # 标记距离小于等于 1.0 的点对（对应距离的平方 <= 1.0）
                violating_mask = np.triu(dist_squared <= 1.0, k=1)  # 只选取上三角部分，避免重复计算
                
                # 找到违反约束的点索引
                violating_indices = np.where(np.any(violating_mask, axis=0))[0]
                
                # 重新标记违法点以重新采样
                sample_points[indices[violating_indices]] = np.nan


        if np.all(~np.isnan(sample_points)):
            solutions.append(sample_points)
            # print("New solution has been added!")
        total_attempts += 1

    if len(solutions) < 100:
        print("无法在最大尝试次数内生成足够的有效采样点。")

    return solutions


def angle_between_1(v1, v2):
    """
    计算两个向量之间的角度（以弧度为单位），支持批量处理。

    参数：
    - v1: 形状为 (M, N, 3) 的 NumPy 数组
    - v2: 形状为 (M, N, 3) 的 NumPy 数组

    返回：
    - angles: 形状为 (M, N) 的 NumPy 数组，表示每对向量之间的角度
    """
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1, axis=2)
    norm_v2 = np.linalg.norm(v2, axis=2)

    # 防止除以零的情况
    zero_mask = (norm_v1 == 0) | (norm_v2 == 0)
    norm_v1[zero_mask] = 1
    norm_v2[zero_mask] = 1

    # 计算余弦值
    cos_theta = np.einsum('ijk,ijk->ij', v1, v2) / (norm_v1 * norm_v2)
    # 处理数值误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算角度
    angles = np.arccos(cos_theta)
    # 对于零向量，角度设为0.0
    angles[zero_mask] = 0.0
    return angles


def compute_delta_angles(front_keypoints, current_keypoints, back_keypoints, degrees=False):
    """
    输入形状为 (M, N, 3) 的前、当前、后关键点，输出角度变化。

    参数：
    - front_keypoints: 形状为 (M, N, 3) 的 NumPy 数组
    - current_keypoints: 形状为 (M, N, 3) 的 NumPy 数组
    - back_keypoints: 形状为 (M, N, 3) 的 NumPy 数组
    - degrees: 如果为 True, 返回的角度以度为单位；否则以弧度为单位。

    返回：
    - delta_angles: 形状为 (M, N) 的 NumPy 数组，表示角度变化
    """
    # 计算向量差
    v1 = current_keypoints - front_keypoints  # 形状为 (M, N, 3)
    v2 = back_keypoints - current_keypoints   # 形状为 (M, N, 3)

    # 计算角度变化
    delta_angles = angle_between_1(v1, v2)
    if degrees:
        delta_angles = np.degrees(delta_angles)

    return delta_angles


def positions_check(cur_positions, last_positions, next_positions):
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


def read_csv_file(file_path):
    if not os.path.exists(file_path):
        return None
    
    data_list = []
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data_list.append(row)
    return data_list