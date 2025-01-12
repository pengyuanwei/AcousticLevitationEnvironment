import numpy as np
from examples.acoustic_utils import *


# 轨迹优化器


def max_displacement(segment, num=2):
    # segment 是 (num+1, num_particle, 3) 的形状
    # segment[时刻, 粒子, 坐标]

    # 初始化存储最大位移的列表
    max_displacements = []

    # 遍历两个时间段 t1->t2 和 t2->t3
    for t in range(num):  # 两个时间段
        # 计算8个粒子在时间段t到t+1之间的位移
        displacements = np.sqrt(
            (segment[t+1, :, 0] - segment[t, :, 0])**2 +  # x坐标差
            (segment[t+1, :, 1] - segment[t, :, 1])**2 +  # y坐标差
            (segment[t+1, :, 2] - segment[t, :, 2])**2    # z坐标差
        )
        
        # 找到当前时间段的最大位移
        max_displacement = np.max(displacements)
        max_displacements.append(max_displacement)

    # 输出每个时间段的最大位移
    return np.array(max_displacements)


def max_displacement_v2(segment):
    # segment 是形状 (num_particle, lengths, 3)

    # 计算连续时间段的坐标差异
    displacement_diff = segment[:, 1:, :] - segment[:, :-1, :]  # 形状 (num_particle, num, 3)
    
    # 计算欧几里得距离（位移）
    displacements = np.linalg.norm(displacement_diff, axis=2)  # 形状 (num_particle, num)
    
    # 对每个时间段找到最大位移
    max_displacements = np.max(displacements, axis=0)  # 对粒子轴取最大值
    
    return max_displacements


def interpolate_positions(coords, delta_time_original=0.1, delta_time_new=0.01):
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
            dist_square = (x - coords[j][0])**2/0.014**2 + (y - coords[j][1])**2/0.014**2 + (z - coords[j][2])**2/0.03**2
            if dist_square <= 1.0:
                collision[i] = 1.0
                collision[j] = 1.0

    return collision


def generate_solutions(n_particles, split_data, max_gorkov_idx, levitator):
    # 对最弱key points生成100个潜在solutions，并排序
    candidate_solutions = np.transpose(
        create_constrained_points_1(
            n_particles, 
            split_data[:, max_gorkov_idx, 2:], 
            split_data[:, max_gorkov_idx-1, 2:], 
            split_data[:, max_gorkov_idx+1, 2:]
        ), 
        (1, 0, 2)
    )

    # 计算 candidate_solutions 的 Gorkov
    solutions_gorkov = levitator.calculate_gorkov(candidate_solutions)
    # 找出每个 candidate_solutions 的最大 Gorkov
    solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
    # 根据 Gorkov 对 candidate_solutions 从小到大排序
    sorted_indices = np.argsort(solutions_max_gorkov)
    sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]

    return candidate_solutions, sorted_indices, sorted_solutions_max_gorkov


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


def calculate_dx_v2(segment):
    # segment 是形状 (num_particle, lengths, 3)

    # 计算连续时间段的坐标差异
    displacement_diff = segment[:, 1:, :] - segment[:, :-1, :]  # 形状 (num_particle, num, 3)
    
    # 计算欧几里得距离（位移）
    displacements = np.linalg.norm(displacement_diff, axis=2)  # 形状 (num_particle, num)

    # 输出每个时间段的所有位移
    return displacements


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


def calculate_mean_v_v2(dx, t_set):
    # dx: (n_segments, n_particles)
    # t_set: (n_keypoints,)
    # 确保 t_set 的长度与 n_segments+1 对应
    assert t_set.shape[0] == dx.shape[0] + 1

    # 使用广播直接计算平均速度
    # # t_set[1:] 的形状是 (n_segments,)，广播到 (n_segments, n_particles)
    velocities_set = dx / t_set[1:, None]  

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