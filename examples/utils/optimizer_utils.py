import numpy as np
from typing import Type
from examples.utils.acoustic_utils import *


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
    '''
    input:
        - segment: (num_particle, lengths, 3)
    '''
    # 计算连续时间段的坐标差异
    displacement_diff = segment[:, 1:, :] - segment[:, :-1, :]  # 形状 (num_particle, lengths-1, 3)
    
    # 计算欧几里得距离（位移）
    displacements = np.linalg.norm(displacement_diff, axis=2)  # 形状 (num_particle, lengths-1)
    
    # 对每个时间段找到最大位移
    max_displacements = np.max(displacements, axis=0)  # 对粒子轴取最大值
    
    return max_displacements


def interpolate_positions(coords, delta_time_original=0.1, delta_time_new=0.01):
    '''
    coords: (num_keypoints, num_particles, 3)
    '''
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


def generate_solutions_whole_paths(n_particles, split_data, max_gorkov_idx, levitator):
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


def generate_solutions_segments(
        n_particles: int, 
        last_positions: np.array, 
        current_positions: np.array, 
        next_positions: np.array, 
        levitator: Type['top_bottom_setup'], 
        reach_index: np.array,
        num_solutions: int=10
    ):
    '''
    input:
        - last_positions: (num_particles, 3)
        - current_positions: (num_particles, 3)
        - next_positions: (num_particles, 3)
    '''    
    # 对最弱key points生成100个潜在solutions，并排序
    # candidate_solutions: (num_solutions, n_particles, 3)
    candidate_solutions = create_constrained_points_5(
        n_particles, 
        last_positions, 
        current_positions,
        next_positions,
        reach_index,
        num_solutions
    )
    if candidate_solutions is None:
        return None, None, None

    # 计算 candidate_solutions 的 Gorkov
    solutions_gorkov = levitator.calculate_gorkov_transposed(candidate_solutions)
    # 找出每个 candidate_solutions 的最大 Gorkov
    solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
    # 根据 Gorkov 对 candidate_solutions 从小到大排序
    sorted_indices = np.argsort(solutions_max_gorkov)
    sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]

    return candidate_solutions, sorted_indices, sorted_solutions_max_gorkov


def generate_solutions_single_frame(
        n_particles: int, 
        previous_frame: np.array, 
        current_frame: np.array, 
        levitator: Type['top_bottom_setup'], 
        reach_index: np.array,
        num_solutions: int=10
    ):
    '''
    为某个已知时刻生成solutions
    previous_frame: (num_particles, 3)
    current_frame: (num_particles, 3)
    '''
    # 对最弱key points生成100个潜在solutions，并排序
    candidate_solutions = create_constrained_points_single_frame(
        n_particles, 
        previous_frame, 
        current_frame,
        reach_index,
        num_solutions
    )
    if candidate_solutions is None:
        return None, None, None

    # 计算 candidate_solutions 的 Gorkov
    solutions_gorkov = levitator.calculate_gorkov(candidate_solutions)
    # 找出每个 candidate_solutions 的最大 Gorkov
    solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
    # 根据 Gorkov 对 candidate_solutions 从小到大排序
    sorted_indices = np.argsort(solutions_max_gorkov)
    sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]

    return candidate_solutions, sorted_indices, sorted_solutions_max_gorkov


def calculate_displacements(segment):
    '''
    输入：
        segment: 形状 (num_particle, n_keypoints, 3)
    输出：
        displacements: 形状 (num_particle, n_keypoints - 1)
    '''
    # 计算连续时间段的坐标差异
    displacement_diff = segment[:, 1:, :] - segment[:, :-1, :]  # 形状 (num_particle, lengths - 1, 3)
    
    # 计算欧几里得距离（位移）
    displacements = np.linalg.norm(displacement_diff, axis=2)

    # 输出每个时间段的所有位移
    return displacements


def calculate_velocities(dx_set, dt_set):
    '''
    输入：
        dx_set: (n_keypoints - 1, n_particles)
        dt_set: (n_keypoints - 1,)
    '''
    # 确保 t_set 的长度与 dx_set 对应    
    assert dt_set.shape[0] == dx_set.shape[0]

    # 使用广播直接计算平均速度
    # dt_set[:] 的形状是 (n_segments,)，广播到 (n_segments, n_particles)
    velocities_set = dx_set / dt_set[:, None]  

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