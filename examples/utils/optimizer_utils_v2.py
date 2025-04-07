import numpy as np
from typing import Type
from scipy.interpolate import interp1d
from examples.utils.acoustic_utils_v2 import *
from examples.utils.path_smoothing_2 import *


# 轨迹优化器


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


def linear_interpolation(data, k):
    """
    对三维坐标数据进行线性插值。
    
    参数：
    - data: numpy 数组，形状为 (m, n, 3)，表示 m 组数据，每组 n 个点，每个点是 3D 坐标。
    - k: int，相邻坐标之间插值的点数（不包含起点）。
    
    返回：
    - interpolated_data: numpy 数组，形状为 (m, new_n, 3)，插值后的数据。
    """
    m, n, _ = data.shape  # 读取 batch size 和坐标点数量
    new_n = (n - 1) * (k + 1) + 1  # 计算插值后总点数
    interpolated_data = np.zeros((m, new_n, 3))  # 预分配空间
    
    for i in range(m):  # 遍历 batch 维度
        interp_points = [data[i, 0]]  # 先添加第一个点，避免重复
        for j in range(n - 1):  # 遍历相邻点对
            p1, p2 = data[i, j], data[i, j + 1]
            interp = np.linspace(p1, p2, k + 2, endpoint=True)[1:]  # 生成 k+1 个插值点，去掉第一个 p1
            interp_points.append(interp)
        interpolated_data[i] = np.vstack(interp_points)  # 合并所有插值结果
    
    return interpolated_data


def safety_area(n_particles, coords):
    '''
    coords: (n_particles, 3)
    '''
    collision = np.zeros(n_particles)
    for i in range(n_particles):
        x, y, z = [coords[i][0], coords[i][1], coords[i][2]]            
        for j in range(i+1, n_particles):
            dist_square = (x - coords[j][0])**2/0.014**2 + (y - coords[j][1])**2/0.014**2 + (z - coords[j][2])**2/0.03**2
            if dist_square <= 1.0:
                collision[i] = 1.0
                collision[j] = 1.0

    return collision


def generate_solutions_whole_paths_v0(n_particles, split_data, max_gorkov_idx, levitator):
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


def generate_solutions_whole_paths_v1(n_particles, split_data, max_gorkov_idx, levitator):
    # 对最弱key points生成100个潜在solutions，并排序
    candidate_solutions = create_constrained_points_v2(
            n_particles, 
            split_data[:, max_gorkov_idx, 2:], 
            split_data[:, max_gorkov_idx-1, 2:], 
            split_data[:, max_gorkov_idx+1, 2:]
    )

    # 计算 candidate_solutions 的 Gorkov
    solutions_gorkov = levitator.calculate_gorkov(candidate_solutions)
    # 找出每个 candidate_solutions 的最大 Gorkov
    solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
    # 根据 Gorkov 对 candidate_solutions 从小到大排序
    sorted_indices = np.argsort(solutions_max_gorkov)
    sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]

    return candidate_solutions, sorted_indices, sorted_solutions_max_gorkov


def generate_solutions_whole_paths_v2(
        n_particles, 
        split_data, 
        max_gorkov_idx, 
        levitator,
        delta_time
    ):
    '''
    插值, 依次计算每个时刻的Gorkov
    '''
    # 对最弱key points生成100个潜在solutions，并排序
    candidate_solutions = create_constrained_points_v2(
            n_particles, 
            split_data[:, max_gorkov_idx, 2:], 
            split_data[:, max_gorkov_idx-1, 2:], 
            split_data[:, max_gorkov_idx+1, 2:]
    )

    # 计算 candidate_solutions 的 Gorkov
    solutions_max_gorkov = np.zeros((candidate_solutions.shape[1], ))
    for i in range(candidate_solutions.shape[1]):
        _, _, _, paths, _ = uniform_velocity_interpolation_v2(
            start=candidate_solutions[:, i, :], end=split_data[:, max_gorkov_idx+1, 2:], total_time=delta_time, dt=0.0032, velocities=0.0
        )
        solution_gorkov = levitator.calculate_gorkov(paths)
        # 找出每个 candidate_solutions 的最大 Gorkov
        solutions_max_gorkov[i] = np.max(solution_gorkov)
        if i % 10 == 0:
            print(i)
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
        num_solutions: int=10, 
        search_factor: float=10
    ):
    '''
    input:
        - last_positions: (num_particles, 3)
        - current_positions: (num_particles, 3)
        - next_positions: (num_particles, 3)
        - reach_index: (num_particles, fixed_or_not)
    '''    
    # 对最弱key points生成100个潜在solutions，并排序
    # candidate_solutions: (num_solutions, n_particles, 3)
    candidate_solutions = create_constrained_points_6(
        n_particles, 
        last_positions, 
        current_positions,
        next_positions,
        reach_index,
        num_solutions,
        search_factor
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


def generate_solutions_segments_v2(
        n_particles: int, 
        last_positions: np.array, 
        current_positions: np.array, 
        next_positions: np.array, 
        levitator: Type['top_bottom_setup'], 
        reach_index: np.array,
        num_solutions: int=10, 
        search_factor: float=10
    ):
    '''
    找出unfixed粒子的最大Gorkov
    Input:
         - last_positions: (num_particles, 3)
         - current_positions: (num_particles, 3)
         - next_positions: (num_particles, 3)
         - reach_index: (num_particles, fixed_or_not)
    Output:
         - candidate_solutions: (num_solutions, num_particles, 3)
         - sorted_indices
         - sorted_solutions_max_gorkov
    '''    
    # 将 reach_index 转换为布尔型掩码
    mask = reach_index == 1    
    # 对最弱key points生成100个潜在solutions，并排序
    # candidate_solutions: (num_solutions, n_particles, 3)
    candidate_solutions = create_constrained_points_6(
        n_particles, 
        last_positions, 
        current_positions,
        next_positions,
        mask,
        num_solutions,
        search_factor
    )
    if candidate_solutions is None:
        return None, None, None

    # 计算 candidate_solutions 的 Gorkov
    solutions_gorkov = levitator.calculate_gorkov_transposed(candidate_solutions)
    step1 = np.expand_dims(mask, axis=0)  # (1, n_particles, 1)
    broadcasted = np.broadcast_to(step1, (num_solutions, n_particles, 1))  # 广播
    broadcasted_mask = broadcasted.squeeze()  # 去掉维度 1，得到 (20, 8)    # # 利用 np.where 将 mask 为 False 的位置赋值为 -∞，

    # 这样在计算最大值时，这些位置不会对结果产生影响
    masked_gorkov = np.where(broadcasted_mask, solutions_gorkov, -np.inf)
    # 找出每个 candidate_solutions 的最大 Gorkov
    solutions_max_gorkov = np.max(masked_gorkov, axis=1)
    # 根据 Gorkov 对 candidate_solutions 从小到大排序
    sorted_indices = np.argsort(solutions_max_gorkov)
    sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]

    return candidate_solutions, sorted_indices, sorted_solutions_max_gorkov


def generate_solutions_single_frame(
        n_particles: int, 
        last_positions: np.array, 
        positions: np.array, 
        last_displacements: np.array, 
        displacements: np.array, 
        levitator: Type['top_bottom_setup'], 
        reach_index: np.array,
        num_solutions: int=10
    ):
    '''
    为某个已知时刻生成solutions
    last_positions: (num_particles, 3)
    positions: (num_particles, 3)
    last_displacements: (num_particles, 3)
    displacements: (num_particles, 3)
    '''
    # 对最弱key points生成100个潜在solutions，并排序
    candidate_solutions = create_constrained_points_single_frame(
        n_particles, 
        last_positions,
        positions,
        last_displacements,
        displacements,
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


def process_data(data_numpy: np.array, debug: bool=False) -> np.array:
    '''
    预处理轨迹
    Output:
        - np.array: (num_particles, path_length, 5), 时刻, 坐标
        - np.array: (path_length, num_particles), keypoints标记
    '''
    # 分组存储
    groups = []
    current_group = []
    time_records = set()

    for row in data_numpy:
        time_records.add(row[1])  # 记录时间序列
        if row[0] == 0 and current_group:
            groups.append(np.array(current_group))
            current_group = []
        current_group.append(row)

    if current_group:
        groups.append(np.array(current_group))

    # 线性插值
    time_records = sorted(time_records)
    interpolated_groups = []
    keypoint_flags = []

    for group in groups:
        times = group[:, 1]
        locations = group[:, 2:]
        original_times = set(times)  # 记录原始时间点
        
        interpolated_group = []
        current_keypoint_flags = []  # 记录哪些是原始点，哪些是插值点
        
        for i in range(locations.shape[1]):  # 对每个location维度插值
            interp_func = interp1d(times, locations[:, i], kind='linear', 
                       fill_value=(locations[0, i], locations[-1, i]), 
                       bounds_error=False)
            interpolated_values = interp_func(time_records)
            interpolated_group.append(interpolated_values)
        
        # 记录关键点标记：原始数据中端点为2, 除端点以外的所有Keypoints为1，其它均为0
        min_time, max_time = min(original_times), max(original_times)
        for t in time_records:
            if t == min_time or t >= max_time:
                current_keypoint_flags.append(2)
            elif t in original_times and t != min_time and t != max_time:
                current_keypoint_flags.append(1)
            else:
                current_keypoint_flags.append(0)
        
        interpolated_group = np.column_stack([time_records] + interpolated_group)
        interpolated_groups.append(interpolated_group)
        keypoint_flags.append(current_keypoint_flags)

    if debug:
        # 打印分组结果
        for i, group in enumerate(interpolated_groups):
            print(f"Interpolated Group {i+1}:")
            print(group, "\n")
        # 打印所有记录的时间序列
        print("Recorded Time Sequences:")
        print(time_records)

    return np.array(interpolated_groups), np.array(keypoint_flags).T