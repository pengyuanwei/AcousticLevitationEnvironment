import os
import csv
import math
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Type
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from matplotlib.animation import PillowWriter

from acousticlevitationenvironment.utils import general_utils
from examples.utils.optimizer_utils import *


def save_path(path, save_dir, n_particles, delta_time, num, file_name='path'):
    '''
    input:
        - path: (paths_length, num_particles, 3)
    '''
    # (num_particles, paths_length, 3)
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


def save_path_v2(file_path, n_particles, split_data):
    '''
    input:
        - split_data: (num_particles, paths_length, 4)
    '''
    # 保存修改后的轨迹
    file_instance = open(file_path, "w", encoding="UTF8", newline='')
    csv_writer = csv.writer(file_instance)

    for i in range(n_particles):
        header = ['Agent ID', i]
        row_1 = ['Number of', split_data.shape[1]]

        csv_writer.writerow(header)
        csv_writer.writerow(row_1)

        rows = []
        for j in range(split_data.shape[1]):
            rows = [j, split_data[i][j][0], split_data[i][j][1], split_data[i][j][2], split_data[i][j][3]]
            csv_writer.writerow(rows)

    file_instance.close() 


def save_path_v3(file_path, n_particles, sum_t, sum_traj):
    '''
    input:
        - split_data: (num_particles, paths_length, 4)
    '''
    # 保存修改后的轨迹
    split_data = np.zeros((sum_traj.shape[0], sum_traj.shape[1], 4))
    split_data[:, :, 0] = sum_t
    split_data[:, :, 1:] = sum_traj

    file_instance = open(file_path, "w", encoding="UTF8", newline='')
    csv_writer = csv.writer(file_instance)

    for i in range(n_particles):
        header = ['Agent ID', i]
        row_1 = ['Number of', split_data.shape[1]]

        csv_writer.writerow(header)
        csv_writer.writerow(row_1)

        rows = []
        for j in range(split_data.shape[1]):
            rows = [j, split_data[i][j][0], split_data[i][j][1], split_data[i][j][2], split_data[i][j][3]]
            csv_writer.writerow(rows)

    file_instance.close()
    

def process_paths(data_numpy, paths_length):
    # split_data_numpy的形状为(n_particles, n_keypoints, 5)
    # Axis 2: keypoints_idx, 时间累加值（时间列）, x, y, z
    split_data_numpy = data_numpy.reshape(-1, paths_length, 5)

    # 时间变化量：dt不变，不需要差分
    delta_time = split_data_numpy[0][1][1]
    # 将时间累加值替换为时间变化量
    split_data_numpy[:, 2:, 1] = delta_time 

    return split_data_numpy


def generate_global_paths(
        env, 
        agent, 
        n_particles: int, 
        max_timesteps: int
    ):
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

    # (num_particles, paths_length, 3)
    paths_array = np.array(paths)
    # 转置后形状为 (paths_length, n_particles, 3)
    original_paths = np.transpose(paths_array, (1, 0, 2))

    return original_paths, truncated


def generate_paths_smoothing(
        env, 
        agent, 
        n_particles: int, 
        max_timesteps: int, 
        levitator: Type['top_bottom_setup'],
        debug: bool=False
    ):
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

    # (num_particles, paths_length, 3)
    paths_array = np.array(paths)
    last_unique_indexs = []
    fixed_locations = np.ones((paths_array.shape[0], paths_array.shape[1], 1))    
    if not truncated:
        for i in range(n_particles):
            last_unique_index = paths_array.shape[1] - 1
            # 从末尾开始，找到最后一个与前一个不同的点的索引
            while last_unique_index > 0 and np.allclose(paths_array[i][last_unique_index], paths_array[i][last_unique_index - 1]):
                last_unique_index -= 1
            last_unique_indexs.append(last_unique_index)
            fixed_locations[i, :last_unique_index+1] = 0.0
        # 转置为 (paths_length, num_particles, 1)
        fixed_locations = np.transpose(fixed_locations, (1, 0, 2))

        # 修正倒数第二个keypoint
        for i in range(n_particles):
            # 所有不重复的坐标
            key_points_unique = paths_array[i][:last_unique_indexs[i] + 1]
            vectors = key_points_unique[1:] - key_points_unique[:-1]
            if len(vectors) > 1:
                angle = angle_between(vectors[-1], vectors[-2])
                if angle >= 1.0:
                    print(f"Path smoothing: the {i}th particle, the {len(key_points_unique)-1}th keypoints, direction change: {angle}")
                    last_positions = paths_array[:, len(key_points_unique)-3, :]
                    positions = paths_array[:, len(key_points_unique)-2, :]
                    next_positions = paths_array[:, len(key_points_unique)-1, :]
                    paths_array[:, len(key_points_unique)-2, :] = gorkov_correction(
                        n_particles, last_positions, positions, next_positions, levitator, fixed_locations[len(key_points_unique)-1]
                    )
                    if debug:
                        key_points_unique = paths_array[i][:last_unique_indexs[i] + 1]
                        vectors = key_points_unique[1:] - key_points_unique[:-1]
                        angle = angle_between(vectors[-1], vectors[-2])
                        print(f"Path smoothing: the corrected direction change: {angle}")

    # 转置后形状为 (paths_length, n_particles, 3)
    original_paths = np.transpose(paths_array, (1, 0, 2))
    return original_paths, last_unique_indexs, fixed_locations, truncated


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


def generate_replan_paths_smoothing(        
        env, 
        agent, 
        n_particles: int, 
        max_timesteps: int, 
        points: np.array,
        levitator: Type['top_bottom_setup']=None,
        debug: bool=False
    ):
    '''
    重新规划
    input:
        - points: 前一个规划器的失败轨迹，用于获取起点和终点
        - levitator: 如果是None, 则进行无声学的路径平滑
    '''
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

    # (num_particles, paths_length, 3)
    paths_array = np.array(paths)
    last_unique_indexs = []
    fixed_locations = np.ones((paths_array.shape[0], paths_array.shape[1], 1))    
    if not truncated:
        for i in range(n_particles):
            last_unique_index = paths_array.shape[1] - 1
            # 从末尾开始，找到最后一个与前一个不同的点的索引
            while last_unique_index > 0 and np.allclose(paths_array[i][last_unique_index], paths_array[i][last_unique_index - 1]):
                last_unique_index -= 1
            last_unique_indexs.append(last_unique_index)
            fixed_locations[i, :last_unique_index+1] = 0.0
        # 转置为 (paths_length, num_particles, 3)
        fixed_locations = np.transpose(fixed_locations, (1, 0, 2))

        # 修正倒数第二个keypoint
        for i in range(n_particles):
            # 所有不重复的坐标
            key_points_unique = paths_array[i][:last_unique_indexs[i] + 1]
            vectors = key_points_unique[1:] - key_points_unique[:-1]
            if len(vectors) > 1:
                angle = angle_between(vectors[-1], vectors[-2])
                if angle >= 1.0:
                    print(f"Path smoothing: the {i}th particle, the {len(key_points_unique)-1}th keypoints, direction change: {angle}")
                    last_positions = paths_array[:, len(key_points_unique)-3, :]
                    positions = paths_array[:, len(key_points_unique)-2, :]
                    next_positions = paths_array[:, len(key_points_unique)-1, :]
                    if levitator is not None:
                        paths_array[:, len(key_points_unique)-2, :] = gorkov_correction(
                            n_particles, last_positions, positions, next_positions, levitator, fixed_locations[len(key_points_unique)-1]
                        )
                    else:
                        paths_array[:, len(key_points_unique)-2, :] = nonAcoustic_correction(
                            n_particles, last_positions, positions, next_positions, fixed_locations[len(key_points_unique)-1]
                        )                        
                    if debug:
                        key_points_unique = paths_array[i][:last_unique_indexs[i] + 1]
                        vectors = key_points_unique[1:] - key_points_unique[:-1]
                        angle = angle_between(vectors[-1], vectors[-2])
                        print(f"Path smoothing: the corrected direction change: {angle}")

    # 转置后形状为 (paths_length, n_particles, 3)
    original_paths = np.transpose(paths_array, (1, 0, 2))
    return original_paths, last_unique_indexs, fixed_locations, truncated


def show(num_particles, particle_paths):
    '''
    input:
        - particle_paths: (num_particles, paths_length, 3)
    '''
    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 初始化粒子和轨迹
    particles, = ax.plot([], [], [], 'ro', markersize=8)
    trajectories = [ax.plot([], [], [], lw=1)[0] for _ in range(num_particles)]

    # 设置坐标轴范围
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"{num_particles} Particles Moving in Different Trajectories")

    # 初始化函数
    def init():
        particles.set_data([], [])
        particles.set_3d_properties([])
        for trajectory in trajectories:
            trajectory.set_data([], [])
            trajectory.set_3d_properties([])
        return [particles] + trajectories

    # 更新函数
    def update(frame):
        x, y, z = particle_paths[:, frame, 0], particle_paths[:, frame, 1], particle_paths[:, frame, 2]
        particles.set_data(x, y)
        particles.set_3d_properties(z)
        
        for i, trajectory in enumerate(trajectories):
            trajectory.set_data(particle_paths[i, :frame+1, 0], particle_paths[i, :frame+1, 1])
            trajectory.set_3d_properties(particle_paths[i, :frame+1, 2])
        
        return [particles] + trajectories

    # 显示图形，并等待用户交互
    plt.draw()  # 先绘制静态图像
    plt.waitforbuttonpress()  # 等待用户点击窗口或按键
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=particle_paths.shape[1], init_func=init, blit=False, interval=50, repeat=False)
    # 保存动画为 MP4
    ani.save("animation.gif", writer=PillowWriter(fps=20))
    plt.show()


def path_smoothing(n_particles, paths_array, num_interpolations: int) -> np.ndarray:
    '''
    Paths smoothing
    input:
        - paths_array: (num_particles, paths_length, 3)
    '''
    smoothed_paths = []
    for i in range(n_particles):
        smoothed_path = cubic_spline_interpolation_3d(paths_array[i], num_interpolations)
        smoothed_paths.append(smoothed_path)
    # 找到所有路径中的最大长度
    max_length = max(path.shape[0] for path in smoothed_paths)
    # 对每条路径进行补齐
    padded_paths = [pad_path(path, max_length) for path in smoothed_paths]

    return np.array(padded_paths)


def cubic_spline_interpolation_3d(key_points: np.array, num_interpolations: int=5):
    """
    使用三次样条插值生成平滑的三维路径。

    参数:
        key_points (list or np.array): 输入的关键点，形状为 (N, 3)，表示 N 个点的 (x, y, z) 坐标。
        num_points (int): 生成的平滑路径中的插值参数，总点数为 num_key_points * num_points + 1

    返回:
        smooth_path (np.array): 平滑路径的坐标，形状为 (num_points, 3)。
    """
    if key_points.shape[1] != 3:
        raise ValueError("关键点必须是三维坐标 (x, y, z)，形状为 (N, 3)。")

    # 移除末尾重复的点
    key_points_unique = remove_trailing_duplicates(key_points)
    if key_points_unique.shape[0] < 2:
        raise ValueError("有效的关键点数量不足以进行插值。")

    # 提取 x, y, z 坐标
    x = key_points_unique[:, 0]
    y = key_points_unique[:, 1]
    z = key_points_unique[:, 2]

    # 使用关键点索引作为参数 t , 以实现keypoint之间的固定数量插值
    t = np.arange(len(x))  # t = [0, 1, 2, ..., N-1]

    # 创建三次样条插值函数
    cs_x = CubicSpline(t, x, bc_type='clamped')  # x 坐标的样条函数
    cs_y = CubicSpline(t, y, bc_type='clamped')  # y 坐标的样条函数
    cs_z = CubicSpline(t, z, bc_type='clamped')  # z 坐标的样条函数

    # 生成平滑路径
    t_smooth = np.linspace(t[0], t[-1], (len(t) - 1) * num_interpolations + 1)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)

    # 返回平滑路径
    smooth_path = np.column_stack((x_smooth, y_smooth, z_smooth))
    return smooth_path


def remove_trailing_duplicates(key_points: np.array) -> np.array:
    """
    移除路径末尾连续的重复点，返回非重复部分的关键点数组。
    """
    if key_points.shape[0] < 2:
        return key_points

    # 从末尾开始，找到最后一个与前一个不同的点
    last_unique_index = key_points.shape[0] - 1
    while last_unique_index > 0 and np.array_equal(key_points[last_unique_index], key_points[last_unique_index - 1]):
        last_unique_index -= 1

    # 包含最后一个唯一的点（索引 last_unique_index），但不包含后面的重复点
    return key_points[:last_unique_index + 1]


def pad_path(path: np.ndarray, target_length: int) -> np.ndarray:
    """
    对单个路径进行补齐，补齐的值为路径最后的坐标。

    参数:
        path (np.ndarray): 输入路径，形状为 (L, 3)。
        target_length (int): 目标长度。

    返回:
        np.ndarray: 补齐后的路径，形状为 (target_length, 3)。
    """
    current_length = path.shape[0]
    if current_length < target_length:
        # 获取最后一个点的坐标
        pad_value = path[-1]
        # 构造补齐的数组，重复最后一个点
        pad_array = np.tile(pad_value, (target_length - current_length, 1))
        # 将原路径和补齐数组拼接
        path = np.vstack((path, pad_array))
    return path


def angle_between(v1, v2):
    """
    计算两个向量之间的角度（以弧度为单位）
    input:
        - v1, v2: (3, )
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


def gorkov_correction(
        num_particles: int, 
        last_positions: np.array, 
        current_positions: np.array, 
        next_positions: np.array, 
        levitator: Type['top_bottom_setup'],
        fixed_index: np.array
    ):
    '''
    input:
        - last_positions: (num_particles, 3)
        - current_positions: (num_particles, 3)
        - next_positions: (num_particles, 3)
        - fixed_index: (num_particles, fixed_or_not)
    '''
    gorkov = levitator.calculate_gorkov_single_state(current_positions)
    max_gorkov = np.max(gorkov, axis=0)
    # candidate_solutions: (num_solutions, n_particles, 3)
    candidate_solutions, sorted_indices, sorted_solutions_max_gorkov = generate_solutions_segments(
        num_particles, last_positions, current_positions, next_positions, levitator, fixed_index, num_solutions=200
    )
    if candidate_solutions is not None:
        if sorted_solutions_max_gorkov[0] > max_gorkov:
            print('Path smoothing: no candidate has better Gorkov than original!')

        # 依次取出 candidate_solutions，先检查是否Gorkov更好，再检查是否满足距离约束
        # 分别求出前后两个 segment 的最大位移，用于缩放时间
        for i in range(candidate_solutions.shape[0]):
            re_plan_segment = np.concatenate([
                last_positions[np.newaxis, :, :], 
                candidate_solutions[sorted_indices[i]:sorted_indices[i]+1, :, :],
                next_positions[np.newaxis, :, :]
            ])

            for k in range(2):
                segment = re_plan_segment[k:(k+2)]
                # 通过插值检查是否碰撞
                interpolated_coords = interpolate_positions(segment)
                
                for j in range(interpolated_coords.shape[0]):
                    collision = safety_area(num_particles, interpolated_coords[j])
                    if np.any(collision != 0):
                        break
                if np.any(collision != 0):
                    break

            if np.all(collision == 0):
                print(f"Path smoothing: final non-collision best candidate solution: No.{i}")
                if sorted_solutions_max_gorkov[i] > max_gorkov:
                    print('Path smoothing: worse Gorkov than original!')
                current_positions = candidate_solutions[sorted_indices[i], :, :]
                break
    else:
        print("Path smoothing: 倒数第二个keypoint修正失败！")

    return current_positions


def nonAcoustic_correction(
        num_particles: int, 
        last_positions: np.array, 
        current_positions: np.array, 
        next_positions: np.array, 
        fixed_index: np.array,
        num_solutions: int=200
    ):
    '''
    input:
        - last_positions: (num_particles, 3)
        - current_positions: (num_particles, 3)
        - next_positions: (num_particles, 3)
        - fixed_index: (num_particles, fixed_or_not)
    '''
    # 对最弱key points生成100个潜在solutions，并排序
    fixed_points = {}
    for i in range(num_particles):
        if fixed_index[i]:
            fixed_points[i] = np.array([current_positions[i][0], current_positions[i][1], current_positions[i][2]])

    cube_size = np.linalg.norm((next_positions - last_positions), axis=1)
    search_area_center = (next_positions + last_positions) / 2.0
    x_min, x_max, y_min, y_max, z_min, z_max = [-0.06, 0.06, -0.06, 0.06, -0.06+0.12, 0.06+0.12]
    solutions = []
    attempts = 0
    while len(solutions) < num_solutions:
        searched_points = {}
        for i in range(num_particles): 
            if not fixed_index[i]:
                # 在search area内随机生成一个点
                movement = np.random.uniform(-cube_size[i]/(10.0), cube_size[i]/(10.0), 3)
                point = np.array([min(max(search_area_center[i][0] + movement[0], x_min), x_max), 
                                min(max(search_area_center[i][1] + movement[1], y_min), y_max), 
                                min(max(search_area_center[i][2] + movement[2], z_min), z_max)])
                
                # 检查与已生成点之间的椭球体距离约束
                valid_fixed = is_valid_distance(point, fixed_points.values())
                valid_searched = is_valid_distance(point, searched_points.values())
                if valid_fixed and valid_searched:
                    searched_points[i] = point
                
        if (len(fixed_points) + len(searched_points)) == num_particles:
            new_points = fixed_points | searched_points
            sorted_keys = sorted(new_points.keys())
            points_array = np.array([new_points[key] for key in sorted_keys])
            solutions.append(points_array)
        
        attempts += 1
        if attempts >= num_solutions*100:
            print(f"Non-acoustic path smoothing: 达到最大迭代次数, candidate solution 数量: {len(solutions)}。")
            break

    if len(solutions) > 0:
        # (num_solutions, num_particles, 3)
        candidate_solutions = np.array(solutions)
        for i in range(candidate_solutions.shape[0]):
            # 检查solution是否满足距离约束
            # 分别求出前后两个 segment 的最大位移，用于缩放时间
            re_plan_segment = np.concatenate([last_positions[np.newaxis, :, :], candidate_solutions[i:i+1, :, :], next_positions[np.newaxis, :, :]])

            for k in range(2):
                segment = re_plan_segment[k:(k+2)]
                interpolated_coords = interpolate_positions(segment)
                
                for j in range(interpolated_coords.shape[0]):
                    collision = safety_area(num_particles, interpolated_coords[j])
                    if np.any(collision != 0):
                        break
                if np.any(collision != 0):
                    break

            if np.all(collision == 0):
                print(f"Non-acoustic path smoothing: final non-collision best candidate solution: No.{i}")
                current_positions = candidate_solutions[i, :, :]
                break
    else:
        print("Non-acoustic path smoothing: 倒数第二个keypoint修正失败！")

    return current_positions


def uniform_accelerated_interpolation(initial_paths: np.array, total_time: np.array, last_unique_indexs: list):
    '''
    对提前到达终点的粒子最后一段位移进行匀减速插值，使得所有粒子同步到达终点：
    input:
        - initial_paths: 初始轨迹, 形状为 (num_particles, paths_length, 3)
        - total_time: 轨迹的累计时间, 形状为 (paths_length,)
        - last_unique_indexs: 每个粒子到达终点时的索引, 长度为 num_particles
    output:
        - corrected_paths: 修正后的轨迹, 形状为 (num_particles, new_paths_length, 3)
        - new_total_time: 轨迹的新的累计时间, 形状为 (new_paths_length,)
    '''


def uniform_accelerated_interpolation(initial_paths: np.array, total_time: np.array, last_unique_indexs: list):
    '''
    对提前到达终点的粒子最后一段位移进行匀减速插值，使得所有粒子同步到达终点：
    input:
        - initial_paths: 初始轨迹, 形状为 (num_particles, paths_length, 3)
        - total_time: 轨迹的累计时间, 形状为 (paths_length,)
        - last_unique_indexs: 每个粒子到达终点时的索引, 长度为 num_particles
    output:
        - corrected_paths: 修正后的轨迹, 形状为 (num_particles, new_paths_length, 3)
    '''
    num_particles, paths_length, _ = initial_paths.shape
    T_end = total_time[-1]
    
    # 初始化修正后的轨迹，初步复制原始轨迹
    corrected_paths = np.copy(initial_paths)
    
    # 对每个粒子进行处理：
    for i in range(num_particles):
        last_idx = last_unique_indexs[i]
        # 如果该粒子已经在最后时刻到达，则无需插值
        if last_idx >= paths_length - 1:
            continue
        
        # 提前到达的粒子：保留前部分轨迹不变，从 last_idx 开始进行匀减速插值
        p0 = initial_paths[i, last_idx-1, :]       # 插值起点
        destination = initial_paths[i, -1, :]      # 插值终点
        t0 = total_time[last_idx-1]                # 起始时间
        delta_T = T_end - t0                       # 剩余时间
        
        # 对 last_idx 到最后的每个时间点，用归一化时间计算新位置
        for j in range(last_idx-1, paths_length):
            tau = (total_time[j] - t0) / delta_T  # 归一化时间, 范围 [0, 1]
            # 匀减速插值公式，保证起点位置为 p0，终点位置为 destination 且末端速度为0
            corrected_paths[i, j, :] = p0 + (destination - p0) * (2 * tau - tau ** 2)
        
    return corrected_paths