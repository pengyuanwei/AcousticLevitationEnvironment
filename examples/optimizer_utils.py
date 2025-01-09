import numpy as np


# 轨迹优化器


def max_displacement(segment):
    # segment 是 (3, num_particle, 3) 的形状
    # segment[时刻, 粒子, 坐标]

    # 初始化存储最大位移的列表
    max_displacements = []

    # 遍历两个时间段 t1->t2 和 t2->t3
    for t in range(2):  # 两个时间段
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


