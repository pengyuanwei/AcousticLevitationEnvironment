import numpy as np

def process_paths(csv_data, n_particles):
    # 每个粒子的轨迹长度相同
    max_length_int = csv_data[1][1] + 1

    # split_data_numpy的形状为(n_particles, n_keypoints, 5)
    # When axis=2: particle_id, time, x, y, z
    split_data_numpy = np.zeros((n_particles, np.max(max_length_int), 5))

    for j in range(len(split_data_numpy)):
        split_data_numpy[j, :max_length_int[j]] = data_numpy[:max_length_int[j]]

        if max_length_int[j] < np.max(max_length_int):
            last_particle_position = data_numpy[max_length_int[j]-1]
            split_data_numpy[j, -(np.max(max_length_int)-max_length_int[j]):] = last_particle_position

        data_numpy = data_numpy[max_length_int[j]:]

    return split_data_numpy


def process_paths(data_numpy, paths_length):
    # split_data_numpy的形状为(n_particles, n_keypoints, 5)
    # Axis 2: keypoints_idx, 时间累加值（时间列）, x, y, z
    split_data_numpy = data_numpy.reshape(-1, paths_length, 5)

    # 时间变化量：dt不变，不需要差分
    delta_time = split_data_numpy[0][1][1]
    # 将时间累加值替换为时间变化量
    split_data_numpy[:, 2:, 1] = delta_time 

    return split_data_numpy