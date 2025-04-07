import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing_2 import *
from examples.utils.path_smoothing_3 import *

# 将S2M2轨迹按delta t = 0.0032 进行插值, Naive计算，评估turing points的acoustic trapping quality

if __name__ == '__main__':
    n_particles = 8
    model_dir_1 = './experiments/experiment_20'
    model_name_1 = '20_19_98_99/planner_v2/S2M2'
    num_file = 30
    file_name_0 = 'S2M2_optimized_path'
    file_name_1 = 'path_S2M2'
    save_dir = os.path.join(model_dir_1, model_name_1)

    # TWGS, iterations=5
    levitator_TWGS = top_bottom_setup(n_particles, algorithm='Naive', iterations=5)
    gorkov_1 = []
    gorkov_2 = []

    for n in range(10):
        print(f'\n-----------------------The paths {n}-----------------------')

        csv_file = os.path.join(model_dir_1, model_name_1, f'{file_name_0}_{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue

        data_numpy, _ = read_paths(csv_data)

        # 每个粒子的轨迹长度相同
        paths_length = int(csv_data[1][1])
        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # When axis=2: keypoints_id, time, x, y, z
        split_data = data_numpy.reshape(-1, paths_length, 5)

        # 找出turning points
        turning_points = find_turning_points(split_data[:, :, 1:])
        print(turning_points)
        agent_new_indices_sorted = np.array(turning_points)
        print(agent_new_indices_sorted)

        # Calculate Gorkov
        gorkov_TWGS_1 = levitator_TWGS.calculate_gorkov(split_data[:, :, 2:]).T
        print(gorkov_TWGS_1.shape)
        for i in range(agent_new_indices_sorted.shape[0]):
            print(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]]) 
            gorkov_1.append(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]])

        
        print(f'\n-----------------------Original path---------------------')
        csv_file = os.path.join(model_dir_1, model_name_1, f'{file_name_1}_{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue

        data_numpy, _ = read_paths(csv_data)
        split_data, _ = process_data(data_numpy)
        split_data = np.concatenate((split_data[:, :, 0:1], split_data), axis=2)

        # 找出turning points
        turning_points = find_turning_points(split_data[:, :, 1:])
        print(turning_points)
        agent_new_indices_sorted = np.array(turning_points)
        print(agent_new_indices_sorted)

        # Calculate Gorkov
        gorkov_TWGS_1 = levitator_TWGS.calculate_gorkov(split_data[:, :, 2:]).T
        print(gorkov_TWGS_1.shape)
        for i in range(agent_new_indices_sorted.shape[0]):
            print(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]]) 
            gorkov_2.append(gorkov_TWGS_1[agent_new_indices_sorted[i][0]][agent_new_indices_sorted[i][1]])

    gorkov_mean_1 = np.mean(gorkov_1)
    gorkov_mean_2 = np.mean(gorkov_2)
    gorkov_std_1 = np.std(gorkov_1)
    gorkov_std_2 = np.std(gorkov_2)    

    print(f"优化后的平均gorkov: {gorkov_mean_1}")
    print(f"优化后的gorkov标准差: {gorkov_std_1}")
    print(f"原平均gorkov: {gorkov_mean_2}")
    print(f"原gorkov标准差: {gorkov_std_2}") 