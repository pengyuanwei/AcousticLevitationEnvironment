import os
import time
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *


# Do random search to find better Gorkov points for finded paths(keypoints)
# Modified based on the trajectory_optimization_10.py: 先处理位置，最后统一处理时间序列


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99'
    num_file = 1
    file_name = 'optimised_data'
    levitator = top_bottom_setup(n_particles, algorithm='Naive')

    computation_time = []
    for n in range(num_file):
        print(f'\n-----------------------The paths {n}-----------------------')
        # 记录当前循环的开始时间
        start_time = time.time()  

        # csv_data是list，其中的元素是list，每个子list保存了每一行的数据
        csv_file = os.path.join(global_model_dir_1, model_name, f'path{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data is None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue

        data_numpy, include_NaN = read_paths(csv_data)
        if include_NaN:
            print(f"Skipping file due to NaN values: {csv_file}")
            continue

        # 每个粒子的轨迹长度相同
        paths_length = int(csv_data[1][1])
        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # Axis 2: keypoints_idx, 时间累加值（时间列）, x, y, z
        split_data = data_numpy.reshape(-1, paths_length, 5)


        # 使用随机搜索来优化最弱Gorkov的timesteps
        for m in range(1):
            # Calculate Gorkov
            gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
            max_gorkov_idx, max_gorkov = calculate_max_gorkov(gorkov)

            # Print initial Gorkov values
            if m == 0:
                print('Max Gorkov before random search:', max_gorkov)
            print(f'\n-----------------------The iteration {m}-----------------------')
            print("Worst gorkov idx:", max_gorkov_idx)
            print("Worst gorkov value:", max_gorkov[max_gorkov_idx])
            
            candidate_solutions, sorted_indices, sorted_solutions_max_gorkov = generate_solutions(
                n_particles, split_data, max_gorkov_idx, levitator
            )

            # 依次取出 candidate_solutions，先检查是否Gorkov更好，再检查是否满足距离约束
            # 分别求出前后两个 segment 的最大位移，用于缩放时间
            re_plan_segment = np.transpose(split_data[:, max_gorkov_idx-1:max_gorkov_idx+2, 2:], (1, 0, 2))

            for i in range(candidate_solutions.shape[1]):
                # 如果 candidate_solutions 的 Gorkov 比原坐标的更差，则 break
                if sorted_solutions_max_gorkov[i] > max_gorkov[max_gorkov_idx]:
                    print('No better candidate than original!')
                    break

                re_plan_segment[1:2, :, :] = np.transpose(
                    candidate_solutions[:, sorted_indices[i]:sorted_indices[i]+1, :], 
                    (1, 0, 2)
                )

                for k in range(2):
                    segment = re_plan_segment[k:(k+2)]
                    interpolated_coords = interpolate_positions(segment)
                    
                    for j in range(interpolated_coords.shape[0]):
                        collision = safety_area(n_particles, interpolated_coords[j])
                        if np.any(collision != 0):
                            break
                    if np.any(collision != 0):
                        print("Collision!")
                        break

                if np.all(collision == 0):
                    print("Best candidate idx (start from 0):", i)
                    split_data[:, max_gorkov_idx, 2:] = np.copy(candidate_solutions[:, sorted_indices[i], :])
                    break

        # Calculate the Gorkov again
        gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
        max_gorkov = np.max(gorkov, axis=1)
        print('\nMax Gorkov after random search:', max_gorkov)


        # # 检查所有关键点是否在圆圈内，如果不在，对其进行优化
        # for m in range(1, split_data.shape[1] - 1):
        #     if positions_check(split_data[:, m, 2:], split_data[:, m-1, 2:], split_data[:, m+1, 2:]):
        #         print(f'\n-----------------------The point {m}-----------------------')
        #         # Calculate the Gorkov
        #         gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
        #         max_gorkov = np.max(gorkov, axis=1)

        #         candidate_solutions, sorted_indices, sorted_solutions_max_gorkov = generate_solutions(
        #             n_particles, split_data, m, levitator
        #         )

        #         # 依次取出solutions，先检查是否Gorkov更好，再检查是否满足距离约束
        #         # 分别求出两个segment的最大位移，用于缩放时间                
        #         re_plan_segment = np.transpose(split_data[:, m-1:m+2, 2:], (1, 0, 2))

        #         for i in range(candidate_solutions.shape[1]):
        #             # 检查solutions的Gorkov是否比原坐标更差
        #             if sorted_solutions_max_gorkov[i] > max_gorkov[m]:
        #                 print('Worse gorkov!')
                        
        #             re_plan_segment[1:2, :, :] = np.transpose(
        #                 candidate_solutions[:, sorted_indices[i]:sorted_indices[i]+1, :], 
        #                 (1, 0, 2)
        #             )

        #             for k in range(2):
        #                 segment = re_plan_segment[k:(k+2)]
        #                 interpolated_coords = interpolate_positions(segment)
                        
        #                 for j in range(interpolated_coords.shape[0]):
        #                     collision = safety_area(n_particles, interpolated_coords[j])
        #                     if np.any(collision != 0):
        #                         break
        #                 if np.any(collision != 0):
        #                     print("Collision!")
        #                     break

        #             if np.all(collision == 0):
        #                 print("Best candidate idx (start from 0):", i)
        #                 split_data[:, m, 2:] = np.copy(candidate_solutions[:, sorted_indices[i], :])
        #                 break

        # 计算时间序列，要求每个片段的最大速度不超过最大速度（0.1m/s）
        max_displacements = max_displacement_v2(split_data[:, :, 2:])
        diff_time = max_displacements / 0.1
        # 向上取整为 32.0/10000 的整数倍
        step = 32.0 / 10000
        rounded_diff_time = np.ceil(diff_time / step) * step
        # 计算累计时间并保存
        total_time = np.cumsum(rounded_diff_time)
        split_data[:, 1:, 1] = total_time


        end_time = time.time()  # 记录当前循环的结束时间
        elapsed_time = end_time - start_time  # 计算当前循环的运行时间
        computation_time.append(elapsed_time)

        # Calculate the Gorkov again
        gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
        max_gorkov = np.max(gorkov, axis=1)
        print('\nMax Gorkov after trajectory smoothing:', max_gorkov)

        # 保存修改后的轨迹
        file_path = os.path.join(global_model_dir_1, model_name, f'{file_name}_{str(n)}.csv')
        save_path_v2(file_path, n_particles, split_data)

    computation_time = np.array(computation_time)
    time_mean = np.mean(computation_time)
    time_std = np.std(computation_time)
    print(f'The mean of computation time: {time_mean}')
    print(f'The std of computation time: {time_std}')