import os
import time
import numpy as np

from top_bottom_setup import top_bottom_setup
from examples.general_utils import *
from examples.acoustic_utils import *
from examples.optimizer_utils import *


# Modified based on the trajectory_optimization_3.py: modularity for better readability
# Do random search to find better Gorkov points for finded paths(keypoints)
# Transform the time series into integer multiples of 32/1000s


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99'
    num_file = 2
    file_name = 'optimised_data'

    levitator = top_bottom_setup(n_particles)

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
        split_data = process_paths(data_numpy, paths_length)


        # 使用随机搜索来优化最弱Gorkov的timesteps
        for m in range(10):
            # Calculate Gorkov
            gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
            max_gorkov_idx, max_gorkov = calculate_max_gorkov(gorkov)

            # Print initial Gorkov values
            if m == 0:
                print('Max Gorkov before random search:', max_gorkov)
            print(f'\n-----------------------The iteration {m}-----------------------')
            print("Worst gorkov idx:", max_gorkov_idx)
            print("Worst gorkov value:", max_gorkov[max_gorkov_idx])
            
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


            # 依次取出 candidate_solutions，先检查是否Gorkov更好，再检查是否满足距离约束
            # 分别求出前后两个 segment 的最大位移，用于缩放时间
            re_plan_segment = np.transpose(
                split_data[:, max_gorkov_idx-1:max_gorkov_idx+2, 2:], 
                (1, 0, 2)
            )
            original_max_displacement = max_displacement(re_plan_segment)
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
                        break

                if np.all(collision == 0):
                    print("Best candidate idx (start from 0):", i)
                    split_data[:, max_gorkov_idx, 2:] = np.copy(candidate_solutions[:, sorted_indices[i], :])

                    # 修改时间间隔
                    new_max_displacement = max_displacement(re_plan_segment)
                    time_zoom = new_max_displacement / original_max_displacement
                    split_data[:, max_gorkov_idx, 1] *= time_zoom[0]
                    split_data[:, max_gorkov_idx+1, 1] *= time_zoom[1]
                    break


        # Calculate the Gorkov again
        gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
        max_gorkov = np.max(gorkov, axis=1)
        print('Max Gorkov after random search:', max_gorkov)


        # 检查所有关键点是否在圆圈内，如果不在，对其进行优化
        for m in range(1, split_data.shape[1] - 1):
            if not positions_check(n_particles, split_data[:, m, 2:], split_data[:, m-1, 2:], split_data[:, m+1, 2:]):
                print(f'-----------------------The point {m}-----------------------')
                # Calculate the Gorkov
                gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
                max_gorkov = np.max(gorkov, axis=1)

                re_plan_segment = np.copy(split_data[:, m-1:m+2, 2:])
                re_plan_segment = np.transpose(re_plan_segment, (1, 0, 2))
                
                # 对最弱key points生成100个潜在solutions，并排序
                solutions = np.transpose(create_constrained_points_1(
                    n_particles, 
                    split_data[:, m, 2:], 
                    split_data[:, m-1, 2:], 
                    split_data[:, m+1, 2:]), 
                    (1, 0, 2))
                #print(solutions.shape)

                # 计算solutions的Gorkov
                solutions_gorkov = levitator.calculate_gorkov(solutions)
                #print(solutions_gorkov.shape)

                # 找出每个solutions的最大Gorkov
                solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
                #print(solutions_max_gorkov.shape)

                # 根据Gorkov对solutions从小到大排序
                sorted_indices = np.argsort(solutions_max_gorkov)
                sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]


                # 依次取出solutions，先检查是否Gorkov更好，再检查是否满足距离约束
                # 分别求出两个segment的最大位移，用于缩放时间                
                original_max_displacement = max_displacement(re_plan_segment)
                for i in range(solutions.shape[1]):
                    # 检查solutions的Gorkov是否比原坐标更差
                    if sorted_solutions_max_gorkov[i] > max_gorkov[m]:
                        print('Worse gorkov!')
                    re_plan_segment[1:2, :, :] = np.transpose(solutions[:, sorted_indices[i]:sorted_indices[i]+1, :], (1, 0, 2))

                    for k in range(2):
                        segment = re_plan_segment[k:(k+2)]
                        interpolated_coords = interpolate_positions(segment)
                        
                        for j in range(interpolated_coords.shape[0]):
                            collision = safety_area(n_particles, interpolated_coords[j])
                            if not np.all(collision == 0):
                                break
                        if not np.all(collision == 0):
                            break

                    if np.all(collision == 0):
                        print(i)
                        split_data[:, m, 2:] = np.copy(solutions[:, sorted_indices[i], :])

                        # 修改时间间隔
                        new_max_displacement = max_displacement(re_plan_segment)
                        time_zoom = new_max_displacement / original_max_displacement
                        split_data[:, m, 1] *= time_zoom[0]
                        split_data[:, m+1, 1] *= time_zoom[1]
                        break

        # # 将时间向上取整，变成0.0032的整数倍
        # split_data[:, 1:, 1] = np.ceil(split_data[:, 1:, 1] / 0.0032) * 0.0032


        end_time = time.time()  # 记录当前循环的结束时间
        elapsed_time = end_time - start_time  # 计算当前循环的运行时间
        computation_time.append(elapsed_time)

    computation_time = np.array(computation_time)
    time_mean = np.mean(computation_time)
    time_std = np.std(computation_time)
    print(f'The mean of computation time: {time_mean}')
    print(f'The std of computation time: {time_std}')


        # # Calculate the Gorkov again
        # gorkov = levitator.calculate_gorkov(split_data[:, :, 2:])
        # max_gorkov = np.max(gorkov, axis=1)
        # print('Max Gorkov after trajectory smoothing:', max_gorkov)


        # # 保存修改后的轨迹
        # save_path = os.path.join(global_model_dir_1, model_name, f'{file_name}_{str(n)}.csv')
        # file_instance = open(save_path, "w", encoding="UTF8", newline='')
        # csv_writer = csv.writer(file_instance)

        # for i in range(n_particles):
        #     header = ['Agent ID', i]
        #     row_1 = ['Number of', split_data.shape[1]]

        #     csv_writer.writerow(header)
        #     csv_writer.writerow(row_1)

        #     rows = []
        #     path_time = 0.0
        #     for j in range(split_data.shape[1]):
        #         path_time += split_data[i][j][1]
        #         rows = [j, path_time, split_data[i][j][2], split_data[i][j][3], split_data[i][j][4]]
        #         csv_writer.writerow(rows)

        # file_instance.close()  