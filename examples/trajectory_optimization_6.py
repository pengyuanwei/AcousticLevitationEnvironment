import csv
import numpy as np
import math
import torch
import time
import os

from utils import *
from calculate_gorkov_utils import *
from top_bottom_setup import top_bottom_setup
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist
import timeit


# Modified based on the trajectory_optimization_5.py


if __name__ == '__main__':
    n_particles = 10
    global_model_dir_1 = './experiments/experiment_85'
    model_name = '85_84'
    num_file = 50
    file_name = 'optimised_7_data'

    levitator = top_bottom_setup(n_particles)

    # computation_time = []
    for n in range(num_file):
        # start_time = time.time()  # 记录当前循环的开始时间

        print(f'-----------------------The paths {n}-----------------------')
        include_NaN = False

        csv_file = os.path.join(global_model_dir_1, model_name, f'path{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            continue

        max_length = np.zeros(n_particles)
        which_particle = 0

        csv_data_float = []
        for j in range(len(csv_data)):
            sub_data_list = []
            if csv_data[j] and len(csv_data[j]) == 5:
                # 检测是否为NaN值
                if any(value == '-nan(ind)' or math.isnan(float(value)) for value in csv_data[j]):
                    include_NaN = True
                    break
                if include_NaN == True:
                    break
                sub_data_list = [float(element) for element in csv_data[j]]
                csv_data_float.append(sub_data_list)
                if sub_data_list[0] >= max_length[which_particle]:
                    max_length[which_particle] = sub_data_list[0]
                else:
                    which_particle += 1

        if np.max(max_length) == 0.0 or include_NaN == True:
            continue

        max_length_int = max_length.astype(int)
        max_length_int += 1

        data_numpy = np.array(csv_data_float)


        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # When axis=2: particle_id, time, x, y, z
        split_data_numpy = np.zeros((n_particles, np.max(max_length_int), 5))

        for j in range(len(split_data_numpy)):
            split_data_numpy[j, :max_length_int[j]] = data_numpy[:max_length_int[j]]

            if max_length_int[j] < np.max(max_length_int):
                last_particle_position = data_numpy[max_length_int[j]-1]
                split_data_numpy[j, -(np.max(max_length_int)-max_length_int[j]):] = last_particle_position

            data_numpy = data_numpy[max_length_int[j]:]


        # print(split_data_numpy.shape)   
        # print(split_data_numpy[0])
        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        # 计算时间变化量（差分）
        delta_time = np.diff(split_data_numpy[:, :, 1], axis=1)

        # 保留初始时间点为0，补齐成与原数据相同的形状
        delta_time = np.concatenate([np.zeros((split_data_numpy.shape[0], 1)), delta_time], axis=1)

        # 将时间累加值替换为时间变化量
        split_data_numpy[:, :, 1] = delta_time 


        # Step 1: 使用随机搜索来优化轨迹运动学指标
        # Step 1.1: 优化速度变化
        for m in range(200):
            # 计算轨迹的角度变化以及每一段的距离
            _, distances_array = compute_angles_and_distances(split_data_numpy[:, :, 2:], degrees=True)
            # 计算移动距离变化
            delta_distance = np.abs(np.diff(distances_array, axis=1))
            #print(distances_array)

            delta_distance = np.transpose(delta_distance, (1, 0))   # (num_keypoints - 2, num_particles)
            max_delta_distance = np.max(delta_distance, axis=1)
            if m == 0:
                print('Max delta distance before random search: \n', max_delta_distance)
            # 找出角度变化最大的state
            max_delta_distance_index = np.argmax(max_delta_distance) + 1


            # 对角度变化最大的state进行随机搜索以平滑轨迹
            # Generate 100 potential solutions for the worst state, and then sort them based on the xy distance
            solutions = np.transpose(create_constrained_points_4(
                n_particles, 
                split_data_numpy[:, max_delta_distance_index, 2:], 
                split_data_numpy[:, max_delta_distance_index-1, 2:], 
                split_data_numpy[:, max_delta_distance_index+1, 2:],
                tan_coefficient = 5.671),
                (1, 0, 2))  # (N, 100, 3)
            

            # 计算solutions的delta angles
            solutions_delta_angles = compute_delta_angles(
                np.tile(split_data_numpy[:, max_delta_distance_index-1, 2:], (100, 1, 1)), 
                np.transpose(solutions, (1, 0, 2)), 
                np.tile(split_data_numpy[:, max_delta_distance_index+1, 2:], (100, 1, 1)), 
                degrees=True)
            
            solutions_min_delta_angles = np.amax(solutions_delta_angles, axis=1)
            solutions_min_delta_distances_index = np.argmin(solutions_min_delta_angles)
            #print(solutions_min_delta_angles[solutions_min_delta_distances_index])

            # 根据最大delta angles对solutions从小到大排序
            sorted_indices = np.argsort(solutions_min_delta_angles)
            sorted_solutions_min_d_angle = solutions_min_delta_angles[sorted_indices]
            # print(sorted_solutions_min_d_angle)


            # 依次取出solutions，检查是否满足距离约束
            # The segment: the front state, the worst state, and the back state
            re_plan_segment = np.copy(split_data_numpy[:, max_delta_distance_index-1:max_delta_distance_index+2, 2:])
            re_plan_segment = np.transpose(re_plan_segment, (1, 0, 2))  # (3, N, 3)
            for i in range(solutions.shape[1]):
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
                    split_data_numpy[:, max_delta_distance_index, 2:] = np.copy(solutions[:, sorted_indices[i], :])
                    break


        # 计算轨迹的角度变化以及每一段的距离
        _, distances_array = compute_angles_and_distances(split_data_numpy[:, :, 2:], degrees=True)
        # 计算移动距离变化
        delta_distance = np.abs(np.diff(distances_array, axis=1))
        delta_distance = np.transpose(delta_distance, (1, 0))   # (num_keypoints - 2, num_particles)
        max_delta_distance = np.max(delta_distance, axis=1)

        print('Max delta distance after random search: \n', max_delta_distance)
        print('The mean of max delta distance after random search: \n', np.mean(max_delta_distance))





        # # Step 1.2: 优化方向变化
        # for m in range(1):
        #     # 计算轨迹的角度变化以及每一段的距离
        #     angles_array, distances_array = compute_angles_and_distances(split_data_numpy[:, :, 2:], degrees=True)

        #     print(angles_array)

        #     angles_array = np.transpose(angles_array, (1, 0))   # (num_keypoints - 2, num_particles)
        #     max_angles = np.max(angles_array, axis=1)
        #     # 找出角度变化最大的state
        #     max_angle_index = np.argmax(max_angles) + 1


        #     # 对角度变化最大的state进行随机搜索以平滑轨迹
        #     # Generate 100 potential solutions for the worst state, and then sort them based on the xy distance
        #     solutions = np.transpose(create_constrained_points_4(
        #         n_particles, 
        #         split_data_numpy[:, max_angle_index, 2:], 
        #         split_data_numpy[:, max_angle_index-1, 2:], 
        #         split_data_numpy[:, max_angle_index+1, 2:]), 
        #         (1, 0, 2))
        #     # print(solutions.shape)    # (N, 100, 3)
            





        #     # 计算solutions的xy distance
        #     points = np.transpose(solutions[:, :, 0:2], (1, 0, 2))  # 提取所有解的 XY 坐标，形状为 (num_solutions, N, 2)
        #     diff = points[:, :, np.newaxis, :] - points[:, np.newaxis, :, :]  # 计算差值，形状为 (num_solutions, N, N, 2)
        #     distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # 计算距离，形状为 (num_solutions, N, N)
        #     # 将对角线元素设置为无穷大
        #     num_solutions, N, _ = points.shape
        #     for i in range(num_solutions):
        #         np.fill_diagonal(distances[i], np.inf)
        #     # 找出每个solution的最小xy distance
        #     solutions_min_xy_dists = np.min(distances, axis=(1, 2))

        #     # 根据最小xy distance对solutions从大到小排序
        #     sorted_indices = np.argsort(-solutions_min_xy_dists)
        #     sorted_solutions_min_dists = solutions_min_xy_dists[sorted_indices]
        #     # print(sorted_solutions_min_dists)


        #     # 依次取出solutions，先检查是否比原state更好，再检查是否满足距离约束
        #     # The segment: the front state, the worst state, and the back state
        #     re_plan_segment = np.copy(split_data_numpy[:, min_dist_index-1:min_dist_index+2, 2:])
        #     re_plan_segment = np.transpose(re_plan_segment, (1, 0, 2))  # (3, N, 3)
            
        #     # 分别求出两个segment的最大位移，用于缩放时间
        #     original_max_displacement = max_displacement(re_plan_segment)
        #     for i in range(solutions.shape[1]):
        #         # 如果solutions的distance比原坐标更差，则break
        #         if sorted_solutions_min_dists[i] <= min_xy_dists[min_dist_index]:
        #             print('Break!')
        #             break
        #         re_plan_segment[1:2, :, :] = np.transpose(solutions[:, sorted_indices[i]:sorted_indices[i]+1, :], (1, 0, 2))

        #         for k in range(2):
        #             segment = re_plan_segment[k:(k+2)]
        #             interpolated_coords = interpolate_positions(segment)
                    
        #             for j in range(interpolated_coords.shape[0]):
        #                 collision = safety_area(n_particles, interpolated_coords[j])
        #                 if not np.all(collision == 0):
        #                     break
        #             if not np.all(collision == 0):
        #                 break

        #         if np.all(collision == 0):
        #             print(i)
        #             split_data_numpy[:, min_dist_index, 2:] = np.copy(solutions[:, sorted_indices[i], :])

        #             # 修改时间间隔
        #             new_max_displacement = max_displacement(re_plan_segment)
        #             time_zoom = new_max_displacement / original_max_displacement
        #             split_data_numpy[:, min_dist_index, 1] *= time_zoom[0]
        #             split_data_numpy[:, min_dist_index+1, 1] *= time_zoom[1]
        #             break

        # for j in range(np.max(max_length_int)):
        #     distances = cdist(split_data_numpy[:, j, 2:4], split_data_numpy[:, j, 2:4])
        #     # 将距离矩阵的对角线（即粒子自身与自身的距离）设置为无穷大，防止在后续检查中误判。
        #     np.fill_diagonal(distances, np.inf)     
        #     min_xy_dists[j][0] = np.min(distances)

        # print('Min distance after random search: \n', min_xy_dists)


    #     # Step 2: 使用随机搜索来优化最弱Gorkov的timesteps
    #     for m in range(1):
    #         print(f'-----------------------The iteration {m}-----------------------')
    #         # Calculate the Gorkov
    #         gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])

    #         max_gorkov = np.max(gorkov, axis=1)
    #         # print初始Gorkov
    #         if m == 0:
    #             print('Max Gorkov before random search:', max_gorkov)
    #         max_gorkov[0] = -1.0
    #         max_gorkov[-1] = -1.0
    #         max_gorkov_index = np.argmax(max_gorkov)
    #         print('The worst index:', max_gorkov_index)
    #         print('The worst value:', max_gorkov[max_gorkov_index])

    #         # The segment: the front state, the worst state, and the back state
    #         re_plan_segment = np.copy(split_data_numpy[:, max_gorkov_index-1:max_gorkov_index+2, 2:])
    #         re_plan_segment = np.transpose(re_plan_segment, (1, 0, 2))
            
    #         # Generate 100 potential solutions for the worst state, and then sort them based on their Gor'kov values.
    #         solutions = np.transpose(create_constrained_points_3(
    #             n_particles, 
    #             split_data_numpy[:, max_gorkov_index, 2:], 
    #             split_data_numpy[:, max_gorkov_index-1, 2:], 
    #             split_data_numpy[:, max_gorkov_index+1, 2:]), 
    #             (1, 0, 2))
    #         print(solutions.shape)

    #         # 计算solutions的Gorkov
    #         solutions_gorkov = levitator.calculate_gorkov(solutions)
    #         #print(solutions_gorkov.shape)

    #         # 找出每个solutions的最大Gorkov
    #         solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
    #         #print(solutions_max_gorkov.shape)

    #         # 根据Gorkov对solutions从小到大排序
    #         sorted_indices = np.argsort(solutions_max_gorkov)
    #         sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]


    #         # 依次取出solutions，先检查是否Gorkov更好，再检查是否满足距离约束
    #         # 分别求出两个segment的最大位移，用于缩放时间
    #         original_max_displacement = max_displacement(re_plan_segment)
    #         for i in range(solutions.shape[1]):
    #             # 如果solutions的Gorkov比原坐标更差，则break
    #             if sorted_solutions_max_gorkov[i] > max_gorkov[max_gorkov_index]:
    #                 print('Break!')
    #                 break
    #             re_plan_segment[1:2, :, :] = np.transpose(solutions[:, sorted_indices[i]:sorted_indices[i]+1, :], (1, 0, 2))

    #             for k in range(2):
    #                 segment = re_plan_segment[k:(k+2)]
    #                 interpolated_coords = interpolate_positions(segment)
                    
    #                 for j in range(interpolated_coords.shape[0]):
    #                     collision = safety_area(n_particles, interpolated_coords[j])
    #                     if not np.all(collision == 0):
    #                         break
    #                 if not np.all(collision == 0):
    #                     break

    #             if np.all(collision == 0):
    #                 print(i)
    #                 split_data_numpy[:, max_gorkov_index, 2:] = np.copy(solutions[:, sorted_indices[i], :])

    #                 # 修改时间间隔
    #                 new_max_displacement = max_displacement(re_plan_segment)
    #                 time_zoom = new_max_displacement / original_max_displacement
    #                 split_data_numpy[:, max_gorkov_index, 1] *= time_zoom[0]
    #                 split_data_numpy[:, max_gorkov_index+1, 1] *= time_zoom[1]
    #                 break


    #     # Calculate the Gorkov again
    #     gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])
    #     max_gorkov = np.max(gorkov, axis=1)
    #     print('Max Gorkov after random search:', max_gorkov)


    #     # 检查所有关键点是否在圆圈内，如果不在，对其进行优化
    #     for m in range(1, split_data_numpy.shape[1] - 1):
    #         if not positions_check(n_particles, split_data_numpy[:, m, 2:], split_data_numpy[:, m-1, 2:], split_data_numpy[:, m+1, 2:]):
    #             print(f'-----------------------The point {m}-----------------------')
    #             # Calculate the Gorkov
    #             gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])
    #             max_gorkov = np.max(gorkov, axis=1)

    #             re_plan_segment = np.copy(split_data_numpy[:, m-1:m+2, 2:])
    #             re_plan_segment = np.transpose(re_plan_segment, (1, 0, 2))
                
    #             # 对最弱key points生成100个潜在solutions，并排序
    #             solutions = np.transpose(create_constrained_points_1(
    #                 n_particles, 
    #                 split_data_numpy[:, m, 2:], 
    #                 split_data_numpy[:, m-1, 2:], 
    #                 split_data_numpy[:, m+1, 2:]), 
    #                 (1, 0, 2))
    #             #print(solutions.shape)

    #             # 计算solutions的Gorkov
    #             solutions_gorkov = levitator.calculate_gorkov(solutions)
    #             #print(solutions_gorkov.shape)

    #             # 找出每个solutions的最大Gorkov
    #             solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
    #             #print(solutions_max_gorkov.shape)

    #             # 根据Gorkov对solutions从小到大排序
    #             sorted_indices = np.argsort(solutions_max_gorkov)
    #             sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]


    #             # 依次取出solutions，先检查是否Gorkov更好，再检查是否满足距离约束
    #             # 分别求出两个segment的最大位移，用于缩放时间                
    #             original_max_displacement = max_displacement(re_plan_segment)
    #             for i in range(solutions.shape[1]):
    #                 # 检查solutions的Gorkov是否比原坐标更差
    #                 if sorted_solutions_max_gorkov[i] > max_gorkov[m]:
    #                     print('Worse gorkov!')
    #                 re_plan_segment[1:2, :, :] = np.transpose(solutions[:, sorted_indices[i]:sorted_indices[i]+1, :], (1, 0, 2))

    #                 for k in range(2):
    #                     segment = re_plan_segment[k:(k+2)]
    #                     interpolated_coords = interpolate_positions(segment)
                        
    #                     for j in range(interpolated_coords.shape[0]):
    #                         collision = safety_area(n_particles, interpolated_coords[j])
    #                         if not np.all(collision == 0):
    #                             break
    #                     if not np.all(collision == 0):
    #                         break

    #                 if np.all(collision == 0):
    #                     print(i)
    #                     split_data_numpy[:, m, 2:] = np.copy(solutions[:, sorted_indices[i], :])

    #                     # 修改时间间隔
    #                     new_max_displacement = max_displacement(re_plan_segment)
    #                     time_zoom = new_max_displacement / original_max_displacement
    #                     split_data_numpy[:, m, 1] *= time_zoom[0]
    #                     split_data_numpy[:, m+1, 1] *= time_zoom[1]
    #                     break

    #     # 将时间向上取整，变成0.0032的整数倍
    #     split_data_numpy[:, 1:, 1] = np.ceil(split_data_numpy[:, 1:, 1] / 0.0032) * 0.0032


    # #     end_time = time.time()  # 记录当前循环的结束时间
    # #     elapsed_time = end_time - start_time  # 计算当前循环的运行时间
    # #     computation_time.append(elapsed_time)

    # # computation_time = np.array(computation_time)
    # # time_mean = np.mean(computation_time)
    # # time_std = np.std(computation_time)
    # # print(f'The mean of computation time: {time_mean}')
    # # print(f'The std of computation time: {time_std}')


    #     # Calculate the Gorkov again
    #     gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])
    #     max_gorkov = np.max(gorkov, axis=1)
    #     print('Max Gorkov after trajectory smoothing:', max_gorkov)


        # 保存修改后的轨迹
        save_path = os.path.join(global_model_dir_1, model_name, f'{file_name}_{str(n)}.csv')
        file_instance = open(save_path, "w", encoding="UTF8", newline='')
        csv_writer = csv.writer(file_instance)

        for i in range(n_particles):
            header = ['Agent ID', i]
            row_1 = ['Number of', split_data_numpy.shape[1]]

            csv_writer.writerow(header)
            csv_writer.writerow(row_1)

            rows = []
            path_time = 0.0
            for j in range(split_data_numpy.shape[1]):
                path_time += split_data_numpy[i][j][1]
                rows = [j, path_time, split_data_numpy[i][j][2], split_data_numpy[i][j][3], split_data_numpy[i][j][4]]
                csv_writer.writerow(rows)

        file_instance.close()  