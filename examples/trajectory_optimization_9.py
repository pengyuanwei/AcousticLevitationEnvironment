import os
import time
import numpy as np

from utils import *
from calculate_gorkov_utils import *
from top_bottom_setup import top_bottom_setup


# Modified based on the trajectory_optimization_3.py: modularity for better readability
# Do random search to find better Gorkov points for finded paths(keypoints)
# Transform the time series into integer multiples of 32/1000s


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


if __name__ == '__main__':
    n_particles = 6
    global_model_dir_1 = './experiments/experiment_89'
    model_name = '89_88'
    num_file = 50
    file_name = 'optimised_data'

    levitator = top_bottom_setup(n_particles)

    computation_time = []
    for n in range(num_file):
        print(f'-----------------------The paths {n}-----------------------')
        # 记录当前循环的开始时间
        start_time = time.time()  

        # csv_data是list，其中的元素是list，每个子list保存了每一行的数据
        csv_file = os.path.join(global_model_dir_1, model_name, f'path{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            continue

        data_numpy, include_NaN = read_paths(csv_data)
        if include_NaN == True:
            continue

        split_data_numpy = process_paths(csv_data, n_particles)


        # print(split_data_numpy.shape)   
        # print(split_data_numpy[0])
        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        # 计算时间变化量（差分）
        delta_time = np.diff(split_data_numpy[:, :, 1], axis=1)

        # 保留初始时间点为0，补齐成与原数据相同的形状
        delta_time = np.concatenate([np.zeros((split_data_numpy.shape[0], 1)), delta_time], axis=1)

        # 将时间累加值替换为时间变化量
        split_data_numpy[:, :, 1] = delta_time 


        # 使用随机搜索来优化最弱Gorkov的timesteps
        for m in range(10):
            print(f'-----------------------The iteration {m}-----------------------')
            # Calculate the Gorkov
            gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])

            max_gorkov = np.max(gorkov, axis=1)
            # print初始Gorkov
            if m == 0:
                print('Max Gorkov before random search:', max_gorkov)
            max_gorkov[0] = -1.0
            max_gorkov[-1] = -1.0
            max_gorkov_index = np.argmax(max_gorkov)
            print(max_gorkov_index)
            print(max_gorkov[max_gorkov_index])

            re_plan_segment = np.copy(split_data_numpy[:, max_gorkov_index-1:max_gorkov_index+2, 2:])
            re_plan_segment = np.transpose(re_plan_segment, (1, 0, 2))
            
            # 对最弱key points生成100个潜在solutions，并排序
            solutions = np.transpose(create_constrained_points_1(
                n_particles, 
                split_data_numpy[:, max_gorkov_index, 2:], 
                split_data_numpy[:, max_gorkov_index-1, 2:], 
                split_data_numpy[:, max_gorkov_index+1, 2:]), 
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
                # 如果solutions的Gorkov比原坐标更差，则break
                if sorted_solutions_max_gorkov[i] > max_gorkov[max_gorkov_index]:
                    print('Break!')
                    break
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
                    split_data_numpy[:, max_gorkov_index, 2:] = np.copy(solutions[:, sorted_indices[i], :])

                    # 修改时间间隔
                    new_max_displacement = max_displacement(re_plan_segment)
                    time_zoom = new_max_displacement / original_max_displacement
                    split_data_numpy[:, max_gorkov_index, 1] *= time_zoom[0]
                    split_data_numpy[:, max_gorkov_index+1, 1] *= time_zoom[1]
                    break


        # Calculate the Gorkov again
        gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])
        max_gorkov = np.max(gorkov, axis=1)
        print('Max Gorkov after random search:', max_gorkov)


        # 检查所有关键点是否在圆圈内，如果不在，对其进行优化
        for m in range(1, split_data_numpy.shape[1] - 1):
            if not positions_check(n_particles, split_data_numpy[:, m, 2:], split_data_numpy[:, m-1, 2:], split_data_numpy[:, m+1, 2:]):
                print(f'-----------------------The point {m}-----------------------')
                # Calculate the Gorkov
                gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])
                max_gorkov = np.max(gorkov, axis=1)

                re_plan_segment = np.copy(split_data_numpy[:, m-1:m+2, 2:])
                re_plan_segment = np.transpose(re_plan_segment, (1, 0, 2))
                
                # 对最弱key points生成100个潜在solutions，并排序
                solutions = np.transpose(create_constrained_points_1(
                    n_particles, 
                    split_data_numpy[:, m, 2:], 
                    split_data_numpy[:, m-1, 2:], 
                    split_data_numpy[:, m+1, 2:]), 
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
                        split_data_numpy[:, m, 2:] = np.copy(solutions[:, sorted_indices[i], :])

                        # 修改时间间隔
                        new_max_displacement = max_displacement(re_plan_segment)
                        time_zoom = new_max_displacement / original_max_displacement
                        split_data_numpy[:, m, 1] *= time_zoom[0]
                        split_data_numpy[:, m+1, 1] *= time_zoom[1]
                        break

        # 将时间向上取整，变成0.0032的整数倍
        split_data_numpy[:, 1:, 1] = np.ceil(split_data_numpy[:, 1:, 1] / 0.0032) * 0.0032


        end_time = time.time()  # 记录当前循环的结束时间
        elapsed_time = end_time - start_time  # 计算当前循环的运行时间
        computation_time.append(elapsed_time)

    computation_time = np.array(computation_time)
    time_mean = np.mean(computation_time)
    time_std = np.std(computation_time)
    print(f'The mean of computation time: {time_mean}')
    print(f'The std of computation time: {time_std}')


        # # Calculate the Gorkov again
        # gorkov = levitator.calculate_gorkov(split_data_numpy[:, :, 2:])
        # max_gorkov = np.max(gorkov, axis=1)
        # print('Max Gorkov after trajectory smoothing:', max_gorkov)


        # # 保存修改后的轨迹
        # save_path = os.path.join(global_model_dir_1, model_name, f'{file_name}_{str(n)}.csv')
        # file_instance = open(save_path, "w", encoding="UTF8", newline='')
        # csv_writer = csv.writer(file_instance)

        # for i in range(n_particles):
        #     header = ['Agent ID', i]
        #     row_1 = ['Number of', split_data_numpy.shape[1]]

        #     csv_writer.writerow(header)
        #     csv_writer.writerow(row_1)

        #     rows = []
        #     path_time = 0.0
        #     for j in range(split_data_numpy.shape[1]):
        #         path_time += split_data_numpy[i][j][1]
        #         rows = [j, path_time, split_data_numpy[i][j][2], split_data_numpy[i][j][3], split_data_numpy[i][j][4]]
        #         csv_writer.writerow(rows)

        # file_instance.close()  