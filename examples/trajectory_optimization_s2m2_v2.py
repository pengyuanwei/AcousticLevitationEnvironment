import os
import time
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils_v2 import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils_v2 import *
from examples.utils.path_smoothing_3 import *

'''
Do random search to find better Gorkov points for finded paths(keypoints)
优化S2M2: 先插值成同步轨迹, 然后修改原keypoints
转化为同步轨迹：
    - 同时间结束
    - 中间插值
使用candiate solutions更新轨迹
'''

if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v2/S2M2'
    original_paths_name = 'path_S2M2'
    new_paths_name = 'S2M2_optimized_path'

    levitator = top_bottom_setup(n_particles, algorithm='Naive', iterations=1)

    num_file = 100
    for n in range(num_file):
        print(f'\n-----------------------The paths {n}-----------------------')

        ### 读取并预处理轨迹
        # csv_data是list，其中的元素是list，每个子list保存了每一行的数据
        csv_file = os.path.join(global_model_dir_1, model_name, f'{original_paths_name}_{str(n)}.csv')
        csv_data = read_csv_file(csv_file)
        if csv_data is None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue
        data_numpy, include_NaN = read_paths(csv_data)
        if include_NaN:
            print(f"Skipping file due to NaN values: {csv_file}")
            continue
        # split_data: np.array, (num_particles, path_length, 4), 时刻, 坐标
        # keypoint_flags: 标记原paths中除端点外的所有keypoints为1
        split_data, keypoint_flags = process_data(data_numpy)
        print('Keypoints:', keypoint_flags)
        #visualize_paths(n_particles, split_data[:, :, 1:])
        
        ### 使用随机搜索来优化原keypoints中的最弱Gorkov点
        for m in range(10):
            # Calculate Gorkov
            gorkov = levitator.calculate_gorkov(split_data[:, :, 1:])
            # 找出除端点外的所有keypoints中的最大Gorkov及其沿时间轴的索引
            max_gorkov_idx, max_gorkov = calculate_flags_max_gorkov(gorkov, keypoint_flags)
            
            # 对所有非直线路径进行修正
            if np.any(max_gorkov != -np.inf):
                # Print initial Gorkov values
                print(f'Max Gorkov before {m}th random search:', max_gorkov)
                print(f'\n-----------------------The iteration {m}-----------------------')
                print("Worst gorkov idx:", max_gorkov_idx)
                print("Worst gorkov value:", max_gorkov[max_gorkov_idx])
                
                # print(keypoint_flags[max_gorkov_idx, :, np.newaxis].shape)
                # print(keypoint_flags[max_gorkov_idx, :, np.newaxis])
                candidate_solutions, sorted_indices, sorted_solutions_max_gorkov = generate_solutions_segments_v2(
                    n_particles, 
                    split_data[:, max_gorkov_idx-1, 1:], 
                    split_data[:, max_gorkov_idx, 1:], 
                    split_data[:, max_gorkov_idx+1, 1:], 
                    levitator,
                    keypoint_flags[max_gorkov_idx, :, np.newaxis], 
                    num_solutions=50,
                    search_factor=5.0
                )

                # print最大Gorkov最大的时刻的原坐标点最优candidate solution
                # print(split_data[:, max_gorkov_idx, 1:])
                # print(candidate_solutions[sorted_indices[0]])

                ### 依次取出 candidate_solutions，先检查是否Gorkov更好，再检查是否满足距离约束
                for i in range(candidate_solutions.shape[0]):                
                    # 如果 candidate_solutions 的 Gorkov 比原坐标的更差，则 break
                    if sorted_solutions_max_gorkov[i] > max_gorkov[max_gorkov_idx]:
                        print('No better candidate than original!')
                        break

                    # 使用candiate solutions更新轨迹
                    re_plan_paths = split_data[:, :, 1:].copy()
                    re_plan_paths[:, max_gorkov_idx, :] = candidate_solutions[sorted_indices[i]]
                    # print(re_plan_paths.shape)
                    # print(keypoint_flags)

                    ## 修改被修改的粒子的轨迹，以减少不必要的direction change
                    # 找出标记为1的粒子的索引
                    # print('The flags of changed time:', keypoint_flags[max_gorkov_idx])
                    agent_idx = np.where(keypoint_flags[max_gorkov_idx] == 1)
                    # print('The index of changed particle:', agent_idx[0][0])
                    # 修改这个粒子的轨迹：消除被修正的keypoint与原轨迹前后keypoints之间的direction change
                    # print('The path flags of this particle:', keypoint_flags[:, agent_idx[0][0]])
                    nearest_index = np.zeros(2, dtype=int)
                    nearest_index[0], nearest_index[1] = find_nearest_nonzero(keypoint_flags[:, agent_idx[0][0]], max_gorkov_idx)
                    # print('The nearest direction changes:', nearest_index[0], nearest_index[1])
                    # print('The time list:', split_data[0, :, 0])
                    # print('\n')
                    # 分开处理前一个segment和后一个segment
                    for j in range(2):
                        if nearest_index[j] == max_gorkov_idx+1:
                            # print(f'Segment {j} does not need to modify!')
                            continue
                        time_diff = split_data[0, min(nearest_index[j], max_gorkov_idx):max(nearest_index[j], max_gorkov_idx)+1, 0] - min(split_data[0, nearest_index[j], 0], split_data[0, max_gorkov_idx, 0])
                        # print(f'The time list between the segment {j}:', time_diff)
                        modified_path = lerp_3d_series(re_plan_paths[agent_idx[0][0], min(nearest_index[j], max_gorkov_idx), :], re_plan_paths[agent_idx[0][0], max(nearest_index[j], max_gorkov_idx), :], time_diff)
                        # print('First point:', re_plan_paths[agent_idx[0][0], min(nearest_index[j], max_gorkov_idx), :])
                        # print('Second point:', re_plan_paths[agent_idx[0][0], max(nearest_index[j], max_gorkov_idx), :])
                        # print(f'Whole modified segment {j}:', modified_path)
                        # print('\n')
                        re_plan_paths[agent_idx[0][0], min(nearest_index[j], max_gorkov_idx):max(nearest_index[j], max_gorkov_idx)+1, :] = modified_path

                    # re_plan_paths: (num_particles, paths_length, 3)
                    interpolated_coords = linear_interpolation(re_plan_paths, k=9)
                    # print(re_plan_paths.shape)
                    # print(interpolated_coords.shape)
                    # print(re_plan_paths[0, 0], re_plan_paths[0, -1])
                    # print(interpolated_coords[0, 0], interpolated_coords[0, -1])
                    
                    for j in range(interpolated_coords.shape[1]):
                        collision = safety_area(n_particles, interpolated_coords[:, j])
                        if np.any(collision != 0):
                            print(f"Collision checking: the {i}th candidate solution collision!")
                            break

                    if np.all(collision == 0):
                        print("Best candidate idx (start from 0):", i)
                        print("New position's Gorkov:", sorted_solutions_max_gorkov[i])
                        split_data[:, :, 1:] = re_plan_paths
                        break

            else:
                print('All particles go to their targets directly!')

        # Calculate Gorkov
        gorkov = levitator.calculate_gorkov(split_data[:, :, 1:])
        _, max_gorkov = calculate_flags_max_gorkov(gorkov, keypoint_flags)
        if np.any(max_gorkov != -np.inf):
            print(f'Final max Gorkovs after random searchs:', max_gorkov)

        # 保存修改后的轨迹
        file_path = os.path.join(global_model_dir_1, model_name, f'{new_paths_name}_{str(n)}.csv')
        save_path_v2(file_path, n_particles, split_data)