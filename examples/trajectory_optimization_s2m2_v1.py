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
不太行，S2M2的轨迹生成的condidate solutions总是在轨迹的其他地方产生冲突
'''

if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v2/S2M2'
    num_file = 100
    levitator = top_bottom_setup(n_particles, algorithm='Naive', iterations=1)

    computation_time = []
    for n in range(1):
        print(f'\n-----------------------The paths {n}-----------------------')

        ### 读取并预处理轨迹
        # csv_data是list，其中的元素是list，每个子list保存了每一行的数据
        csv_file = os.path.join(global_model_dir_1, model_name, f'path_S2M2_{str(n)}.csv')
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
        # visualize_paths(n_particles, split_data[:, :, 1:])
        
        ### 使用随机搜索来优化原keypoints中的最弱Gorkov点
        for m in range(1):
            # Calculate Gorkov
            gorkov = levitator.calculate_gorkov(split_data[:, :, 1:])
            # 找出除端点外的所有keypoints中的最大Gorkov及其沿时间轴的索引
            max_gorkov_idx, max_gorkov = calculate_flags_max_gorkov(gorkov, keypoint_flags)
            
            # 对所有非直线路径进行修正
            if np.any(max_gorkov != -np.inf):
                # Print initial Gorkov values
                if m == 0:
                    print('Max Gorkov before random search:', max_gorkov)
                print(f'\n-----------------------The iteration {m}-----------------------')
                print("Worst gorkov idx:", max_gorkov_idx)
                print("Worst gorkov value:", max_gorkov[max_gorkov_idx])
                
                print(keypoint_flags[max_gorkov_idx, :, np.newaxis].shape)
                print(keypoint_flags[max_gorkov_idx, :, np.newaxis])
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
                print(split_data[:, max_gorkov_idx, 1:])
                print(candidate_solutions[sorted_indices[0]])

                ### 依次取出 candidate_solutions，先检查是否Gorkov更好，再检查是否满足距离约束
                # re_plan_segment: (3, num_particles, xyz)
                re_plan_segment = np.transpose(split_data[:, max_gorkov_idx-1:max_gorkov_idx+2, 1:], (1, 0, 2))

                for i in range(candidate_solutions.shape[0]):
                    # 如果 candidate_solutions 的 Gorkov 比原坐标的更差，则 break
                    if sorted_solutions_max_gorkov[i] > max_gorkov[max_gorkov_idx]:
                        print('No better candidate than original!')
                        break

                    re_plan_segment[1:2, :, :] = candidate_solutions[sorted_indices[i]:sorted_indices[i]+1, :, :]

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
                        print("New position's Gorkov:", sorted_solutions_max_gorkov[i])
                        split_data[:, max_gorkov_idx, 1:] = np.copy(candidate_solutions[sorted_indices[i], :, :])
                        break

            else:
                print('All particles go to their targets directly!')

        # visualize_paths(n_particles, split_data[:, :, 1:])
