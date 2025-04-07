import os
import time
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils_v2 import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils_v2 import *
from examples.utils.path_smoothing_3 import *


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v11'
    num_file = 100
    levitator = top_bottom_setup(n_particles, algorithm='Naive', iterations=1)

    computation_time = []
    for n in range(10):
        print(f'\n-----------------------The paths {n}-----------------------')

        ### 读取并预处理轨迹
        # csv_data是list，其中的元素是list，每个子list保存了每一行的数据
        csv_file = os.path.join(global_model_dir_1, model_name, f'path_{str(n)}.csv')
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
        # split_data的形状为(n_particles, n_keypoints, 5)
        # Axis 2: keypoints_idx, 时间累加值（时间列）, x, y, z
        split_data = data_numpy.reshape(-1, paths_length, 5)

        visualize_paths(n_particles, split_data[:, :, 2:])

