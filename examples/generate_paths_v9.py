import os
import math
import gymnasium as gym
from acoustorl import MADDPG
from scipy.interpolate import interp1d

from examples.utils.general_utils import *
from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.path_smoothing_3 import *

# Change from generate_path_8.py and trajectory_visulization.py

if __name__ == "__main__":
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v11'
    num_file = 100
    levitator = top_bottom_setup(n_particles, algorithm='Naive', iterations=1)

    for n in range(1):
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
        # split_data 的形状为(n_keypoints, n_particles, 3)
        split_data = data_numpy.reshape(-1, paths_length, 5)[:, :, 2:]

        visualize_paths(n_particles, split_data)

        ### 先检查是否可以直接使用直线
        source = split_data[:, 0:1, :]
        target = split_data[:, -1:, :]
        straight_line = np.concatenate((source, target), axis=1)
        print(straight_line.shape)

        diffs = np.diff(straight_line, axis=1)
        dists = np.linalg.norm(diffs, axis=2)

        # 根据最短时间最大的路径确定插值数量
        v_max = 0.1
        max_time_min = np.max(dists) / v_max
        # 向上取整为 32.0/10000 的整数倍
        step = 32.0 / 10000
        num_keypoints = np.ceil(max_time_min / step) + 1


        ### 从起点到终点，检查kepoint是否是可删除的
        # 每组keypoint，根据起点和终点距离来确定优先级