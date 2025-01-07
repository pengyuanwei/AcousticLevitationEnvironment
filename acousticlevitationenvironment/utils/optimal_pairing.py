import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def optimal_pairing(coords1, coords2):
    # 计算两组坐标之间的距离矩阵
    distance_matrix = cdist(coords1, coords2, metric='euclidean')

    # 使用匈牙利算法找到最优匹配
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # 返回最优匹配
    return list(zip(row_indices, col_indices))