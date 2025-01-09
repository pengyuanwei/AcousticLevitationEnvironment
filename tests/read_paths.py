import numpy as np
import math

def read_paths(csv_data, n_particles):
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

    return np.array(csv_data_float), max_length, include_NaN  


def read_paths(csv_data):
    include_NaN = False
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

    return np.array(csv_data_float), include_NaN 


def read_paths(csv_data):
    """
    读取路径数据并转换为浮点型数组，同时检测是否包含 NaN 值。

    参数:
        csv_data (list of list): 输入的 CSV 数据，每行应该包含 5 个值。
    
    返回:
        tuple: 包含两部分 (numpy array of floats, bool indicating if NaN is present)
    """
    try:
        # 转换为浮点数组，并过滤长度不是5的行
        csv_data_float = [
            [float(element) for element in row] 
            for row in csv_data if len(row) == 5
        ]
        
        # 转换为 numpy 数组
        csv_array = np.array(csv_data_float)
        
        # 检查是否存在 NaN 值
        if np.isnan(csv_array).any():
            return np.array([]), True

        return csv_array, False

    except ValueError:
        # 捕获不能转换为浮点数的异常
        return np.array([]), True