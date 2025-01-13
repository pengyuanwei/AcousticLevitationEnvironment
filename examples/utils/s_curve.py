import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def smooth_acceleration_trajectories(start: np.array, end: np.array, total_time: float, dt: float):
    '''
    输入：
        start: (N, 3) 每个粒子的起点
        end: (N, 3) 每个粒子的终点
        total_time: 总时间（所有粒子相同）
        dt: 时间步长
    输出：
        t: 时间数组
        accelerations: (N, len(t)) 每个粒子随时间的加速度
        velocities: (N, len(t)) 每个粒子随时间的速度
        trajectories: (N, len(t), 3) 每个粒子的轨迹
    '''
    # 粒子数量
    N = start.shape[0]

    # 计算路径长度和方向
    L = np.linalg.norm(end - start, axis=1)  # (N,) 每个粒子的总路径长度
    direction = (end - start) / L[:, np.newaxis]  # (N, 3) 单位方向向量

    # 最大加速度和最大速度
    a_max = 4 * L / total_time**2  # (N,)
    v_max = 0.5 * a_max * total_time  # (N,)

    # 时间数组
    t = np.arange(0, total_time, dt)
    num_steps = len(t)

    # 初始化结果数组
    accelerations = np.zeros((N, num_steps))
    velocities = np.zeros((N, num_steps))
    positions = np.zeros((N, num_steps))

    # 时间点对应的加速度、速度和位移
    for i, ti in enumerate(t):
        if ti <= (total_time / 2):
            # 加速阶段前半部分：加速度均匀增加
            a = (2 * a_max * ti) / total_time  # (N,)
            v = (a_max * ti**2) / total_time  # (N,)
            s = (1/3) * (a_max * ti**3) / total_time  # (N,)
        elif ti <= total_time:
            # 加速阶段后半部分：加速度均匀减少
            a = (-2 * a_max * ti) / total_time + 2 * a_max  # (N,)
            v = (-1 * a_max * ti**2) / total_time + 2 * a_max * ti - (a_max * total_time) / 2  # (N,)
            s = (-1/3) * (a_max * ti**3) / total_time + a_max * ti**2 - (1/2) * a_max * total_time * ti + (1/12) * a_max * total_time**2  # (N,)
        else:
            a = np.zeros(N)  # 全部置为 0
            v = v_max  # 最大速度保持
            s = L  # 终点

        accelerations[:, i] = a
        velocities[:, i] = v
        positions[:, i] = s

    # 计算三维轨迹
    trajectories = positions[:, :, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

    return t, accelerations, velocities, trajectories