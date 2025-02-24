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
    v_max = 2 * L / total_time  # (N,)

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


# 随机生成 N 个粒子的起点和终点
def generate_random_particles(N, space_range):
    '''
    随机生成 N 个粒子的起点和终点
    space_range: 粒子的坐标范围 (x_min, x_max, y_min, y_max, z_min, z_max)
    '''
    x_min, x_max, y_min, y_max, z_min, z_max = space_range
    start = np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max], size=(N, 3))
    end = np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max], size=(N, 3))
    return start, end


# 可视化
def visualize_all_particles(t, accelerations, velocities, trajectories, jerks=None, show_paths=False):
    '''
    同时可视化所有粒子的加速度、速度和轨迹
    '''
    num_particles = accelerations.shape[0]
    colors = plt.cm.get_cmap('tab10', num_particles)  # 使用不同颜色绘制

    # 加速度和速度
    if jerks is not None:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for i in range(num_particles):
        axs[0].plot(t, velocities[i], label=f'Particle {i}', color=colors(i))
        axs[1].plot(t, accelerations[i], label=f'Particle {i}', color=colors(i))
        if jerks is not None:
            axs[2].plot(t, jerks[i], label=f'Particle {i}', color=colors(i))

    axs[0].set_title('Velocity vs Time')
    axs[0].set_ylabel('Velocity')
    #axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Acceleration vs Time')
    axs[1].set_ylabel('Acceleration')
    #axs[1].legend()
    axs[1].grid()

    if jerks is not None:
        axs[2].set_title('Jerk vs Time')
        axs[2].set_ylabel('Jerk')
        #axs[2].legend()
        axs[2].grid()

    # 三维轨迹
    if show_paths:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(num_particles):
            traj = trajectories[i]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Particle {i}', color=colors(i))
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=colors(i), marker='o')  # 起点
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors(i), marker='x')  # 终点

        ax.set_title('3D Trajectories of All Particles')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    plt.show()


def visualize_lengths(t, displacements):
    '''
    同时可视化所有粒子的位移
    '''
    num_particles = displacements.shape[0]
    colors = plt.cm.get_cmap('tab10', num_particles)  # 使用不同颜色绘制

    fig, axs = plt.subplots(1, 1, figsize=(12, 10))
    for i in range(num_particles):
        axs.plot(t, displacements[i], color=colors(i))

    axs.set_title('Dispalcement vs Time')
    axs.set_ylabel('Dispalcement')
    #axs.legend()
    axs.grid()
    plt.show()


def smooth_trajectories_arbitrary_initial_velocity(start: np.array, end: np.array, total_time: float, dt: float, velocities: np.array):
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

    # 初速度
    v_0 = velocities
    # 最大加速度和末速度
    a_max = 4 * (L - v_0 * total_time) / total_time ** 2  # (N,)
    v_2 = v_0 + 2 * (L - v_0 * total_time) / total_time  # (N,)

    # 修正时间以确保所有粒子速度<=0.1m/s
    if np.any(v_2 > 0.1 + 1e-9):
        t_lower_bound = 20 * L / (10 * v_0 + 1)
        total_time = np.max(t_lower_bound)
        #t_upper_bound = 2 * L / v_0
        #total_time = np.min(t_upper_bound)
        a_max = 4 * (L - v_0 * total_time) / total_time ** 2  # (N,)
        v_2 = v_0 + 2 * (L - v_0 * total_time) / total_time  # (N,)
    
    # 找出小于 0.0 的元素，并反向计算
    mask = v_2 < 0.0
    v_0[mask] = 2 * L[mask] / total_time
    a_max[mask] = -4 * L[mask] / total_time**2
    v_2[mask] = 0.0

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
            # 加速阶段前半部分
            a = (2 * a_max * ti) / total_time  # (N,)
            v = v_0 + (a_max * ti**2) / total_time  # (N,)
            s = v_0 * ti + (1/3) * (a_max * ti**3) / total_time  # (N,)
        elif ti <= total_time:
            # 加速阶段后半部分
            a = (-2 * a_max * ti) / total_time + 2 * a_max  # (N,)
            v = v_0 - (a_max * ti**2) / total_time + 2 * a_max * ti - (1/2) * a_max * total_time  # (N,)
            s = v_0 * ti - (1/3) * (a_max * ti**3) / total_time + a_max * ti**2 - (1/2) * a_max * total_time * ti + (1/12) * a_max * total_time**2  # (N,)
        else:
            a = np.zeros(N)  # 全部置为 0
            v = v_2
            s = L  # 终点

        accelerations[:, i] = a
        velocities[:, i] = v
        positions[:, i] = s

    # 计算三维轨迹
    trajectories = positions[:, :, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

    return t, accelerations, velocities, trajectories


def s_curve_smoothing_with_zero_end_velocity_simple(start: np.array, end: np.array, total_time: float, dt: float, velocities: np.array):
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

    # 初速度
    v_0 = velocities
    # 最大加速度和末速度
    #a_max = 4 * (L - v_0 * total_time) / total_time ** 2  # (N,)
    a_max = -2 * v_0 / total_time
    v_2 = 0.0  # (N,)

    # 时间数组
    t = np.arange(0, total_time+dt, dt)
    num_steps = len(t)

    # 初始化结果数组
    accelerations = np.zeros((N, num_steps))
    velocities = np.zeros((N, num_steps))
    positions = np.zeros((N, num_steps))

    # 时间点对应的加速度、速度和位移
    for i, ti in enumerate(t):
        if ti <= ((total_time+dt) / 2):
            # 加速阶段前半部分
            a = (2 * a_max * ti) / total_time  # (N,)
            v = v_0 + (a_max * ti**2) / total_time  # (N,)
            s = v_0 * ti + (1/3) * (a_max * ti**3) / total_time  # (N,)
        elif ti <= (total_time+dt):
            # 加速阶段后半部分
            a = (-2 * a_max * ti) / total_time + 2 * a_max  # (N,)
            v = v_0 - (a_max * ti**2) / total_time + 2 * a_max * ti - (1/2) * a_max * total_time  # (N,)
            s = v_0 * ti - (1/3) * (a_max * ti**3) / total_time + a_max * ti**2 - (1/2) * a_max * total_time * ti + (1/12) * a_max * total_time**2  # (N,)
        else:
            a = np.zeros(N)  # 全部置为 0
            v = v_2
            s = L  # 终点

        accelerations[:, i] = a
        velocities[:, i] = v
        positions[:, i] = s

    # 计算三维轨迹
    trajectories = positions[:, :, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

    return t, accelerations, velocities, trajectories


def uniformly_accelerated_with_arbitrary_initial_velocity(start: np.array, end: np.array, total_time: float, dt: float, velocities: np.array):
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

    # 初速度
    v_0 = velocities
    # 加速度和末速度
    a_1 = 2 * (L - v_0 * total_time) / total_time ** 2  # (N,)
    v_1 = v_0 + 2 * (L - v_0 * total_time) / total_time  # (N,)

    if np.any(v_1 > 0.1 + 1e-9):
        t_lower_bound = 20 * L / (10 * v_0 + 1)
        total_time = np.max(t_lower_bound)
        #t_upper_bound = 2 * L / v_0
        #total_time = np.min(t_upper_bound)
        a_1 = 2 * (L - v_0 * total_time) / total_time ** 2  # (N,)
        v_1 = v_0 + 2 * (L - v_0 * total_time) / total_time  # (N,)
    
    # 找出小于 0.0 的元素，并反向计算
    mask = v_1 < 0.0
    v_0[mask] = 2 * L[mask] / total_time
    a_1[mask] = -2 * L[mask] / total_time**2
    v_1[mask] = 0.0

    # 时间数组
    t = np.arange(0, total_time, dt)
    num_steps = len(t)

    # 初始化结果数组
    accelerations = np.zeros((N, num_steps))
    velocities = np.zeros((N, num_steps))
    positions = np.zeros((N, num_steps))

    # 时间点对应的加速度、速度和位移
    for i, ti in enumerate(t):
        if ti == 0:
            # 初始状态
            a = 0.0
            v = 0.0
            s = 0.0
        elif ti <= total_time:
            # 匀加速阶段
            a = a_1
            v = v_0 + a * ti  # (N,)
            s = v_0 * ti + (1/2) * a * ti**2  # (N,)
        else:
            a = 0.0
            v = v_1  # 最大速度保持
            s = L  # 终点

        accelerations[:, i] = a
        velocities[:, i] = v
        positions[:, i] = s

    # 计算三维轨迹
    trajectories = positions[:, :, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

    return t, accelerations, velocities, trajectories


def uniformly_accelerated_with_zero_end_velocity(start: np.array, end: np.array, total_time: float, dt: float, velocities: np.array):
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

    # 初速度
    v_0 = velocities
    # 加速度和末速度
    v_1 = 0.0
    a_1 = v_1 - v_0 / total_time  # (N,)

    # 时间数组
    t = np.arange(0, total_time, dt)
    num_steps = len(t)

    # 初始化结果数组
    accelerations = np.zeros((N, num_steps))
    velocities = np.zeros((N, num_steps))
    positions = np.zeros((N, num_steps))

    # 时间点对应的加速度、速度和位移
    for i, ti in enumerate(t):
        if ti == 0.0:
            # 初始状态
            a = 0.0
            v = v_0  # (N,)
            s = 0.0  # (N,)            
        elif ti <= total_time:
            # 匀加速阶段
            a = a_1
            v = v_0 + a * ti  # (N,)
            s = v_0 * ti + (1/2) * a * ti**2  # (N,)
        else:
            a = 0.0
            v = v_1  # 最大速度保持
            s = L  # 终点

        accelerations[:, i] = a
        velocities[:, i] = v
        positions[:, i] = s

    # 计算三维轨迹
    trajectories = positions[:, :, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

    return t, accelerations, velocities, trajectories


def calculate_jerk(time_series, acceleration_data):
    """
    Calculate the Jerk for a given time series and acceleration data.

    Parameters:
    - time_series (array-like): An array of time points.
    - acceleration_data (2D array-like): A 2D array where each row corresponds to a time point
      and each column corresponds to an agent's acceleration.

    Returns:
    - jerk_data (2D numpy array): The calculated Jerk for each agent at each time point (except the first).
    """
    # Check that dimensions match
    if len(time_series) != acceleration_data.shape[1]:
        raise ValueError("The length of time_series must match the number of rows in acceleration_data.")

    # Calculate time differences
    dt = np.diff(time_series)
    if np.any(dt <= 0):
        raise ValueError("Time series must be strictly increasing.")

    # Calculate Jerk: derivative of acceleration (da/dt)
    jerk_data = np.diff(acceleration_data, axis=1) / dt[None, :]

    return jerk_data


def uniform_velocity_interpolation_simple(start: np.array, end: np.array, total_time: float, dt: float):
    '''
    输入：
        start: (N, 3) 每个粒子的起点
        end: (N, 3) 每个粒子的终点
        total_time: 总时间（所有粒子相同）
        dt: 时间步长
    输出：
        t: 时间数组
        trajectories: (N, len(t), 3) 每个粒子的轨迹
    '''
    # 粒子数量
    N = start.shape[0]

    # 计算路径长度和方向
    L = np.linalg.norm(end - start, axis=1)  # (N,) 每个粒子的总路径长度
    # 初始化方向向量，默认所有为零向量
    direction = np.zeros_like(end)
    # 对于路径长度不为0的粒子，计算单位方向向量
    nonzero = L > 0
    direction[nonzero] = (end - start)[nonzero] / L[nonzero, np.newaxis]

    # 时间数组
    t = np.arange(0, total_time, dt)
    num_steps = len(t)

    segments = np.array([np.linspace(0, ds, num_steps) for ds in L])
    final_v = segments[:, -1] / total_time

    # 计算三维轨迹
    trajectories = segments[:, :-1, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

    return t[:-1], trajectories, final_v


def uniform_velocity_interpolation_v2(
        start: np.array, 
        end: np.array, 
        total_time: float, 
        dt: float, 
        velocities: np.array
    ):
    '''
    输入：
        start: (N, 3) 每个粒子的起点
        end: (N, 3) 每个粒子的终点
        total_time: 总时间（所有粒子相同）
        dt: 时间步长
        velocities: 上一段轨迹的最后一段位移的平均速度，用于计算加速度
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
    # 初始化方向向量，默认所有为零向量
    direction = np.zeros_like(end)
    # 对于路径长度不为0的粒子，计算单位方向向量
    nonzero = L > 0
    direction[nonzero] = (end - start)[nonzero] / L[nonzero, np.newaxis]

    # 时间数组
    t = np.arange(0, total_time, dt)
    num_steps = len(t)

    # 计算速度和加速度
    v_0 = velocities
    segments = np.array([np.linspace(0, ds, num_steps) for ds in L])
    v_1 = segments[:, -1] / total_time
    a_max = (v_1 - v_0) / dt  # (N,)
    # 初始化结果数组
    accelerations = np.zeros((N, num_steps))
    velocities = np.zeros((N, num_steps))

    # 时间点对应的加速度、速度和位移
    for i, ti in enumerate(t):
        if i == 0:
            # 初始状态
            v = v_0
            a = 0.0
        elif i == 1:
            # 瞬时加速
            v = v_1
            a = a_max
        else:
            # 匀速阶段
            v = v_1  # (N,)
            a = 0.0

        accelerations[:, i] = a
        velocities[:, i] = v

    # 计算三维轨迹
    trajectories = segments[:, :-1, np.newaxis] * direction[:, np.newaxis, :] + start[:, np.newaxis, :]

    return t[:-1], accelerations[:, :-1], velocities[:, :-1], trajectories, v_1


def uniform_dt_interpolation(trajectories: np.array, delta_times: np.array, base_dt: float = 0.0032):
    """
    对一组粒子的同步轨迹进行匀速直线插值，
    使得输出轨迹的时间步长变为 base_dt (默认 0.0032 秒)，
    同时计算每个插值点对应的累积时间，保证两者长度一致。

    参数：
        trajectories: np.array, 形状 (N, M, D)
            N 个粒子在 M 个同步时刻的轨迹数据，每个时刻的坐标维数为 D。
        delta_times: np.array, 形状 (M-1,)
            每段轨迹（相邻时刻之间）的时间间隔，要求每个值都是 base_dt 的整数倍。
        base_dt: float, 默认 0.0032
            目标时间步长。

    返回：
        accumulated_time: np.array, 一维数组
            每个插值点对应的累积时间，从 0 开始累加，与 new_trajectories 的时间步数匹配。
        new_trajectories: np.array, 形状 (N, new_M, D)
            经插值后的轨迹数据，时间步长均为 base_dt。
    """
    N, M, D = trajectories.shape
    new_traj_list = []          # 用于存储各段插值后的轨迹数据
    accumulated_time_list = []  # 用于存储各段对应的累积时间
    current_time = 0.0          # 累计时间初始值

    # 遍历每一段（相邻两个同步时刻之间）
    for i in range(M - 1):
        dt_segment = delta_times[i]
        # 计算该段的步数（dt_segment 必须为 base_dt 的整数倍）
        n_steps = int(round(dt_segment / base_dt))
        # 生成 n_steps+1 个插值点（包括起点与终点）
        interp_segment = np.linspace(trajectories[:, i, :], trajectories[:, i+1, :], n_steps+1, axis=1)

        # 为避免各段连接处重复采样（重复包含终点），除最后一段外均舍去末端点
        if i < M - 2:
            t_segment = np.arange(n_steps) * base_dt
            interp_segment = interp_segment[:, :-1, :]
        else:
            t_segment = np.arange(n_steps+1) * base_dt
        
        new_traj_list.append(interp_segment)
        accumulated_time_list.append(current_time + t_segment)
        # 更新累计时间
        current_time += n_steps * base_dt

    # 拼接各段数据
    accumulated_time = np.concatenate(accumulated_time_list)
    new_trajectories = np.concatenate(new_traj_list, axis=1)

    return accumulated_time, new_trajectories