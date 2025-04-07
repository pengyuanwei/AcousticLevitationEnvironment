import numpy as np
from scipy.interpolate import interp1d
from examples.utils.path_smoothing_2 import *


def kinodynamics_analysis(
        n_particles: int, 
        split_data: np.array, 
        delta_time: np.array,
        save: bool=False
    ):
        # 初始化
        t = []
        accelerations = []
        velocities = []
        trajectories = []
        max_a = []
        a_sum = []

        dt = 32.0/10000
        sub_initial_t = 0.0
        sub_initial_v = np.zeros((n_particles,))


        # 第一段匀加速
        sub_t, sub_accelerations, sub_velocities, sub_trajectories = smooth_trajectories_arbitrary_initial_velocity(
            split_data[:, 0, 2:], split_data[:, 1, 2:], delta_time[0], dt=dt, velocities=sub_initial_v
        )

        sub_t += sub_initial_t
        sub_initial_t = sub_t[-1] + dt
        sub_initial_v = sub_velocities[:, -1]

        t.append(sub_t)
        accelerations.append(sub_accelerations)
        velocities.append(sub_velocities)
        trajectories.append(sub_trajectories)  

        # 中间段匀速直线
        for i in range(1, split_data.shape[1]-2):
            sub_t, sub_accelerations, sub_velocities, sub_trajectories, sub_initial_v = uniform_velocity_interpolation_v2(
                start=split_data[:, i, 2:], end=split_data[:, i+1, 2:], total_time=delta_time[i], dt=dt, velocities=sub_initial_v
            )

            sub_t += sub_initial_t
            sub_initial_t = sub_t[-1] + dt

            sub_max_a = np.max(abs(sub_accelerations))
            sub_a_sum = np.sum(abs(sub_accelerations))
            max_a.append(sub_max_a)
            a_sum.append(sub_a_sum)

            t.append(sub_t)
            accelerations.append(sub_accelerations)
            velocities.append(sub_velocities)
            trajectories.append(sub_trajectories)  

        # 最后一段匀减速
        sub_t, sub_accelerations, sub_velocities, sub_trajectories = s_curve_smoothing_with_zero_end_velocity_simple(
            split_data[:, -2, 2:], split_data[:, -1, 2:], delta_time[-1], dt=dt, velocities=sub_initial_v
        )

        sub_t += sub_initial_t
        sub_initial_t = sub_t[-1] + dt
        sub_initial_v = sub_velocities[:, -1]

        t.append(sub_t)
        accelerations.append(sub_accelerations)
        velocities.append(sub_velocities)
        trajectories.append(sub_trajectories)  


        # 可视化速度和加速度
        # 将所有子数组沿 axis=1 拼接成一个总数组
        sum_t = np.concatenate(t, axis=0)
        sum_a = np.concatenate(accelerations, axis=1)
        sum_v = np.concatenate(velocities, axis=1)
        sum_traj = np.concatenate(trajectories, axis=1)
        #visualize_all_particles(sum_t, sum_a, sum_v, sum_traj, jerks=None, show_paths=False)
        
        if not save:
            return np.array(max_a), np.array(a_sum)
        else:
            visualize_all_particles(sum_t, sum_a, sum_v, sum_traj, jerks=None, show_paths=False)
            return np.array(max_a), np.array(a_sum), sum_t, sum_traj
        

def kinodynamics_analysis_v2(
        n_particles: int, 
        split_data: np.array, 
        delta_time: np.array,
        save: bool=False
    ):
        '''
        两端无 S-curve smoothing
        '''
        # 初始化
        t = []
        accelerations = []
        velocities = []
        trajectories = []

        dt = 32.0/10000
        sub_initial_t = 0.0
        sub_initial_v = np.zeros((n_particles,))

        # 所有段匀速直线
        for i in range(split_data.shape[1]-1):
            sub_t, sub_accelerations, sub_velocities, sub_trajectories, sub_initial_v = uniform_velocity_interpolation_v2(
                start=split_data[:, i, 2:], end=split_data[:, i+1, 2:], total_time=delta_time[i], dt=dt, velocities=sub_initial_v
            )

            if sub_t.shape[0] > 0:
                sub_t += sub_initial_t
                sub_initial_t = sub_t[-1] + dt
            else:
                sub_initial_t += dt

            t.append(sub_t)
            accelerations.append(sub_accelerations)
            velocities.append(sub_velocities)
            trajectories.append(sub_trajectories)  


        # 可视化速度和加速度
        # 将所有子数组沿 axis=1 拼接成一个总数组
        sum_t = np.concatenate(t, axis=0)
        sum_a = np.concatenate(accelerations, axis=1)
        sum_v = np.concatenate(velocities, axis=1)
        sum_traj = np.concatenate(trajectories, axis=1)
        #visualize_all_particles(sum_t, sum_a, sum_v, sum_traj, jerks=None, show_paths=False)
        
        visualize_all_particles(sum_t, sum_a, sum_v, sum_traj, jerks=None, show_paths=False)
        return None, None, sum_t, sum_traj
        

def visualize_all_particles_v2(t, velocities, accelerations, jerks=None):
    '''
    同时可视化所有粒子的速度、加速度
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

    axs[0].set_title('TWGS vs Time')
    axs[0].set_ylabel('Gor\'kov Potential')
    #axs[0].legend()
    axs[0].grid()

    axs[1].set_title('WGS vs Time')
    axs[1].set_ylabel('Gor\'kov Potential')
    #axs[1].legend()
    axs[1].grid()

    if jerks is not None:
        axs[2].set_title('Jerk vs Time')
        axs[2].set_ylabel('Jerk')
        #axs[2].legend()
        axs[2].grid()

    plt.show()


def visualize_all_particles_v3(t, velocities):
    '''
    同时可视化所有粒子的速度
    '''
    num_particles = velocities.shape[0]
    colors = plt.cm.get_cmap('tab10', num_particles)  # 使用不同颜色绘制

    fig, ax = plt.subplots(figsize=(12, 6))  # 只创建一个子图
    for i in range(num_particles):
        ax.plot(t, velocities[i], label=f'Particle {i}', color=colors(i))  # 颜色映射

    ax.set_title('Velocity vs Time')
    ax.set_ylabel('Velocity')
    #ax.legend()
    ax.grid()

    plt.show()
    

def visualize_all_particles_v4(t_velocities, velocities, t_accelerations, accelerations, t_jerks=None, jerks=None):
    '''
    可视化所有粒子的速度、加速度，并支持不同的时间轴
    '''
    num_particles = accelerations.shape[0]
    colors = plt.cm.get_cmap('tab10', num_particles)  # 使用不同颜色绘制

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Velocity', color='tab:blue')
    ax1.set_title('Particle Dynamics Over Time')
    
    # 绘制速度曲线
    for i in range(num_particles):
        ax1.plot(t_velocities, velocities[i], label=f'Velocity {i}', color=colors(i))
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid()
    
    # 创建第二个 y 轴绘制加速度
    ax2 = ax1.twinx()
    ax2.set_ylabel('Acceleration', color='tab:red')
    for i in range(num_particles):
        ax2.plot(t_accelerations, accelerations[i], linestyle='dashed', label=f'Acceleration {i}', color=colors(i))
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # 如果存在 jerk 数据，则再创建一个次 y 轴
    if jerks is not None and t_jerks is not None:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # 偏移一点以避免重叠
        ax3.set_ylabel('Jerk', color='tab:green')
        for i in range(num_particles):
            ax3.plot(t_jerks, jerks[i], linestyle='dotted', label=f'Jerk {i}', color=colors(i))
        ax3.tick_params(axis='y', labelcolor='tab:green')
    
    fig.tight_layout()
    plt.show()


def calculate_v_a(paths: np.array, delta_time: float=0.0032, debug: bool=False):
    '''
    轨迹的速度与加速度分析。
    速度：source points v=0，每一段的平均速度，target points v=0
    加速度：source points a=0，每个平均速度之间的加速度（包含从v_{T-1}到v_T）
    输入：
        paths: (N, paths_length, 3)
        delta_time: 每段位移的时间变化
    输出：
        velocities: (N, paths_length+1, 3) 每个粒子随时间的速度
        accelerations: (N, paths_length+1, 3) 每个粒子随时间的加速度
    '''
    # 粒子数量
    N = paths.shape[0]
    if debug:
        print(paths.shape)

    # 计算路径长度和方向
    # 每个粒子的每段位移vectors: (N, paths_length-1, 3)
    vectors = np.diff(paths, axis=1)
    if debug:
        print(vectors.shape)

    # 每段位移的平均速度velocities：(N, paths_length-1, 3)
    velocities = vectors / delta_time
    # 端点速度
    edge_velocities = np.zeros((N, 1, 3))
    # 速度velocities：(N, paths_length+1, 3)
    velocities = np.concatenate((edge_velocities, velocities, edge_velocities), axis=1)
    if debug:
        print(velocities.shape)

    # 平均速度之间的加速度accelerations
    diff_v = np.diff(velocities, axis=1)
    # (N, paths_length, 3)
    accelerations = diff_v / delta_time
    # 初始位置加速度
    a_0 = np.zeros((N, 1, 3))
    # (N, paths_length+1, 3)
    accelerations = np.concatenate((a_0, accelerations), axis=1)
    if debug:
        print(accelerations.shape)

    return velocities, accelerations


def moving_sum(arr, window_size, axis):
    return np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window_size, dtype=int), mode='valid'),
        axis=axis,
        arr=arr
    )


def visualize_paths(num_particles, trajectories):
    '''
    绘制轨迹
    Input:
         - trajectories: (num_particles, paths_length, 3)
    '''
    colors = plt.cm.get_cmap('tab10', num_particles)  # 使用不同颜色绘制

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


def interpolate_trajectories(trajectories, delta_time):
    """
    对所有轨迹进行线性插值，使用给定的delta_time来生成更多的时间点。
    
    参数:
    - trajectories: 形状为(N, M, 4)的numpy数组，包含N个agent的M个关键点及其时间和位置。
    - delta_time: 目标的时间间隔，用于插值。
    
    返回:
    - interpolated_trajectories: 插值后的轨迹，形状为(N, new_M, 4)，其中new_M是插值后的关键点数量。
    """
    N, M, _ = trajectories.shape
    
    # 结果的时间间隔和新的时间点数量
    interpolated_trajectories = []
    
    for i in range(N):
        # 获取当前agent的轨迹
        agent_trajectory = trajectories[i]  # shape (M, 4)
        
        # 时间 t 和位置 (x, y, z)
        times = agent_trajectory[:, 0]
        positions = agent_trajectory[:, 1:]
        
        # 计算原始时间点之间的时间差
        original_deltas = np.diff(times)  # size (M-1,)
        
        # 确保每个时间差是 delta_time 的整数倍
        assert np.all(np.isclose((original_deltas+1e-9) % delta_time, 0.0)), "时间差必须是delta_time的整数倍"
        
        # 根据原始时间点的时间差，确定新的时间点
        new_times = []
        for j in range(M-1):
            # 当前时间点和下一个时间点之间的时间差
            start_time = times[j]
            end_time = times[j+1]
            num_new_points = int((end_time - start_time) / delta_time)  # 插值的点数
            new_times.extend(np.arange(start_time, end_time, delta_time))
        
        new_times.append(times[-1])  # 确保最后一个时间点被添加
        
        # 对每个坐标进行插值
        new_positions = np.zeros((len(new_times), 3))  # shape (new_M, 3)
        
        for j in range(3):
            # 使用线性插值对每个坐标进行插值
            interpolator = interp1d(times, positions[:, j], kind='linear', fill_value='extrapolate')
            new_positions[:, j] = interpolator(new_times)
        
        # 将插值后的时间和位置合并
        new_trajectory = np.column_stack((new_times, new_positions))  # shape (new_M, 4)
        interpolated_trajectories.append(new_trajectory)
    
    # 转换为numpy数组
    interpolated_trajectories = np.array(interpolated_trajectories)
    
    return interpolated_trajectories


def compute_angle(v1, v2):
    """计算两个向量之间的夹角，返回弧度"""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # 或者 return np.nan，如果你想之后过滤掉这类点

    dot_product = np.dot(v1, v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # 保证cos值在合法范围内，防止浮点误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


# def find_turning_points(trajectories, angle_threshold=np.pi/32):
#     """
#     查找转折点
#     trajectories: 形状为(N, M, 4)的轨迹数据，包含时间和坐标
#     angle_threshold: 转折角度阈值（弧度）
#     """
#     N, M, _ = trajectories.shape
#     turning_points = []

#     # 遍历每个agent
#     for i in range(N):
#         agent_turning_points = []
#         # 遍历该agent的所有关键点
#         for j in range(1, M-1):
#             # 获取相邻的方向向量
#             v1 = trajectories[i, j] - trajectories[i, j-1]  # v1: P(j) 到 P(j-1)
#             v2 = trajectories[i, j+1] - trajectories[i, j]  # v2: P(j+1) 到 P(j)
            
#             # 计算夹角
#             angle = compute_angle(v1[1:], v2[1:])  # 忽略时间维度，只考虑空间坐标 (x, y, z)
            
#             # 判断是否为转折点
#             if angle > angle_threshold:
#                 agent_turning_points.append(j)
#                 print(f"Particle {i}, point {j}:", v1[1:], v2[1:])
        
#         turning_points.append(agent_turning_points)
    
#     return turning_points


def find_turning_points(trajectories, angle_threshold=np.pi/4):
    """
    返回所有转折点的索引元组 (agent_id, point_id)
    """
    N, M, _ = trajectories.shape
    turning_points = []

    for i in range(N):
        for j in range(1, M - 1):
            v1 = trajectories[i, j] - trajectories[i, j - 1]
            v2 = trajectories[i, j + 1] - trajectories[i, j]
            
            # 只计算空间部分
            v1_xyz = v1[1:]
            v2_xyz = v2[1:]

            norm_v1 = np.linalg.norm(v1_xyz)
            norm_v2 = np.linalg.norm(v2_xyz)

            if norm_v1 == 0 or norm_v2 == 0:
                continue  # 跳过静止点

            dot_product = np.dot(v1_xyz, v2_xyz)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)

            if angle > angle_threshold:
                turning_points.append((i, j))

    return turning_points