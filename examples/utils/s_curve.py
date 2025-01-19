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
def visualize_all_particles(t, accelerations, velocities, trajectories):
    '''
    同时可视化所有粒子的加速度、速度和轨迹
    '''
    num_particles = accelerations.shape[0]
    colors = plt.cm.get_cmap('tab10', num_particles)  # 使用不同颜色绘制

    # 加速度和速度
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for i in range(num_particles):
        axs[0].plot(t, accelerations[i], label=f'Particle {i}', color=colors(i))
        axs[1].plot(t, velocities[i], label=f'Particle {i}', color=colors(i))

    axs[0].set_title('Acceleration vs Time')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Acceleration')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Velocity vs Time')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()
    axs[1].grid()

    # 三维轨迹
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