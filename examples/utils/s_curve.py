import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def smooth_acceleration_trajectory(start, end, total_time: float, dt):
    '''
    start: numpy array
    end: numpy array
    total_time: float
    '''
    start = np.array(start)
    end = np.array(end)
    # 计算轨迹长度和方向
    L = np.linalg.norm(end - start)  # 总路径长度
    direction = (end - start) / L   # 单位方向向量

    # 时间分段
    T_acc = total_time

    # 最大加速度和最大速度
    a_max = 4 * L / T_acc**2  # 最大加速度
    v_max = (1/2) * a_max * T_acc  # 最大速度

    # 时间数组
    t = np.arange(0, T_acc, dt)
    accelerations = []
    velocities = []
    positions = []

    for ti in t:
        if ti <= (T_acc / 2):
            # 加速阶段前半部分：加速度均匀增加
            a = (2 * a_max * ti) / T_acc
            v = (a_max * ti**2) / T_acc
            s = (1/3) * (a_max * ti**3) / T_acc
        elif ti <= T_acc:
            # 加速阶段后半部分：加速度均匀减少
            a = (-2 * a_max * ti) / T_acc + 2 * a_max
            v = (-1 * a_max * ti**2) / T_acc + 2 * a_max * ti - (a_max * T_acc) / 2
            s = (-1/3) * (a_max * ti**3) / T_acc + a_max * ti**2 - (1/2) * a_max * T_acc * ti + (1/12) * a_max * T_acc**2
        else:
            a = 0.0
            v = v_max
            s = L

        accelerations.append(a)
        velocities.append(v)
        positions.append(s)

    # 转换为 3D 轨迹
    positions = np.array(positions)
    trajectory = np.outer(positions, direction) + start

    print(L)
    print(direction)
    print(positions)

    return t, accelerations, velocities, trajectory

# 参数
start = (0, 0, 0)
end = (10, 10, 10)
total_time = 10.0
dt = 0.1

# 计算轨迹
t, accelerations, velocities, trajectory = smooth_acceleration_trajectory(start, end, total_time, dt)

# # 可视化速度
# plt.figure(figsize=(10, 6))
# plt.plot(t, velocities, label="Velocity (Smooth)", color='blue')
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (units/s)")
# plt.title("Velocity vs Time (Smooth Profile)")
# plt.grid()
# plt.legend()
# plt.show()

# # 可视化加速度
# plt.figure(figsize=(10, 6))
# plt.plot(t, accelerations, label="Acceleration (Smooth)", color='orange')
# plt.xlabel("Time (s)")
# plt.ylabel("Acceleration (units/s²)")
# plt.title("Acceleration vs Time (Smooth Profile)")
# plt.grid()
# plt.legend()
# plt.show()

# 可视化轨迹
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color='green')
ax.scatter(*start, color='red', label="Start", s=50)
ax.scatter(*end, color='blue', label="End", s=50)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Smooth Acceleration Trajectory in 3D")
ax.legend()
plt.show()