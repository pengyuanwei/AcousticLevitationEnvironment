import numpy as np
import math


def check_and_correct_positions(positions, step_size=0.001, max_iterations=100, bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.06+0.12, 0.06+0.12))):
    """
    检查无人机之间的距离是否满足约束，并根据三维斥力场修正坐标。

    Args:
        positions (np.ndarray): 无人机的位置数组，形状为 (n, 3)，其中 n 是无人机数量。
        step_size (float): 修正坐标的步长。
        max_iterations (int): 最大迭代次数。
        bounds (tuple): 三维空间的边界，格式为 ((x_min, x_max), (y_min, y_max), (z_min, z_max))。

    Returns:
        np.ndarray: 修正后的无人机位置数组。
        bool: 是否所有无人机均满足距离约束。
    """
    positions = np.array(positions)  # 确保输入是 NumPy 数组
    num_drones = positions.shape[0]

    for iteration in range(max_iterations):
        forces = np.zeros_like(positions)  # 初始化所有无人机的斥力

        # 计算斥力
        for i in range(num_drones):
            for j in range(num_drones):
                if i != j:                
                    diff = positions[i] - positions[j]  # 方向向量，the vector from agent to obstacle
                    distance = np.linalg.norm(diff)    # 距离

                    distance_xy = np.linalg.norm(diff[:2])  # x 和 y 平面上的距离
                    distance_z = abs(diff[2])              # z 方向上的距离

                    dist = math.sqrt((diff[0])**2/0.015**2 + 
                                     (diff[1])**2/0.015**2 + 
                                     (diff[2])**2/0.031**2)
                    if dist <= 1.0:
                        # 计算斥力：反方向，大小与距离的平方成反比
                        force_xy = 1e-5 * (1.0/distance - 1.0/0.015) * (1.0/(distance**2 + 1e-6)) * (diff[:2]/(distance_xy + 1e-6))
                        force_z = 1e-5 * (1.0/distance - 1.0/0.031) * (1.0/(distance**2 + 1e-6)) * (diff[2]/(distance_z + 1e-6))
                        
                        repulsion_force = np.zeros(3)
                        repulsion_force[:2] = force_xy  # x 和 y 方向的斥力
                        repulsion_force[2] = force_z   # z 方向的斥力

                        forces[i] += repulsion_force

        # 根据合成力修正位置
        positions += step_size * forces

        # 投影到边界内
        if bounds is not None:
            for i in range(3):  # 对每个维度进行投影
                positions[:, i] = np.clip(positions[:, i], bounds[i][0], bounds[i][1])

        # 检查所有无人机之间的距离是否满足约束
        all_satisfied = True
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                dist = math.sqrt((positions[i][0] - positions[j][0])**2/0.014**2 + 
                                 (positions[i][1] - positions[j][1])**2/0.014**2 + 
                                 (positions[i][2] - positions[j][2])**2/0.03**2)
                if dist <= 1.0:
                    all_satisfied = False

        if all_satisfied:
            print(f"约束在第 {iteration + 1} 次迭代后满足。")
            return positions, True

    print("达到最大迭代次数，约束可能未完全满足。")
    return positions, False


# 示例使用
if __name__ == "__main__":
    # 初始位置
    initial_positions = [
        [0.01, 0.0, 0.13],
        [0.0, 0.0, 0.13]
    ]

    corrected_positions, satisfied = check_and_correct_positions(initial_positions)
    print("修正后的位置:")
    print(corrected_positions)
    print("是否满足距离约束:", satisfied)