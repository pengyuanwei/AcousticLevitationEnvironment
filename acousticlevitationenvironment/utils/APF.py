import numpy as np
from typing import Tuple

def compute_repulsion_force(diff, distance, distance_xy, distance_z):
    """
    计算斥力，根据方向向量和距离。
    """
    force_xy = 1e-5 * (1.0 / distance - 1.0 / 0.015) * (1.0 / (distance**2 + 1e-6)) * (diff[:2] / (distance_xy + 1e-6))
    force_z = 1e-5 * (1.0 / distance - 1.0 / 0.031) * (1.0 / (distance**2 + 1e-6)) * (diff[2] / (distance_z + 1e-6))
    repulsion_force = np.zeros(3)
    repulsion_force[:2] = force_xy
    repulsion_force[2] = force_z
    return repulsion_force


def check_distances(positions, thresholds=(0.014, 0.014, 0.03)) -> bool:
    """
    检查所有无人机之间的距离是否满足约束。
    """
    num_drones = positions.shape[0]
    for i in range(num_drones):
        for j in range(i + 1, num_drones):  # 避免重复计算
            diff = positions[i] - positions[j]
            dist_square = (diff[0]**2 / thresholds[0]**2) + (diff[1]**2 / thresholds[1]**2) + (diff[2]**2 / thresholds[2]**2)
            if dist_square <= 1.0:
                return False
    return True


def check_and_correct_positions(positions: np.ndarray, step_size=0.001, max_iterations=100,
                                bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.06 + 0.12, 0.06 + 0.12))) -> Tuple[np.ndarray, bool]:
    """
    检查无人机之间的距离是否满足约束，并根据三维斥力场修正坐标。

    Args:
        positions (np.ndarray): 无人机的位置数组，形状为 (n, 3)。
        step_size (float): 修正坐标的步长。
        max_iterations (int): 最大迭代次数。
        bounds (tuple): 三维空间的边界，格式为 ((x_min, x_max), (y_min, y_max), (z_min, z_max))。

    Returns:
        np.ndarray: 修正后的无人机位置数组。
        bool: 是否所有无人机均满足距离约束。
    """
    positions = np.array(positions)
    num_drones = positions.shape[0]

    for iteration in range(max_iterations):
        forces = np.zeros_like(positions)  # 初始化所有无人机的斥力

        # 计算斥力
        for i in range(num_drones):
            for j in range(i + 1, num_drones):  # 避免重复计算
                diff = positions[i] - positions[j]
                distance = np.linalg.norm(diff)
                distance_xy = np.linalg.norm(diff[:2])
                distance_z = abs(diff[2])

                dist_square = (diff[0]**2 / 0.015**2) + (diff[1]**2 / 0.015**2) + (diff[2]**2 / 0.031**2)
                if dist_square <= 1.0:
                    repulsion_force = compute_repulsion_force(diff, distance, distance_xy, distance_z)
                    forces[i] += repulsion_force
                    forces[j] -= repulsion_force  # 反作用力

        # 根据合成力修正位置
        positions += step_size * forces

        # 投影到边界内
        if bounds is not None:
            positions = np.clip(positions, [b[0] for b in bounds], [b[1] for b in bounds])

        # 检查是否满足约束
        if check_distances(positions):
            return positions, True

    print("APF: 达到最大迭代次数，约束未完全满足。")
    return positions, False


def check_and_correct_positions_fixed(
        positions: np.array, 
        reach_index: np.array,
        step_size: float=0.001, 
        max_iterations: int=100,
        bounds: tuple=((-0.06, 0.06), (-0.06, 0.06), (-0.06 + 0.12, 0.06 + 0.12))
    ) -> Tuple[np.ndarray, bool]:
    """
    检查无人机之间的距离是否满足约束，并根据三维斥力场修正坐标。

    Args:
        positions (np.ndarray): 无人机的位置数组，形状为 (n, 3)。
        reach_index: 标记是否到达终点, 到达终点为1。
        step_size (float): 修正坐标的步长。
        max_iterations (int): 最大迭代次数。
        bounds (tuple): 三维空间的边界，格式为 ((x_min, x_max), (y_min, y_max), (z_min, z_max))。

    Returns:
        np.ndarray: 修正后的无人机位置数组。
        bool: 是否所有无人机均满足距离约束。
    """
    num_drones = positions.shape[0]
    for _ in range(max_iterations):
        forces = np.zeros_like(positions)  # 初始化所有无人机的斥力

        # 计算斥力
        for i in range(num_drones):
            for j in range(i + 1, num_drones):  # 避免重复计算
                diff = positions[i] - positions[j]
                distance = np.linalg.norm(diff)
                distance_xy = np.linalg.norm(diff[:2])
                distance_z = abs(diff[2])

                dist_square = (diff[0]**2 / 0.015**2) + (diff[1]**2 / 0.015**2) + (diff[2]**2 / 0.031**2)
                if dist_square <= 1.0:
                    repulsion_force = compute_repulsion_force(diff, distance, distance_xy, distance_z)
                    forces[i] += repulsion_force
                    forces[j] -= repulsion_force  # 反作用力

        # 根据合成力修正位置
        positions += (1-reach_index) * step_size * forces

        # 投影到边界内
        if bounds is not None:
            positions = np.clip(positions, [b[0] for b in bounds], [b[1] for b in bounds])

        # 检查是否满足约束
        if check_distances(positions):
            return positions, True

    print("APF: 达到最大迭代次数，约束未完全满足。")
    return positions, False