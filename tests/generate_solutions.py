import numpy as np


def generate_solutions(n_particles, split_data, max_gorkov_idx, levitator):
    # 对最弱key points生成100个潜在solutions，并排序
    candidate_solutions = np.transpose(
        create_constrained_points_1(
            n_particles, 
            split_data[:, max_gorkov_idx, 2:], 
            split_data[:, max_gorkov_idx-1, 2:], 
            split_data[:, max_gorkov_idx+1, 2:]
        ), 
        (1, 0, 2)
    )

    # 计算 candidate_solutions 的 Gorkov
    solutions_gorkov = levitator.calculate_gorkov(candidate_solutions)
    # 找出每个 candidate_solutions 的最大 Gorkov
    solutions_max_gorkov = np.max(solutions_gorkov, axis=1)
    # 根据 Gorkov 对 candidate_solutions 从小到大排序
    sorted_indices = np.argsort(solutions_max_gorkov)
    sorted_solutions_max_gorkov = solutions_max_gorkov[sorted_indices]

    return candidate_solutions, sorted_indices, sorted_solutions_max_gorkov