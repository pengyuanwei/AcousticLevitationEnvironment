import numpy as np
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
            max_a.append(sub_max_a)

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
        # visualize_all_particles(sum_t, sum_a, sum_v, sum_traj, jerks=None, show_paths=False)
        
        if not save:
            return np.array(max_a)
        else:
            # visualize_all_particles(sum_t, sum_a, sum_v, sum_traj, jerks=None, show_paths=False)
            return np.array(max_a), sum_t, sum_traj