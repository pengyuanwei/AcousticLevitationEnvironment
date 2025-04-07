import os
import numpy as np

from examples.utils.top_bottom_setup import top_bottom_setup
from examples.utils.general_utils import *
from examples.utils.acoustic_utils import *
from examples.utils.optimizer_utils import *
from examples.utils.path_smoothing_2 import *
from examples.utils.path_smoothing_3 import *
from examples.utils.TWGS_Josh_v3 import *

# Modified based on the path_analysis_v2.py: 尝试Josh的TWGS实现

def calculate_gorkov(paths, levitator):
    '''
    paths: (N, paths_length, 3)
    '''
    N = paths.shape[0]
    T_in = torch.pi/32 #Hologram phase change threshold
    T_out = 0 #Point activations phase change threshold
    locations = levitator.preprocess_coordinates(paths)
    gorkov = torch.zeros((N, locations.shape[1]))

    for i in range(locations.shape[1]):
        p = locations[:, i].T[np.newaxis, :]
        A = forward_model_batched(p,TRANSDUCERS).to(torch.complex64)
        if i == 0:
            _,_,x = wgs_batch(A, torch.ones(N, 1, dtype=A.dtype).to(device)+0j, 5)
        else:
            _,_,x = temporal_wgs(A,torch.ones(N, 1, dtype=A.dtype).to(device)+0j, 5, ref_in, ref_out, T_in, T_out)
        y = A@x
        #print("Points' pressure?", torch.abs(y))
        # Update the reference phase
        ref_in = x
        ref_out = y

        # Add signature to hologram phase
        ph = torch.angle(x) + torch.cat((torch.zeros(256,1), math.pi*torch.ones(256,1)), axis=0)

        
        Ax2, Ay2, Az2 = levitator.surround_points(locations[:, i, :])
        Ax_sim = Ax2.to(torch.complex64)
        Ay_sim = Ay2.to(torch.complex64)
        Az_sim = Az2.to(torch.complex64)
        gorkov[:, i:i+1], _ , _ , _ , _  = levitator.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)

    return gorkov.T.numpy()


if __name__ == '__main__':
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    model_name = '20_19_98_99/planner_v2'
    num_file = 30
    file_name_0 = 'smoothed_path'
    file_name_1 = 'new_smoothed_path'

    # TWGS, iterations=5
    levitator = top_bottom_setup(n_particles, algorithm='TWGS', iterations=5)

    # TWGS Josh setup
    TRANSDUCERS = transducers()

    computation_time = []
    for n in range(10):
        print(f'\n-----------------------The paths {n}-----------------------')

        csv_file = os.path.join(global_model_dir_1, model_name, f'{file_name_0}_{str(n)}.csv')
        #csv_file = 'F:\Desktop\Projects\AcousticLevitationGym\examples\experiments\S2M2_8_experiments\data0.csv'
        csv_data = read_csv_file(csv_file)
        if csv_data == None:
            print(f"Skipping file due to read failure: {csv_file}")
            continue

        data_numpy, _ = read_paths(csv_data)

        # 每个粒子的轨迹长度相同
        paths_length = int(csv_data[1][1])
        # split_data_numpy的形状为(n_particles, n_keypoints, 5)
        # When axis=2: keypoints_id, time, x, y, z
        split_data = data_numpy.reshape(-1, paths_length, 5)

        # 计算时间变化量（差分）
        # split_data_numpy[:,:,1] 是时间累加值（时间列）
        delta_time = np.diff(split_data[0, :, 1], axis=0)

        velocities, accelerations = calculate_v_a(split_data[:, :, 2:])
        # 由于垂直方向上的F_max是水平方向的大约6倍，给xy方向的加速度6倍的权重以提高其影响
        accelerations[:, :, :2] *= 6.0
        sum_a = np.linalg.norm(accelerations, axis=2)
        print(sum_a.shape)

        # 在时间序列尾部添加一个累计时间以匹配速度序列的长度
        accumulative_time = split_data[0, :, 1]
        last_time = accumulative_time[-1] + 0.0032
        accumulative_time = np.append(accumulative_time, last_time)
        # 可视化速度
        # visualize_all_particles_v3(accumulative_time, sum_a)


        # Calculate Gorkov
        # 给paths末尾添加一个frame以匹配sum_a
        paths = np.concatenate((split_data[:, :, 2:], split_data[:, -1:, 2:]), axis=1)
        gorkov_1 = levitator.calculate_gorkov(paths)
        print(gorkov_1.T.shape)
        gorkov_2 = calculate_gorkov(paths, levitator)
        print(gorkov_2.T.shape)

        # 可视化速度
        visualize_all_particles_v2(accumulative_time, gorkov_1.T, gorkov_2.T)
 