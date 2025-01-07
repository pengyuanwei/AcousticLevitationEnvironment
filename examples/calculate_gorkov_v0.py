import csv
import numpy as np
import math
import torch
import os
import matplotlib.pyplot as plt

from acousticlevitationenvironment.utils import Gorkov_new

# Modified based on the DataTransfer_v1.py
# Calculate Gorkov for S2M2 generated trajectories: find the min(|G|) at each timestep.


def calculate_gorkov(split_data_numpy, n_particles, transducer, delta, b, num_transducer, k1, k2):

    transformed_coordinate = split_data_numpy.copy()
    transformed_coordinate[:, :, 4] -= 0.12

    gorkov_all_timestep = np.zeros((split_data_numpy.shape[1], n_particles))

    for i in range(split_data_numpy.shape[1]):
        points = np.zeros((n_particles, 3))
        for j in range(n_particles):
            points[j] = [transformed_coordinate[j][i][2], transformed_coordinate[j][i][3], transformed_coordinate[j][i][4]]

        points1 = torch.tensor(points)
        Ax2, Ay2, Az2 = Gorkov_new.surround_points(transducer, points1, delta)
        Ax2 = Ax2.to(torch.complex64)
        Ay2 = Ay2.to(torch.complex64)
        Az2 = Az2.to(torch.complex64)
        H = Gorkov_new.piston_model_new(transducer, points1)
        H = H.to(torch.complex64)
        gorkov = Gorkov_new.wgs_new(H, Ax2, Ay2, Az2, b, num_transducer, k1, k2, 1)

        gorkov_numpy = gorkov.numpy()
        
        gorkov_numpy_transpose = gorkov_numpy.T
        gorkov_numpy_exp_dims = np.expand_dims(gorkov_numpy, axis=1)

        gorkov_all_timestep[i:i+1, :] = gorkov_numpy_transpose
        split_data_numpy[:, i:i+1, 5:6] = gorkov_numpy_exp_dims

    return gorkov_all_timestep, split_data_numpy


def read_specific_row(csv_file, row_number):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == row_number:
                return row


def read_csv_file(file_path):
    data_list = []
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data_list.append(row)
    return data_list


if __name__ == '__main__':
    n_particles = 6

    # Setup gorkov
    l=.00865
    delta = l/32
    density_0=1.2
    speed_0=343
    density_p=1052
    speed_p=1150
    radius=.001

    w=2*np.pi*(speed_0/l)
    volume=4*np.pi*radius**3/3
    k1=volume/4*(1/(density_0*speed_0**2)-1/(density_p*speed_p**2))
    k2=3*volume/4*(density_0-density_p)/(w**2*density_0*(2*density_p+density_0))

    transducer = torch.cat((Gorkov_new.create_board(17,-.24/2),Gorkov_new.create_board(17,.24/2)),axis=0)
    num_transducer = transducer.shape[0]
    m = n_particles
    b = torch.ones(m,1) +1j*torch.zeros(m,1)
    b = b.to(torch.complex64)

    for n in range(10):
        # S2M2 data
        csv_file = 'trainingData/data' + str(n) + '.csv'
        csv_data = read_csv_file(csv_file)

        max_length = np.zeros(n_particles)
        which_particle = 0

        csv_data_float = []
        for j in range(len(csv_data)):
            sub_data_list = []
            if csv_data[j] and len(csv_data[j]) == 5:
                sub_data_list = [float(element) for element in csv_data[j]]
                csv_data_float.append(sub_data_list)
                if sub_data_list[0] >= max_length[which_particle]:
                    max_length[which_particle] = sub_data_list[0]
                else:
                    which_particle += 1

        if np.max(max_length) == 0.0:
            continue

        max_length_int = max_length.astype(int)
        max_length_int += 1
        print(max_length_int)

        data_numpy = np.array(csv_data_float)


        # When axis=1: id, time, x, y, z, gorkov
        split_data_numpy = np.zeros((n_particles, np.max(max_length_int), 6))
        split_data_numpy[:, :, 5:6] -= 1.0

        for j in range(len(split_data_numpy)):
            split_data_numpy[j, :max_length_int[j], :5] = data_numpy[:max_length_int[j]]

            if max_length_int[j] < np.max(max_length_int):
                last_particle_position = data_numpy[max_length_int[j]-1]
                split_data_numpy[j, -(np.max(max_length_int)-max_length_int[j]):, :5] = last_particle_position

            data_numpy = data_numpy[max_length_int[j]:]

        #print(split_data_numpy[0])
        #print(split_data_numpy.shape)            

        
        # Calculate the Gorkov
        gorkov, split_data_with_gorkov = calculate_gorkov(split_data_numpy, n_particles, transducer, delta, b, num_transducer, k1, k2)

        print(split_data_with_gorkov[0][0])
        print(split_data_with_gorkov[1][0])
        print(split_data_with_gorkov[2][0])
        print(split_data_with_gorkov[3][0])
        print(split_data_with_gorkov.shape)

        print(gorkov[0])
        print(gorkov.shape)

        # Find the maximum value of Gorkov at each timestep
        max_values1 = np.max(gorkov, axis=1)
        print(max_values1[0])
        print(max_values1.shape)
        print(type(max_values1))


        # RL model data ####################################################################################################################
        csv_file = 'Experiments/Experiment_364/path' + str(n) + '.csv'
        csv_data = read_csv_file(csv_file)

        max_length = np.zeros(n_particles)
        which_particle = 0

        csv_data_float = []
        for j in range(len(csv_data)):
            sub_data_list = []
            if csv_data[j] and len(csv_data[j]) == 5:
                sub_data_list = [float(element) for element in csv_data[j]]
                csv_data_float.append(sub_data_list)
                if sub_data_list[0] >= max_length[which_particle]:
                    max_length[which_particle] = sub_data_list[0]
                else:
                    which_particle += 1

        if np.max(max_length) == 0.0:
            continue

        max_length_int = max_length.astype(int)
        max_length_int += 1
        print(max_length_int)

        data_numpy = np.array(csv_data_float)


        # When axis=1: id, time, x, y, z, gorkov
        split_data_numpy = np.zeros((n_particles, np.max(max_length_int), 6))
        split_data_numpy[:, :, 5:6] -= 1.0

        for j in range(len(split_data_numpy)):
            split_data_numpy[j, :max_length_int[j], :5] = data_numpy[:max_length_int[j]]

            if max_length_int[j] < np.max(max_length_int):
                last_particle_position = data_numpy[max_length_int[j]-1]
                split_data_numpy[j, -(np.max(max_length_int)-max_length_int[j]):, :5] = last_particle_position

            data_numpy = data_numpy[max_length_int[j]:]

        #print(split_data_numpy[0])
        #print(split_data_numpy.shape)            

        
        # Calculate the Gorkov
        gorkov, split_data_with_gorkov = calculate_gorkov(split_data_numpy, n_particles, transducer, delta, b, num_transducer, k1, k2)

        print(split_data_with_gorkov[0][0])
        print(split_data_with_gorkov[1][0])
        print(split_data_with_gorkov[2][0])
        print(split_data_with_gorkov[3][0])
        print(split_data_with_gorkov.shape)

        print(gorkov[0])
        print(gorkov.shape)

        # Find the maximum value of Gorkov at each timestep
        max_values2 = np.max(gorkov, axis=1)
        print(max_values2[0])
        print(max_values2.shape)
        print(type(max_values2))


        # 获取新数组的长度，作为x值 ####################################################################################################################
        x_values1 = np.arange(len(max_values1))
        x_values2 = np.arange(len(max_values2))
        x_values_2 = x_values2 * np.sqrt(3)

        print(x_values2)
        print(range(0, len(x_values2), 10))

        # 绘制曲线图
        plt.plot(x_values1, max_values1, label='S2M2')
        plt.plot(x_values_2, max_values2, label='RL Model')
        plt.legend()
        plt.xlabel('Timesteps (1/240s)')
        plt.ylabel('Max Gorkov (Pa)')
        plt.title('The No.{} start-end pairs'.format(n))
        plt.grid(True)

        #plt.xticks(range(0, len(x_values2), 10))

        plt.show()