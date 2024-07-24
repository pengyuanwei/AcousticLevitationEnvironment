import os
import csv
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def read_csv_file(file_path):
    if not os.path.exists(file_path):
        return None
    
    data_list = []
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data_list.append(row)
    return data_list


def extract_trajectories(n_particles, csv_data):
    max_length = np.zeros(n_particles)
    particle_index = 0
    csv_data_float = []
    for j in range(len(csv_data)):
        sub_data_list = []
        if csv_data[j] and len(csv_data[j]) == 5:
            sub_data_list = [float(element) for element in csv_data[j]]
            csv_data_float.append(sub_data_list)
            if sub_data_list[0] >= max_length[particle_index]:
                max_length[particle_index] = sub_data_list[0]
            else:
                particle_index += 1

    if np.max(max_length) == 0.0:
        raise ValueError("Max length of paths equal to zero!")
    
    return csv_data_float, max_length
        

if __name__ == "__main__":
    n_particles = 8

    for n in range(1):
        # read the start and end points
        csv_file = './experiments/experiment_20/paths/path' + str(n) + '.csv'
        csv_data = read_csv_file(csv_file)
        if not csv_data:
            continue

        csv_data_float, max_length = extract_trajectories(n_particles, csv_data)

        max_length_int = max_length.astype(int)
        max_length_int += 1
        print(max_length_int)

        data_numpy = np.array(csv_data_float)


        # When axis=1: id, time, x, y, z
        split_data_numpy = np.zeros((n_particles, 2, 5))

        for j in range(len(split_data_numpy)):
            split_data_numpy[j, :1, :] = data_numpy[:1]
            split_data_numpy[j, -1:, :] = data_numpy[(max_length_int[j]-1):max_length_int[j]]

            data_numpy = data_numpy[max_length_int[j]:]

        print(split_data_numpy)
        print(split_data_numpy.shape)