import csv
import numpy as np
import math
import torch
import time

import temp_wgs
import utilities
from scipy.stats import norm
import matplotlib.pyplot as plt

from acoustools.Mesh import load_scatterer
from acoustools.Utilities import TOP_BOARD, create_points
from acoustools.Solvers import wgs
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical, propagate_BEM, get_cache_or_compute_H
import acoustools.Constants as c

# Modified based on the calculate_gorkov_v0.py
# Calculate Gorkov for S2M2 generated trajectories: save all gorkov values
# Using BEM


def calculate_gorkov_BEM(split_data_numpy, n_particles, reflector, H):

    transformed_coordinate = split_data_numpy.copy()
    transformed_coordinate[:, :, 4] -= 0.06

    gorkov_all_timestep = np.zeros((split_data_numpy.shape[1], n_particles))

    for i in range(split_data_numpy.shape[1]):
        points = np.zeros((n_particles, 3))
        for j in range(n_particles):
            points[j] = [transformed_coordinate[j][i][2], transformed_coordinate[j][i][3], transformed_coordinate[j][i][4]]

        points1 = torch.tensor(points).T.unsqueeze(0)
        #print(points1.shape)

        E = compute_E(reflector,points=points1,board=TOP_BOARD,path=r"F:\Desktop\siggraphAsia2024\acousticFunctions\BEMMedia",H=H)
        x = wgs(points1,iter=5,A=E)
        
        trap_up = points1
        trap_up[:,2] += c.wavelength/4
        gorkov = BEM_gorkov_analytical(x,trap_up,reflector,TOP_BOARD,path=r"F:\Desktop\siggraphAsia2024\acousticFunctions\BEMMedia",H=H)

        gorkov_numpy = gorkov.squeeze(0).numpy()
        
        gorkov_numpy_transpose = gorkov_numpy.T
        gorkov_numpy_exp_dims = np.expand_dims(gorkov_numpy, axis=1)

        gorkov_all_timestep[i:i+1, :] = gorkov_numpy_transpose
        split_data_numpy[:, i:i+1, 5:6] = gorkov_numpy_exp_dims

    return gorkov_all_timestep, split_data_numpy


def calculate_dist(split_data_numpy, n_particles):
    max_length = split_data_numpy.shape[1]
    max_length -= 1

    # Distance with targets
    for i in range(n_particles):
        for j in range(split_data_numpy.shape[1]):
            split_data_numpy[i][j][10] = math.sqrt((split_data_numpy[i][j][2] - split_data_numpy[i][max_length][2])**2 
                                                 + (split_data_numpy[i][j][3] - split_data_numpy[i][max_length][3])**2 
                                                 + (split_data_numpy[i][j][4] - split_data_numpy[i][max_length][4])**2)
    
    # Distance with other particles
    for i in range(n_particles):
        for j in range(n_particles):
            if j != i:
                dist = np.sqrt((split_data_numpy[i, :, 2] - split_data_numpy[j, :, 2])**2/0.014**2 
                             + (split_data_numpy[i, :, 3] - split_data_numpy[j, :, 3])**2/0.014**2
                             + (split_data_numpy[i, :, 4] - split_data_numpy[j, :, 4])**2/0.03**2)
                
                for k in range(split_data_numpy.shape[1]):
                    if dist[k] < split_data_numpy[i][k][11]:
                        split_data_numpy[i][k][11] = dist[k]
            
    return split_data_numpy


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
    model = '4particles'
    n_particles = 4

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

    transducer = torch.cat((utilities.create_board(17,-.24/2),utilities.create_board(17,.24/2)),axis=0)
    num_transducer = transducer.shape[0]
    m = n_particles
    y0 = torch.ones(m,1) +1j*torch.zeros(m,1)
    y0 = y0.to(torch.complex64)

    K = 5
    T_in = math.pi/32
    T_out = math.pi/32

    path = r"F:\Desktop\siggraphAsia2024\acousticFunctions\BEMMedia\flat-lam2.stl"
    reflector = load_scatterer(path,dz=-0.06)
    H = get_cache_or_compute_H(reflector, TOP_BOARD, path=r"F:\Desktop\siggraphAsia2024\acousticFunctions\BEMMedia")


    gorkov_list = []
    for n in range(200):
        #start_time = time.time()

        include_NaN = False

        #csv_file = model + '/path' + str(n) + '.csv'
        csv_file = 'trainingData/data' + str(n) + '.csv'

        csv_data = read_csv_file(csv_file)

        max_length = np.zeros(n_particles)
        which_particle = 0

        csv_data_float = []
        for j in range(len(csv_data)):
            sub_data_list = []
            if csv_data[j] and len(csv_data[j]) == 5:
                # 检测是否为NaN值
                if any(value == '-nan(ind)' or math.isnan(float(value)) for value in csv_data[j]):
                    include_NaN = True
                    break
                if include_NaN == True:
                    break
                sub_data_list = [float(element) for element in csv_data[j]]
                csv_data_float.append(sub_data_list)
                if sub_data_list[0] >= max_length[which_particle]:
                    max_length[which_particle] = sub_data_list[0]
                else:
                    which_particle += 1

        if np.max(max_length) == 0.0 or include_NaN == True:
            continue

        max_length_int = max_length.astype(int)
        max_length_int += 1
        #print(max_length_int)

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
        gorkov, _ = calculate_gorkov_BEM(split_data_numpy, n_particles, reflector, H)

        #print(gorkov[0])
        #print(gorkov.shape)

        gorkov_1d = gorkov.flatten()
        gorkov_list.append(gorkov_1d)

        print(gorkov_1d.shape)
        print(n/20, '%')

        #end_time = time.time()
        #execution_time = end_time - start_time
        #print(execution_time)

    gorkov_concatenated = np.concatenate(gorkov_list)
    print(gorkov_concatenated.shape)

    np.save('gorkovDistribution_'+model+'_BEM.npy', gorkov_concatenated)
    #np.save('gorkov_S2M2_4_new.npy', gorkov_concatenated)

    '''
    # 使用 norm.fit() 拟合数据集的高斯分布
    mu, sigma = norm.fit(gorkov_concatenated)

    # 绘制数据直方图和拟合的高斯分布曲线
    plt.hist(gorkov_concatenated, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2)
    formatted_mu = "{:.2e}".format(mu)
    formatted_sigma = "{:.2e}".format(sigma)
    #title = "Fit results: mu = %.10f,  sigma = %.10f" % (mu, sigma)
    plt.title("Fit results: mu =" + formatted_mu + "sigma =" + formatted_sigma)
    plt.show()'''