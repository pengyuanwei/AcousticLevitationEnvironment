import numpy as np
import torch
import math


class top_bottom_setup():
    def __init__(self, n_particles, algorithm='Naive'):
        '''
        algorithm: 'Naive', 'TWGS'.
        '''
        if algorithm not in ['Naive', 'TWGS']:
            raise ValueError(f"Invalid algorithm: {algorithm}. Choose 'Naive' or 'TWGS'.")
        self.n_particles = n_particles
        self.algorithm = algorithm
        
        # Setup gorkov
        self.l = 0.00865
        self.delta = self.l / 32
        self.density_0 = 1.2
        self.speed_0 = 343
        self.density_p = 1052
        self.speed_p = 1150
        self.radius = 0.001

        self.w = 2 * np.pi * (self.speed_0 / self.l)
        self.volume = 4 * np.pi * self.radius ** 3 / 3
        self.k1 = self.volume / 4 * (1 / (self.density_0 * self.speed_0 ** 2) - 1 / (self.density_p * self.speed_p ** 2))
        self.k2 = 3 * self.volume / 4 * (self.density_0 - self.density_p) / (self.w ** 2 * self.density_0 * (2 * self.density_p + self.density_0))

        self.transducer = torch.cat((self.create_board(17, -0.24 / 2), self.create_board(17, 0.24 / 2)), axis = 0)
        self.num_transducer = self.transducer.shape[0]
        self.m = n_particles
        b = torch.ones(self.m, 1) + 1j * torch.zeros(self.m, 1)
        self.b = b.to(torch.complex64)

        self.T_in = torch.pi/64  #Hologram phase change threshold
        self.T_out = 0  #Point activations phase change threshold


    def calculate_gorkov(self, key_points):
        algorithms = {
            'Naive': self.calculate_gorkov_with_wgs,
            'TWGS': self.calculate_gorkov_with_twgs
        }
        return algorithms[self.algorithm](key_points)


    def calculate_gorkov_wgs(self, key_points):
        gorkov_all_timesteps = np.zeros((key_points.shape[1], self.n_particles))
        transformed_coordinate = self.preprocess_coordinates(key_points)

        for i in range(key_points.shape[1]):
            points = transformed_coordinate[:, i, :]

            points1 = torch.tensor(points)
            Ax2, Ay2, Az2 = self.surround_points(points1)
            Ax2 = Ax2.to(torch.complex64)
            Ay2 = Ay2.to(torch.complex64)
            Az2 = Az2.to(torch.complex64)
            H = self.piston_model(points1).to(torch.complex64)
            gorkov = self.wgs_v1(H, Ax2, Ay2, Az2, self.b, self.num_transducer, 1)

            gorkov_numpy = gorkov.numpy()
            
            gorkov_numpy_transpose = gorkov_numpy.T

            gorkov_all_timesteps[i:i+1, :] = gorkov_numpy_transpose

        return gorkov_all_timesteps
    

    def calculate_gorkov_twgs(self, key_points):
        gorkov_all_timesteps = np.zeros((key_points.shape[1], self.n_particles))
        transformed_coordinate = self.preprocess_coordinates(key_points)

        for i in range(key_points.shape[1]):
            points = transformed_coordinate[:, i, :]
            points1 = torch.tensor(points)

            Ax2, Ay2, Az2 = self.surround_points(points1)
            Ax_sim = Ax2.to(torch.complex64)
            Ay_sim = Ay2.to(torch.complex64)
            Az_sim = Az2.to(torch.complex64)
            A = self.piston_model(points1).to(torch.complex64)

            if i == 0:
                x, y = self.wgs(A, self.b, 10)
            else:
                x, y = self.temporal_wgs(A, self.b, 10, self.ref_in, self.ref_out, self.T_in, self.T_out)
            self.ref_in = x
            self.ref_out = y

            ph = torch.angle(x) + torch.cat((torch.zeros(int(self.num_transducer/2),1), math.pi*torch.ones(int(self.num_transducer/2),1)), axis=0)
            gorkov, _ , _ , _ , _  = self.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)

            gorkov_numpy = gorkov.numpy()
            gorkov_numpy_transpose = gorkov_numpy.T
            gorkov_all_timesteps[i:i+1, :] = gorkov_numpy_transpose

        return gorkov_all_timesteps
    

    def preprocess_coordinates(self, key_points):
        transformed_coordinate = key_points.copy()
        transformed_coordinate[:, :, 2] -= 0.12
        return transformed_coordinate