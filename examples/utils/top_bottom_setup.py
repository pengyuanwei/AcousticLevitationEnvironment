import numpy as np
import torch
import math


class top_bottom_setup():
    def __init__(self, n_particles, algorithm='Naive', iterations=1):
        '''
        algorithm: 'Naive', 'TWGS'.
        '''
        if algorithm not in ['Naive', 'TWGS']:
            raise ValueError(f"Invalid algorithm: {algorithm}. Choose 'Naive' or 'TWGS'.")
        self.n_particles = n_particles
        self.algorithm = algorithm
        self.iterations = iterations
        
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

        # TWGS parameters
        self.T_in = torch.pi/32  #Hologram phase change threshold
        self.T_out = torch.pi/32  #Point activations phase change threshold


    def create_board(self, N, z):  
        pitch=0.0105
        grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1))
        x, y = torch.meshgrid(grid_vec,grid_vec)
        trans_x=torch.reshape(x,(torch.numel(x),1))
        trans_y=torch.reshape(y,(torch.numel(y),1))
        trans_z=z*torch.ones((torch.numel(x),1))
        trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)

        return trans_pos


    def forward_full_gorkov(self, ph, A, Ax, Ay, Az):
        x = torch.exp(1j*ph)
        p = torch.matmul(A,x)
        px = torch.matmul(Ax,x)
        py = torch.matmul(Ay,x)
        pz = torch.matmul(Az,x)
        U = self.k1*torch.abs(p)**2 + self.k2*torch.abs(px)**2 + self.k2*torch.abs(py)**2 + self.k2*torch.abs(pz)**2

        return U, p, px, py, pz


    # def wgs_v1(self, A, Ax_sim, Ay_sim, Az_sim, b, n, K):
    #     '''
    #     Old version
    #     Inputs:
    #         -A: piston model transmission matrix (forward model matrix).
    #         -K: iteration number.
    #     Variables:
    #         -ph: phase hologram.
    #     '''
    #     AT = torch.conj(A).T
    #     b0 = b  # target amplitudes
    #     y = b  # initial guess - normally use `torch.ones(N,1) + 0j`

    #     # When K=1, the WGS degenerate to the Naive.
    #     for _ in range(K):
    #         x = torch.matmul(AT, y)
    #         x = torch.divide(x, torch.abs(x))      
    #         y = torch.matmul(A, x)
    #         y = y/torch.max(torch.abs(y))
    #         b = torch.multiply(b0, torch.divide(b,torch.abs(y)))
    #         y = torch.multiply(b, torch.divide(y,torch.abs(y)))
            
    #     ph = torch.angle(x) + torch.cat((torch.zeros(int(n/2),1),math.pi*torch.ones(int(n/2),1)),axis=0)
    #     Ur, _ , _ , _ , _  = self.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)

    #     return Ur
    

    def wgs(self, A, y0, K):
        '''
        New version
        `A` Forward model matrix to use \\ 
        `y0` initial guess - normally use `torch.ones(N,1).to(device)+0j`\\
        `K` number of iterations to run for \\
        returns (hologram x, field y)
        '''
        #Written by Giorgos Christopoulos 2022
        AT = torch.conj(A).T
        b0 = torch.ones(A.shape[0],1) + 0j  # target amplitudes
        y = y0
        x = torch.ones(A.shape[1],1) + 0j

        # When K=1, the WGS degenerate to the Naive.
        for _ in range(K):
            x = torch.matmul(AT,y)                                 
            x = torch.divide(x,torch.abs(x))                          
            z = torch.matmul(A,x) 
            y = b0*z*torch.abs(y)/(torch.abs(z)**2) # this same as the code you have, update target, impose this amplitude, and keep phase
            y = y/torch.max(torch.abs(y))                           
            
        # here I don't return phase with signature as previous implementation
        # compute outside the algorithm A, Ax, Ay, Az, add the signature to torch.angle(x) and compute Gor'kov
        return x, y


    def temporal_wgs(self, A, y0, K, ref_in, ref_out, T_in, T_out):
        '''
        `A` Forward model matrix to use \\ 
        `y0` initial guess - comes from previous frame, for first frame use WGS above and in the first for temporal input the returned y\\
        `K` number of iterations to run for \\
        `ref_in` previous hologram (transducers) x \\
        `ref_out` previous field y \\
        `T_in` transducer threshold (use pi/64 or pi/32) \\
        `T_out` point threshold (use 0 or pi/64 or pi/32) \\
        returns (hologram x, field y)
        '''    
        AT = torch.conj(A).T
        b0 = torch.ones(A.shape[0],1) + 0j  # target amplitudes
        y = y0
        x = torch.ones(A.shape[1],1) + 0j

        for _ in range(K):
            x = torch.matmul(AT,y)
            x = torch.divide(x,torch.abs(x))   
            x = self.ph_thresh(ref_in,x,T_in)              # clip transducer phase change

            z = torch.matmul(A,x) 
            y = b0*z*torch.abs(y)/(torch.abs(z)**2)
            y = y/torch.max(torch.abs(y))  
            y = self.ph_thresh(ref_out,y,T_out)            # clip point phase change
            
        # here I don't return phase with signature as previous implementation
        # compute outside the algorithm A, Ax, Ay, Az, add the signature to torch.angle(x) and compute Gor'kov
        return x, y


    def ph_thresh(self, z_last,z,threshold):
        '''
        Phase threshhold between two timesteps point phases, clamps phase changes above `threshold` to be `threshold`\\
        `z_last` point activation at timestep t-1\\
        `z` point activation at timestep t\\
        `threshold` maximum allowed phase change\\
        returns constrained point activations
        '''

        ph1 = torch.angle(z_last)
        ph2 = torch.angle(z)
        dph = ph2 - ph1
        
        # dph[dph>math.pi] = dph[dph>math.pi] - 2*math.pi
        # dph[dph<-1*math.pi] = dph[dph<-1*math.pi] + 2*math.pi    
        dph = torch.atan2(torch.sin(dph),torch.cos(dph)) 
        
        dph[dph>threshold] = threshold
        dph[dph<-1*threshold] = -1*threshold
        
        ph2 = ph1 + dph
        z = abs(z)*torch.exp(1j*ph2)
        
        return z


    def piston_model(self, points):
        m = points.shape[0]
        n = self.transducer.shape[0]
        k=2*math.pi/0.00865
        radius=0.005
        transducers_x=torch.reshape(self.transducer[:,0],(n,1))
        transducers_y=torch.reshape(self.transducer[:,1],(n,1))
        transducers_z=torch.reshape(self.transducer[:,2],(n,1))
        points_x=torch.reshape(points[:,0],(m,1))
        points_y=torch.reshape(points[:,1],(m,1))
        points_z=torch.reshape(points[:,2],(m,1))

        distance=torch.sqrt((transducers_x.T-points_x)**2+(transducers_y.T-points_y)**2+(transducers_z.T-points_z)**2)
        planar_distance=torch.sqrt((transducers_x.T-points_x)**2+(transducers_y.T-points_y)**2)
        bessel_arg=k*radius*torch.divide(planar_distance,distance)
        directivity=1/2-bessel_arg**2/16+bessel_arg**4/384-bessel_arg**6/18432+bessel_arg**8/1474560-bessel_arg**10/176947200
        phase=torch.exp(1j*k*distance)
        trans_matrix=2*8.02*torch.multiply(torch.divide(phase,distance),directivity)

        return trans_matrix
    

    def surround_points(self, points):
        d = torch.zeros(1,3)
        d[0,0] = self.delta
        Ax = self.piston_model(points + d)
        A_x = self.piston_model(points - d)

        d = torch.zeros(1,3)
        d[0,1] = self.delta
        Ay = self.piston_model(points + d)
        A_y = self.piston_model(points - d)

        d = torch.zeros(1,3)
        d[0,2] = self.delta
        Az = self.piston_model(points + d)
        A_z = self.piston_model(points - d)

        Ax2 = (Ax - A_x)/(2*self.delta)
        Ay2 = (Ay - A_y)/(2*self.delta)
        Az2 = (Az - A_z)/(2*self.delta)
        
        return Ax2, Ay2, Az2
    

    def calculate_gorkov(self, key_points):
        '''
        key_points: numpy array, (num_particles, path_lengths, 3)
        '''
        algorithms = {
            'Naive': self.calculate_gorkov_wgs,
            'TWGS': self.calculate_gorkov_twgs
        }
        return algorithms[self.algorithm](key_points)


    def calculate_gorkov_transposed(self, key_points):
        '''
        key_points: numpy array, (path_lengths, num_particles, 3)
        '''
        algorithms = {
            'Naive': self.calculate_gorkov_wgs_transposed,
            'TWGS': self.calculate_gorkov_twgs
        }
        return algorithms[self.algorithm](key_points)
    

    def calculate_gorkov_single_state(self, key_points):
        '''
        key_points: numpy array, (num_particles, 3)
        '''
        algorithms = {
            'Naive': self.calculate_gorkov_wgs_single_state,
            'TWGS': self.calculate_gorkov_twgs
        }
        return algorithms[self.algorithm](key_points)
    

    def calculate_gorkov_wgs(self, key_points):
        '''
        input:
            - key_points: numpy array, (num_particles, path_lengths, 3)
        output:
            - gorkov.T.numpy(): (path_lengths, num_particles)
        '''
        gorkov = torch.zeros((key_points.shape[0], key_points.shape[1]))
        locations = self.preprocess_coordinates(key_points)

        for i in range(key_points.shape[1]):
            A = self.piston_model(locations[:, i, :]).to(torch.complex64)
            x, _ = self.wgs(A, self.b, self.iterations)
            # Add signature to hologram phase
            ph = torch.angle(x) + torch.cat((torch.zeros(int(self.num_transducer/2),1), math.pi*torch.ones(int(self.num_transducer/2),1)), axis=0)

            Ax_sim, Ay_sim, Az_sim = self.surround_points(locations[:, i, :])
            Ax_sim = Ax_sim.to(torch.complex64)
            Ay_sim = Ay_sim.to(torch.complex64)
            Az_sim = Az_sim.to(torch.complex64)
            gorkov[:, i:i+1], _ , _ , _ , _  = self.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)
            
        return gorkov.T.numpy()
    

    def calculate_gorkov_wgs_transposed(self, key_points):
        '''
        key_points: numpy array, (path_lengths, num_particles, 3)
        output: numpy array, (num_particles, path_lengths)
        '''
        gorkov = torch.zeros((self.n_particles, key_points.shape[0]))
        locations = self.preprocess_coordinates(key_points)

        for i in range(key_points.shape[0]):
            A = self.piston_model(locations[i, :, :]).to(torch.complex64)
            x, _ = self.wgs(A, self.b, self.iterations)
            # Add signature to hologram phase
            ph = torch.angle(x) + torch.cat((torch.zeros(int(self.num_transducer/2),1), math.pi*torch.ones(int(self.num_transducer/2),1)), axis=0)

            Ax_sim, Ay_sim, Az_sim = self.surround_points(locations[i, :, :])
            Ax_sim = Ax_sim.to(torch.complex64)
            Ay_sim = Ay_sim.to(torch.complex64)
            Az_sim = Az_sim.to(torch.complex64)
            gorkov[:, i:i+1], _ , _ , _ , _  = self.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)
            
        return gorkov.T.numpy()
    

    def calculate_gorkov_wgs_single_state(self, key_points):
        '''
        key_points: numpy array, (num_particles, 3)
        output: numpy array, (num_particles, 1)
        '''
        gorkov = torch.zeros((self.n_particles, 1))
        locations = self.preprocess_coordinates_single_state(key_points)

        A = self.piston_model(locations).to(torch.complex64)
        x, _ = self.wgs(A, self.b, self.iterations)
        # Add signature to hologram phase
        ph = torch.angle(x) + torch.cat((torch.zeros(int(self.num_transducer/2),1), math.pi*torch.ones(int(self.num_transducer/2),1)), axis=0)

        Ax_sim, Ay_sim, Az_sim = self.surround_points(locations)
        Ax_sim = Ax_sim.to(torch.complex64)
        Ay_sim = Ay_sim.to(torch.complex64)
        Az_sim = Az_sim.to(torch.complex64)
        gorkov, _ , _ , _ , _  = self.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)
            
        return gorkov.numpy()
    

    def calculate_gorkov_twgs(self, key_points):
        '''
        key_points: numpy array, (num_particles, path_lengths, 3)
        '''
        gorkov = torch.zeros((self.n_particles, key_points.shape[1]))
        locations = self.preprocess_coordinates(key_points)

        for i in range(key_points.shape[1]):
            A = self.piston_model(locations[:, i, :]).to(torch.complex64)
            if i == 0:
                x, y = self.wgs(A, self.b, self.iterations)
            else:
                # temporal_wgs()可能存在一些问题：算出来的Gorkov值为正值。已检查wgs()没有问题。
                x, y = self.temporal_wgs(A, self.b, self.iterations, self.ref_in, self.ref_out, self.T_in, self.T_out)
            # Update the reference phase
            self.ref_in = x
            self.ref_out = y
            # Add signature to hologram phase
            ph = torch.angle(x) + torch.cat((torch.zeros(int(self.num_transducer/2),1), math.pi*torch.ones(int(self.num_transducer/2),1)), axis=0)

            Ax2, Ay2, Az2 = self.surround_points(locations[:, i, :])
            Ax_sim = Ax2.to(torch.complex64)
            Ay_sim = Ay2.to(torch.complex64)
            Az_sim = Az2.to(torch.complex64)
            gorkov[:, i:i+1], _ , _ , _ , _  = self.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)

        return gorkov.T.numpy()
    

    def preprocess_coordinates(self, key_points):
        transformed_coordinate = key_points.copy()
        transformed_coordinate[:, :, 2] -= 0.12
        points = torch.tensor(transformed_coordinate)
        return points
    

    def preprocess_coordinates_single_state(self, key_points):
        transformed_coordinate = key_points.copy()
        transformed_coordinate[:, 2] -= 0.12
        points = torch.tensor(transformed_coordinate)
        return points