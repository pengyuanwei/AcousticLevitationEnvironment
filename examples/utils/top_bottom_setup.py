import numpy as np
import torch
import math


class top_bottom_setup():
    def __init__(self, n_particles):
        self.n_particles = n_particles
        
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


    def wgs(self, A, Ax_sim, Ay_sim, Az_sim, b, n, K):
        '''
        Inputs:
            -A: piston model transmission matrix.
            -K: iteration number.
        Variables:
            -ph: phase hologram.
        '''
        AT = torch.conj(A).T
        b0 = b
        y = b

        # When K=1, the WGS degenerate to the Naive.
        for _ in range(K):
            x = torch.matmul(AT, y)
            x = torch.divide(x, torch.abs(x))      
            y = torch.matmul(A, x)
            y = y/torch.max(torch.abs(y))
            b = torch.multiply(b0, torch.divide(b,torch.abs(y)))
            y = torch.multiply(b, torch.divide(y,torch.abs(y)))
            
        ph = torch.angle(x) + torch.cat((torch.zeros(int(n/2),1),math.pi*torch.ones(int(n/2),1)),axis=0)
        Ur, _ , _ , _ , _  = self.forward_full_gorkov(ph, A, Ax_sim, Ay_sim, Az_sim)

        return Ur


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
        transformed_coordinate = key_points.copy()
        transformed_coordinate[:, :, 2] -= 0.12

        gorkov_all_timestep = np.zeros((key_points.shape[1], self.n_particles))

        for i in range(key_points.shape[1]):
            points = transformed_coordinate[:, i, :]

            points1 = torch.tensor(points)
            Ax2, Ay2, Az2 = self.surround_points(points1)
            Ax2 = Ax2.to(torch.complex64)
            Ay2 = Ay2.to(torch.complex64)
            Az2 = Az2.to(torch.complex64)
            H = self.piston_model(points1).to(torch.complex64)
            gorkov = self.wgs(H, Ax2, Ay2, Az2, self.b, self.num_transducer, 1)

            gorkov_numpy = gorkov.numpy()
            
            gorkov_numpy_transpose = gorkov_numpy.T

            gorkov_all_timestep[i:i+1, :] = gorkov_numpy_transpose

        return gorkov_all_timestep