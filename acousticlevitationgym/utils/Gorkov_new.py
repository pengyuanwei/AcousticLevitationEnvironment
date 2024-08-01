import numpy as np
import torch
import math


def create_board(N, z):  
    pitch=0.0105
    grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1))
    x, y = torch.meshgrid(grid_vec,grid_vec)
    trans_x=torch.reshape(x,(torch.numel(x),1))
    trans_y=torch.reshape(y,(torch.numel(y),1))
    trans_z=z*torch.ones((torch.numel(x),1))
    trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)
    return trans_pos


def create_board_left_right(N, y):  
    pitch=0.0105
    grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1))
    x, z = torch.meshgrid(grid_vec,grid_vec)
    trans_x=torch.reshape(x,(torch.numel(x),1))
    trans_z=torch.reshape(z,(torch.numel(z),1))
    trans_y=y*torch.ones((torch.numel(x),1))
    trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)
    return trans_pos


def piston_model_new(transducers, points):
    
    m = points.shape[0]
    n = transducers.shape[0]
    k=2*math.pi/0.00865
    radius=0.005
    transducers_x=torch.reshape(transducers[:,0],(n,1))
    transducers_y=torch.reshape(transducers[:,1],(n,1))
    transducers_z=torch.reshape(transducers[:,2],(n,1))
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


def forward_full_gorkov(ph,A,Ax,Ay,Az,k1,k2):
    x = torch.exp(1j*ph)
    p = torch.matmul(A,x)
    px = torch.matmul(Ax,x)
    py = torch.matmul(Ay,x)
    pz = torch.matmul(Az,x)
    U = k1*torch.abs(p)**2 + k2*torch.abs(px)**2 + k2*torch.abs(py)**2 + k2*torch.abs(pz)**2
    return U, p, px, py, pz


# When K = 1, the algorithm becomes Naive.
def wgs_new(A,Ax_sim,Ay_sim,Az_sim,b,n,k1,k2,K):
    AT = torch.conj(A).T
    b0 = b
    y = b

    for kk in range(K):
        
        x = torch.matmul(AT, y)
        x = torch.divide(x, torch.abs(x))      
        y = torch.matmul(A, x)
        y = y/torch.max(torch.abs(y))
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))
        y = torch.multiply(b,torch.divide(y,torch.abs(y)))
        
    ph = torch.angle(x) + torch.cat((torch.zeros(int(n/2),1),math.pi*torch.ones(int(n/2),1)),axis=0)
    Ur, _ , _ , _ , _  = forward_full_gorkov(ph,A,Ax_sim,Ay_sim,Az_sim,k1,k2)

    return Ur


def surround_points(transducer, points, delta):
    d = torch.zeros(1,3)
    d[0,0] = delta
    Ax = piston_model_new(transducer, points + d)
    A_x = piston_model_new(transducer, points - d)

    d = torch.zeros(1,3)
    d[0,1] = delta
    Ay = piston_model_new(transducer, points + d)
    A_y = piston_model_new(transducer, points - d)

    d = torch.zeros(1,3)
    d[0,2] = delta
    Az = piston_model_new(transducer, points + d)
    A_z = piston_model_new(transducer, points - d)

    Ax2 = (Ax - A_x)/(2*delta)
    Ay2 = (Ay - A_y)/(2*delta)
    Az2 = (Az - A_z)/(2*delta)
    
    return Ax2, Ay2, Az2


























"""
Function to find phases of transducers to create a control point using Naive.
Inputs:
   -A: piston model transmission matrix.
   -b: complex acoustic pressure at points.
Variables:
   -x: complex acoustic pressure at transducers.
   -y: complex acoustic pressure at points.
"""
def naive(A, b):
    AT = torch.conj(A).T
    x = torch.matmul(AT, b)
    x = torch.divide(x, torch.abs(x))
    y = torch.matmul(A, x)
    return (x, y)


"""
Function to calculate Gor'kov potential at one given location.
Inputs:
   -point: location of point in 3D.
   -transducers: location of transducers in 3D.
   -transducer_values: complex acoustic pressures of transducers.
Output:
   -Gor'kov potential at point
"""
def gorkov_potential(points, transducers, transducer_values):
    l=.00865
    k=2*np.pi/l

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
    potentials = []
    for point in points:
        point_mx = [point[0]-delta,point[1],point[2]]
        point_px = [point[0]+delta,point[1],point[2]]
        point_my = [point[0],point[1]-delta,point[2]]
        point_py = [point[0],point[1]+delta,point[2]]
        point_mz = [point[0],point[1],point[2]-delta]
        point_pz = [point[0],point[1],point[2]+delta]
        points = [point_mx, point_px, point_my, point_py, point_mz, point_pz, point]
        H = piston_model_new(transducers, points)
        H = H.to(torch.complex64)
        y = np.matmul(H,np.asarray(transducer_values))

        amplitude = abs(np.asarray(y))
        x_component = (amplitude[0] - amplitude[1])/(2*delta)
        y_component = (amplitude[2] - amplitude[3])/(2*delta)
        z_component = (amplitude[4] - amplitude[5])/(2*delta)

        potential = k1*amplitude[6]**2 - k2*(x_component**2 + y_component**2 + z_component**2)
        potentials.append(potential)
    return(potentials)


from acoustic_levitation_gym.robots.particle_slim import Particle as particle_slim
if __name__ == "__main__":

    # Setup PATs
    n_particles = 4
    particles = [particle_slim(0.04, 0.0, 0.0077+0.117, 0), 
                 particle_slim(-0.04, 0.0, 0.0077+0.117, 1),
                 particle_slim(0.04, 0.04, 0.0077+0.117, 2), 
                 particle_slim(-0.04, -0.04, 0.0077+0.117, 3)]

    points = [0]*n_particles
    gorkov = [0]*n_particles

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

    transducer = torch.cat((create_board(17,-.234/2), create_board(17,.234/2)),axis=0)
    num_transducer = transducer.shape[0]
    m = n_particles
    b = torch.ones(m,1) +1j*torch.zeros(m,1)
    b = b.to(torch.complex64)

    for i, particle in enumerate(particles):
        points[i] = [particle.x, particle.y, particle.z-0.0077-0.117]
    points1 = torch.tensor(points)

    # Calculate Gorkov at points: WGS
    Ax2, Ay2, Az2 = surround_points(transducer, points1, delta)
    Ax2 = Ax2.to(torch.complex64)
    Ay2 = Ay2.to(torch.complex64)
    Az2 = Az2.to(torch.complex64)
    H = piston_model_new(transducer, points1)
    H = H.to(torch.complex64)
    gorkov = wgs_new(H,Ax2,Ay2,Az2,b,num_transducer,k1,k2,50)

    print(gorkov)

    # Calculate Gorkov at points: Naive
    H = piston_model_new(transducer, points1)
    H = H.to(torch.complex64)
    transducer_field, points_field = naive(H, b)
    midpoint = transducer_field.shape[0] // 2
    transducer_field[:midpoint] *= -1

    points = torch.tensor(points1)
    transducers = torch.tensor(transducer)
    transducer_values = torch.tensor(transducer_field)

    potentials = []
    for point in points:
        point_mx = [point[0]-delta,point[1],point[2]]
        point_px = [point[0]+delta,point[1],point[2]]
        point_my = [point[0],point[1]-delta,point[2]]
        point_py = [point[0],point[1]+delta,point[2]]
        point_mz = [point[0],point[1],point[2]-delta]
        point_pz = [point[0],point[1],point[2]+delta]
        points = [point_mx, point_px, point_my, point_py, point_mz, point_pz, point]
        H = piston_model_new(transducers, points)
        H = H.to(torch.complex64)
        y = np.matmul(H,np.asarray(transducer_values))

        amplitude = abs(np.asarray(y))
        x_component = (amplitude[0] - amplitude[1])/(2*delta)
        y_component = (amplitude[2] - amplitude[3])/(2*delta)
        z_component = (amplitude[4] - amplitude[5])/(2*delta)

        potential = k1*amplitude[6]**2 - k2*(x_component**2 + y_component**2 + z_component**2)
        potentials.append(potential)

    print(potentials)

