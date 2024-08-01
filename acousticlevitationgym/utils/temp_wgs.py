import torch
import math


def wgs(A, y0, K):
    '''
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
    for kk in range(K):
        x = torch.matmul(AT,y)                                 
        x = torch.divide(x,torch.abs(x))                          
        z = torch.matmul(A,x) 
        y = b0*z*torch.abs(y)/(torch.abs(z)**2) # this same as the code you have, update target, impose this amplitude, and keep phase
        y = y/torch.max(torch.abs(y))                           
        
    # here I don't return phase with signature as in your current implementation
    # compute outside the algorithm A, Ax, Ay, Az, add the signature to torch.angle(x) and compute Gor'kov
    return x, y


def temporal_wgs(A, y0, K, ref_in, ref_out,T_in,T_out):
    '''
    `A` Forward model matrix to use \\ 
    `y0` initial guess - comes from previous frame, for first frame use WGS above and in the first for temporal input the returned y\\
    `K` number of iterations to run for \\
    `ref_in` previous hologram (transducers) x \\
    `ref_out` previous field y \\
    `T_in` transducer threshold (use pi/32) \\
    `T_out` point threshold (use 0 or pi/32) \\
    returns (hologram x, field y)
    '''    
    b0 = torch.ones(A.shape[0],1) + 0j  # target amplitudes
    AT = torch.conj(A).T
    y = y0
    x = torch.ones(A.shape[1],1) + 0j
    for kk in range(K):
        x = torch.matmul(AT,y)
        x = torch.divide(x,torch.abs(x))   
        x = ph_thresh(ref_in,x,T_in)              # clip transducer phase change
        z = torch.matmul(A,x) 
        y = b0*z*torch.abs(y)/(torch.abs(z)**2)
        y = y/torch.max(torch.abs(y))  
        y = ph_thresh(ref_out,y,T_out)            # clip point phase change
        
    # here I don't return phase with signature as in your current implementation
    # compute outside the algorithm A, Ax, Ay, Az, add the signature to torch.angle(x) and compute Gor'kov
    return x, y


def ph_thresh(z_last,z,threshold):
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
    
    dph[dph>math.pi] = dph[dph>math.pi] - 2*math.pi
    dph[dph<-1*math.pi] = dph[dph<-1*math.pi] + 2*math.pi    
#     dph = torch.atan2(torch.sin(dph),torch.cos(dph)) 
    
    dph[dph>threshold] = threshold
    dph[dph<-1*threshold] = -1*threshold
    
    ph2 = ph1 + dph
    z = abs(z)*torch.exp(1j*ph2)
    
    return z
        