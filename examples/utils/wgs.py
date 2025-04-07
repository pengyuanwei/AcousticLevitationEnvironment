import torch


def wgs_batch(A, b, iterations):
    '''
    batched WGS solver for transducer phases\\
    `A` Forward model matrix to use \\ 
    `b` initial guess - normally use `torch.ones(self.N,1).to(device)+0j`\\
    `iterations` number of iterations to run for \\
    returns (hologram image, point phases, hologram)
    '''
    AT = torch.conj(A).mT.to(device)
    b0 = b.to(device)
    x = torch.ones(A.shape[2],1).to(device) + 0j

    for kk in range(iterations):
        y = torch.matmul(A,x)                                   # forward propagate
        y = y/torch.max(torch.abs(y))                           # normalize forward propagated field (useful for next step's division)
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))     # update target - current target over normalized field
        b = b/torch.max(torch.abs(b))                           # normalize target
        p = torch.multiply(b,torch.divide(y,torch.abs(y)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram  
                    
    return y, p, x


def wgs_v1(A, b, iterations):
    '''
    Input: 
        - A: propogation matrix based on the piston model
        - b: initial guess of target points' complex pressure
        - iterations: iteration number
    Output:
        - x: complex pressure of transducers
        - y: complex pressure of target points
    '''
    AT = torch.conj(A).T
    b0 = b
    x = torch.ones(A.shape[1], 1) + 0j

    for _ in range(iterations):
        y = torch.matmul(A, x)
        y = y/torch.max(torch.abs(y))
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))
        y = torch.multiply(b,torch.divide(y,torch.abs(y)))
        x = torch.matmul(AT, y)
        x = torch.divide(x, torch.abs(x))      

    return x, y


def wgs_v2(A, b, iterations):
    '''
    Input: 
        - A: propogation matrix based on the piston model
        - b: initial guess of target points' complex pressure
        - iterations: iteration number
    Output:
        - x: complex pressure of transducers
        - y: complex pressure of target points
    '''
    AT = torch.conj(A).T
    b0 = b  # record initial guess
    x = torch.ones(A.shape[1], 1) + 0j

    for _ in range(iterations):
        z = torch.matmul(A,x) 
        y = b0*z*torch.abs(y)/(torch.abs(z)**2)
        y = y/torch.max(torch.abs(y))                           
        x = torch.matmul(AT,y)                                 
        x = torch.divide(x,torch.abs(x))                          
        
    return x, y


def wgs_v3(A, b, iterations):
    '''
    Input: 
        - A: propogation matrix based on the piston model
        - b: initial guess of target points' complex pressure
        - iterations: iteration number
    Output:
        - x: complex pressure of transducers
        - y: complex pressure of target points
    '''
    AT = torch.conj(A).T
    b0 = b
    x = torch.ones(A.shape[1],1) + 0j

    for _ in range(iterations):
        y = torch.matmul(A,x)                                   # forward propagate
        y = y/torch.max(torch.abs(y))                           # normalize forward propagated field (useful for next step's division)
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))     # update target - current target over normalized field
        b = b/torch.max(torch.abs(b))                           # normalize target
        p = torch.multiply(b,torch.divide(y,torch.abs(y)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram  
                    
    return x, y


def wgs_v4(A, b, iterations):
    '''
    Input: 
        - A: propogation matrix based on the piston model
        - b: initial guess of target points' complex pressure?
        - iterations: iteration number
    Output:
        - x: complex pressure of transducers
        - y: complex pressure of target points
    '''
    AT = torch.conj(A).T
    b0 = b  # record initial guess
    x = torch.ones(A.shape[1], 1) + 0j

    for _ in range(iterations):
        # Forward
        z = torch.matmul(A,x) 
        y = b0*z*torch.abs(y)/(torch.abs(z)**2)
        # Normalise
        y = y/torch.max(torch.abs(y))  
        # Backward                         
        x = torch.matmul(AT,y)     
        # Normalise                            
        x = torch.divide(x,torch.abs(x))                          

    return x, y