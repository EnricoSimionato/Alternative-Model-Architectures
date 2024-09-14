import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch 

class SGD_HadDec(nn.Module):
    def __init__(self, n,c,m):
        super(SGD_HadDec, self).__init__()
        self.A1 = nn.Parameter(torch.rand(n,c, requires_grad=True)) # parameter which we want to optimize
        self.B1 = nn.Parameter(torch.rand(c,m, requires_grad=True)) # parameter which we want to optimize
        self.A2 = nn.Parameter(torch.rand(n,c, requires_grad=True)) # parameter which we want to optimize
        self.B2 = nn.Parameter(torch.rand(c,m, requires_grad=True)) # parameter which we want to optimize


    def forward(self, start_index, end_index):
        res1 = (torch.matmul(self.A1[start_index:end_index],self.B1)) # prox_plus
        res2 = (torch.matmul(self.A2[start_index:end_index],self.B2)) # prox_plus
        return res1*res2 
    
        
def makerealdata(n,c,m,noise=0.0, zeroone=False):
    W = np.random.random_sample(size=(n,c))
    H = np.random.random_sample(size=(c,m))
    D = W@H
    return W,H,D

def mb_stochastic_gradient_descent_hadDec(D, r, lr=0.1, n_epochs=225000, batch_size=32): 
    ''' (mini-batch) stochastic gradient descent for Hadamard decomposition with two rank-r factors 
    
    Params:
    D: input matrix (numpy array) 
    r: rank of both Hadamard factors (integer) 
    eta: learning rate (float)
    n_epochs: maximum number of iterations/epochs (integer)
    batch_size: number of rows processed for each gradient computation 
    
    
    Returns: 
    D_estimate: nxm reconstruction of D 
    A_1, B_1, A_2, B_2: reconstruction of factors of each Hadamard factor of D 
    loss: L2 approximation error 
    ''' 
    
    loss_fn = nn.MSELoss(reduction="sum")
    n,m = D.shape 
    
    task = SGD_HadDec(n, r, m) 
    optimizer = optim.SGD(task.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        # Initialize indices of mini-batches per epoch 
        start_index = 0 
        end_index = 0
        # Iterate mini-batches 
        while end_index < D.shape[0]-1: 
            end_index = min(start_index + batch_size, D.shape[0]-1)
            D_estimate = task(start_index , end_index) # current approximation 
            loss = loss_fn(D_estimate, D[start_index:end_index])  # compare approximation with the data abd compute loss 
            task.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task.parameters(), max_norm=1) # clip (i.e., normalize) gradients to facilitate convergence 
            optimizer.step()
            start_index = end_index # update index 

    return D_estimate.detach().numpy(), [task.A_1.detach().numpy(), task.B_1.detach().numpy(), task.A_2.detach().numpy(), task.B_2.detach().numpy()], loss

