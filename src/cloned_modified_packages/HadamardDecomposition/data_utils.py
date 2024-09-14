import numpy as np 


def makerealdata_hadamard_gaussian(n, r, m, loc, scale):
    A1 = np.random.normal(loc, scale, size=(n,r))
    B1 = np.random.normal(loc, scale, size=(r,m))
    D1 = A1 @ B1
    A2 = np.random.normal(loc, scale, size=(n,r))
    B2 = np.random.normal(loc, scale, size=(r,m))
    D2 = A2 @ B2
    return D1, D2 , D1 * D2 

def makerealdata_hadamard_uniform(n,r,m):
    A1 = np.random.uniform(size=(n,r))
    B1 = np.random.uniform(size=(r,m))
    D1 = A1 @ B1
    A2 = np.random.uniform(size=(n,r))
    B2 = np.random.uniform(size=(r,m))
    D2 = A2 @ B2
    return D1, D2 , D1 * D2 

def makerealdata_full_rank_uniform(n,m):
    D = np.random.uniform(0,1, size=(n,m))
    return D

def makerealdata_full_rank_gaussian(n,m, loc, scale):
    D = np.random.normal(loc, scale, size=(n,m))
    return D


