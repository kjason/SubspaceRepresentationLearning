"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen

Modified from https://github.com/kjason/DnnNormTimeFreq4DoA/tree/main/SpeechEnhancement
"""
from typing import List
import os
import torch
import numpy as np

def MRA(N:int, d:float):
    spacing_list = [[1],[1,2],[1,3,2],[1,3,3,2],[1,3,1,6,2],[1,3,6,2,3,2],[1,3,6,6,2,3,2],[1,3,6,6,6,2,3,2],[1,2,3,7,7,7,4,4,1],[1,2,3,7,7,7,7,4,4,1],[1,2,3,7,7,7,7,7,4,4,1]]
    if N < 2:
        raise ValueError("N should be larger than or equal to 2")
    elif N > 12:
        raise ValueError("only support N up to 12") 
    else:
        spacing = spacing_list[N-2]
        sensor_grid = [sum(spacing[:i]) for i in range(N)] # sensor locations on the grid of nonnegative integers
        N_a = sensor_grid[-1]
        sensor_locations = [(i-N_a/2)*d for i in sensor_grid] # sensor locations in the space
        return sensor_locations, sensor_grid, N_a

def dir_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def file_path(path):
    if os.path.isfile(path) or path == None:
        return path
    else:
        raise ValueError('{} is not a valid file'.format(path))

def check_device(device):
    if device == 'cpu':
        return device
    elif torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            if device == 'cuda:'+str(i):
                return device
        raise ValueError('{} not found in the available cuda or cpu list'.format(device))
    else:
        raise ValueError('{} is not a valid cuda or cpu device'.format(device))

def get_device_name(device):
    if device[:4] == 'cuda':
        return torch.cuda.get_device_name(int(device[-1])) # print the GPU
    else:
        return device

def data_in_preprocess(data_in: torch.Tensor):
    if len(data_in.shape) == 3:
        batch_size = data_in.size(0)
    elif len(data_in.shape) == 2:
        data_in = data_in.unsqueeze(0)
        batch_size = 1
    else:
        raise ValueError(f"len(data_in.shape)={len(data_in.shape)}, invalid data_in")
    return data_in, batch_size

def cov_normalize(cov: torch.Tensor, mode: str, N: int):
    if mode == 'max':
        n_cov = cov / torch.amax(torch.abs(cov),dim=[-2,-1],keepdim=True)
    elif mode == 'sensors':
        n_cov = cov / N
    elif mode == 'disabled':
        n_cov = cov
    else:
        raise ValueError(f'normalization={mode} is invalid')
    return n_cov

def ComplexMat2RealImagMat(cov: torch.Tensor):
    return torch.cat((cov.real.unsqueeze(1),cov.imag.unsqueeze(1)),1)

def RealImagMat2ComplexMat(cov: torch.Tensor):
    return torch.complex(cov[:,0,:,:],cov[:,1,:,:])

def RealImagMat2GramComplexMat(cov: torch.Tensor):
    c = RealImagMat2ComplexMat(cov)
    return torch.matmul(c,c.conj().transpose(-2,-1))

def HermitianMat2RealVec(cov: torch.Tensor):
    N = cov.size(-1)
    tri = torch.triu(torch.ones(N, N)) == 1
    otri = (torch.triu(torch.ones(N,N)) == 1).fill_diagonal_(False)
    def OneHMat2RealVec(c: torch.Tensor):
        return torch.cat((c.real[tri == 1],c.imag[otri == 1]),0)
    return torch.vmap(OneHMat2RealVec)(cov)

def RealVec2HermitianMat(vec: torch.Tensor):
    N = int(np.sqrt(vec.shape[1]))
    batch_size = vec.shape[0]
    H_real = torch.zeros(batch_size,N,N,dtype=vec.dtype,device=vec.device)
    H_imag = torch.zeros(batch_size,N,N,dtype=vec.dtype,device=vec.device)
    for i in range(N):
        j = int(i*N-i*(i-1)/2)
        H_real[:,i,i:] = vec[:,j:j+N-i]
        H_real[:,1+i:,i] = H_real[:,i,1+i:]
    k = (N+1)*N/2
    Nm = N - 1
    for i in range(Nm):
        j = int(k+i*Nm-i*(i-1)/2)
        H_imag[:,i,1+i:] = vec[:,j:j+Nm-i]
        H_imag[:,1+i:,i] = -H_imag[:,i,1+i:]
    H = torch.complex(H_real,H_imag)
    return H

def HermitianToeplitzMat2RealVec(cov: torch.Tensor):
    return torch.cat([cov[:,0,:].real,cov[:,0,1:].imag],1)

def RealVec2HermitianToeplitzMat(vec: torch.Tensor):
    N = int((vec.shape[-1]+1)/2)
    batch_size = vec.shape[0]
    T_real = torch.zeros(batch_size,N,N,device=vec.device)
    T_imag = torch.zeros(batch_size,N,N,device=vec.device)
    T_real[:,0,:] = vec[:,:N]
    T_real[:,1:,0] = vec[:,1:N]
    T_imag[:,0,1:] = vec[:,N:]
    T_imag[:,1:,0] = - vec[:,N:]
    for i in range(1,N):
        T_real[:,i,i:] = T_real[:,i-1,i-1:-1]
        T_real[:,i+1:,i] = T_real[:,i,i+1:]
        T_imag[:,i,i:] = T_imag[:,i-1,i-1:-1]
        T_imag[:,i+1:,i] = - T_imag[:,i,i+1:]
    T = torch.complex(T_real,T_imag)
    return T

if __name__ == '__main__':
    import time
    cov = torch.zeros(3,3,dtype=torch.complex64,device='cuda:0')
    cov[0,0] = 2
    cov[1,1] = 2
    cov[2,2] = 2
    cov[1,0] = 3+1*1j
    cov[2,1] = 3+1*1j
    cov[2,0] = 4+5*1j
    cov[0,1] = 3-1*1j
    cov[1,2] = 3-1*1j
    cov[0,2] = 4-5*1j
    print(cov)
    cov = cov.unsqueeze(0)
    cov = cov.repeat(7,1,1)
    tic = time.time()
    vec = HermitianToeplitzMat2RealVec(cov)
    cov_r = RealVec2HermitianToeplitzMat(vec)
    print(torch.all(torch.isclose(cov_r,cov)))
    err_part = torch.isclose(cov_r,cov) == False
    print(cov[err_part]-cov_r[err_part])
    toc = time.time()
    print(toc-tic)