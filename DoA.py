"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen
"""
import math
import numpy as np
from numpy.polynomial import Polynomial
import torch
from utils import MRA, data_in_preprocess

def coarray_and_weight_function(cov,sensor_grid):
    """
    Create the co-array from the sample covariance matrix and the sensor grid

    :param cov: the N-by-N complex covariance matrix
    :param sensor_grid: an N-element array of nonnegative integers representing the location of each sensor in the linear array
    :return: Co-array and the corresponding weight function
    """
    N = len(sensor_grid)
    if N != cov.size(0):
        raise ValueError("cov does not match with the sensor_grid")
    N_a = sensor_grid[-1]
    N_coarray = 2*N_a+1
    coarray = torch.zeros(N_coarray,dtype=torch.cfloat) # from -N_a to N_a
    weight_fn = torch.zeros(N_coarray,dtype=torch.cfloat) # from -N_a to N_a
    for i in range(N):
        for j in range(N):
            diff = sensor_grid[i]-sensor_grid[j]
            coarray[diff+N_a] += cov[i,j]
            weight_fn[diff+N_a] += 1
    return coarray, weight_fn

def direct_augmentation(cov: torch.Tensor):
    _, MRA_sensor_grid, N_a = MRA(cov.size(0),1)
    ULA_M_sensor = N_a+1
    ULA_sensor_grid = [(i-N_a/2) for i in range(N_a+1)]
    coarray, weight_fn = coarray_and_weight_function(cov,MRA_sensor_grid)
    r = coarray / weight_fn
    aug_cov = torch.zeros(ULA_M_sensor,ULA_M_sensor,dtype=torch.cfloat)
    for i in range(ULA_M_sensor):
        aug_cov[:,-i-1] = r[i:i+ULA_M_sensor]
    return aug_cov, ULA_sensor_grid

def spatial_smoothing(cov: torch.Tensor):
    aug_cov, ULA_sensor_grid = direct_augmentation(cov)
    return (1/len(ULA_sensor_grid))*torch.matmul(aug_cov,aug_cov.conj().transpose(0,1)), ULA_sensor_grid

def SRP_LA(Y,cov,lam,sensor_locations,N_gridpoints):
    grid = [i/(N_gridpoints-1) for i in range(N_gridpoints)]
    p = []
    for t in grid:
        imag = torch.tensor(sensor_locations).unsqueeze(1)*2*torch.pi*(1/lam)*torch.cos(torch.tensor(t)*torch.pi)
        v = (1/math.sqrt(len(sensor_locations)))*torch.exp(torch.complex(torch.zeros_like(imag),imag))
        if cov is None:
            a = torch.abs(torch.matmul(v.conj().transpose(0,1),Y))**2
        else:
            a = torch.abs(torch.matmul(torch.matmul(v.conj().transpose(0,1),cov),v))
        p.append(torch.sum(a))
    p = torch.tensor(p)
    return p/torch.max(p), grid

def MUSIC_LA(Y,cov,lam,sensor_locations,N_gridpoints,num_sources):
    eps = 1e-8
    if cov is None:
        U,_,_ = torch.linalg.svd(Y)
        E_n = U[:,num_sources:]
    else:
        L, Q = torch.linalg.eigh(cov)
        E_n = Q[:,:-num_sources]
    grid = [i/(N_gridpoints-1) for i in range(N_gridpoints)]
    p = []
    for t in grid:
        imag = torch.tensor(sensor_locations).unsqueeze(1)*2*torch.pi*(1/lam)*torch.cos(torch.tensor(t)*torch.pi)
        v = (1/math.sqrt(len(sensor_locations)))*torch.exp(torch.complex(torch.zeros_like(imag),imag))
        p.append(1/(torch.sum(torch.abs(torch.matmul(v.conj().transpose(0,1),E_n))**2)+eps))
    p = torch.tensor(p)
    return p/torch.max(p), grid

def RootMUSIC_ULA(Y,cov,num_sources,EnEnH):
    if cov is None:
        U,_,_ = torch.linalg.svd(Y)
        E_n = U[:,num_sources:]
    elif EnEnH is False:
        _, Q = np.linalg.eigh(cov.numpy())
        E_n = Q[:,:-num_sources]
    else:
        _, Q = np.linalg.eigh(cov.numpy())
        E_n = Q[:,num_sources:]
    N = E_n.shape[0]
    M = E_n.shape[1]
    tmp = torch.zeros(2*N-1,M,dtype=torch.cfloat).numpy()
    for i in range(M):
        tmp[:,i] = np.convolve(E_n[:,i],np.flip(E_n[:,i].conj()))
    coeff = np.sum(tmp,axis=1)
    r = Polynomial(coeff[::-1]).roots()
    rmin = r[np.abs(r)<=1]
    order = np.argsort(-np.abs(rmin))
    signalroot = rmin[order[:num_sources]]
    DoAs = np.sort(np.arccos(np.angle(signalroot)/np.pi))
    remaining_num_src = num_sources - DoAs.shape[0]
    success = not remaining_num_src > 0
    if not success:
        #print(f"Number of DoAs found is not equal to num_sources, we will guess the remaining sources are located at pi/2 rad or 90 deg (remaining_num_src={remaining_num_src})")
        DoAs = np.sort(np.concatenate((DoAs,np.array([np.pi/2]*remaining_num_src))))
    return DoAs, success

def RootMUSIC_ULA_2(Y,cov,num_sources,EnEnH):
    if cov is None:
        U,_,_ = torch.linalg.svd(Y)
        E_n = U[:,num_sources:].numpy()
    elif EnEnH is False:
        _, Q = np.linalg.eigh(cov.numpy())
        E_n = Q[:,:-num_sources]
    else:
        _, Q = np.linalg.eigh(cov.numpy())
        E_n = Q[:,num_sources:]
    # 1
    N = E_n.shape[0]
    M = E_n.shape[1]
    tmp = torch.zeros(2*N-1,M,dtype=torch.cfloat).numpy()
    for i in range(M):
        tmp[:,i] = np.convolve(E_n[:,i],np.flip(E_n[:,i].conj()))
    coeff = np.sum(tmp,axis=1)
    # 2
    #m = E_n.shape[0]
    #C = E_n @ E_n.T.conj()
    #coeff = np.zeros((m - 1,), dtype=np.complex_)
    #for i in range(1, m):
    #    coeff[i - 1] = np.sum(np.diag(C, i))
    #coeff = np.hstack((coeff[::-1], np.sum(np.diag(C)), coeff.conj()))

    z = Polynomial(coeff[::-1]).roots()
    # the root finding procedure below is borrowed from https://github.com/morriswmz/doatools.py/blob/master/doatools/estimation/music.py
    nz = len(z)
    mask = np.ones((nz,), dtype=np.bool_)
    for i in range(nz):
        absz = abs(z[i])
        if absz > 1.0:
            mask[i] = False
        elif absz == 1.0:
            idx = -1
            dist = np.inf
            for j in range(nz):
                if j != i:
                    cur_dist = abs(z[i] - z[j])
                    if cur_dist < dist:
                        dist = cur_dist
                        idx = j
            if idx < 0:
                raise RuntimeError('Unpaired point found on the unit circle, which is impossible.')
            if mask[idx] is True and mask[i] is True:
                mask[idx] = False
    z = z[mask]
    sorted_indices = np.argsort(-np.abs(z))
    z = z[sorted_indices[:num_sources]]
    DoAs = np.sort(np.arccos(np.angle(z)/np.pi))
    remaining_num_src = num_sources - DoAs.shape[0]
    success = not remaining_num_src > 0
    if not success:
        #print(f"Number of DoAs found is not equal to num_sources, we will guess the remaining sources are located at pi/2 rad or 90 deg (remaining_num_src={remaining_num_src})")
        DoAs = np.sort(np.concatenate((DoAs,np.array([np.pi/2]*remaining_num_src))))
    return DoAs, success

class BasePredictor:
    EnEnH = False
    need_snapshot = False
    use_noise_var = False
    def _get_one_ULA_cov(self, cov: torch.Tensor):
        return NotImplemented

    def get_ULA_cov(self, data_in: torch.Tensor, is_snapshot: bool, noise_var: torch.Tensor = None):
        data_in, batch_size = data_in_preprocess(data_in)
        if is_snapshot is False and self.need_snapshot is True:
            raise ValueError(f"given covariance matrices but the predictor actually needs snapshots")
        if is_snapshot is True and self.need_snapshot is False:
            T_snapshots = data_in.shape[-1]
            data = (1/T_snapshots)*torch.matmul(data_in,data_in.conj().transpose(-2,-1))
        else:
            data = data_in
        output_cov = []
        for b in range(batch_size):
            if self.use_noise_var is True:
                if noise_var is None:
                    raise ValueError("Please provide the noise_var because self.use_noise_var is True")
                out = self._get_one_ULA_cov(data[b,:,:],noise_var[b])
            else:
                out = self._get_one_ULA_cov(data[b,:,:])
            if isinstance(out,np.ndarray):
                out = torch.from_numpy(out)
            out, _ = data_in_preprocess(out)
            output_cov.append(out)
        output_cov = torch.cat(output_cov,0)
        return output_cov

    def get_DoA_by_rootMUSIC(self, data_in: torch.Tensor, num_sources: int, is_snapshot: bool, noise_var: torch.Tensor = None):
        if self.use_noise_var is True:
            cov = self.get_ULA_cov(data_in,is_snapshot,noise_var)
        else:
            cov = self.get_ULA_cov(data_in,is_snapshot)
        batch_size = cov.size(0)
        DoA_list = []
        success_list = []
        for b in range(batch_size):
            DoAs, success = RootMUSIC_ULA_2(None,cov[b,:,:],num_sources,self.EnEnH)
            DoA_list.append(torch.from_numpy(DoAs).unsqueeze(0))
            success_list.append(success)
        DoA = torch.cat(DoA_list,0)
        return DoA, success_list

    def get_DoA(self, data_in: torch.Tensor, num_sources: int, is_snapshot: bool, noise_var: torch.Tensor = None):
        return self.get_DoA_by_rootMUSIC(data_in, num_sources, is_snapshot, noise_var)

class CovMRA2ULA_DA(BasePredictor):
    def _get_one_ULA_cov(self,cov: torch.Tensor):
        return direct_augmentation(cov)[0]
    
class CovMRA2ULA_SS(BasePredictor):
    def _get_one_ULA_cov(self,cov: torch.Tensor):
        return spatial_smoothing(cov)[0]