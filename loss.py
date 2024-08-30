"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen

https://arxiv.org/abs/2408.16605
"""
import torch

def AngleMSE(outputs, target_cov, source_numbers, angles):
    rank = source_numbers[0]
    error = torch.sort(outputs[:,:rank])[0] - angles[:,:rank]
    return torch.mean(error ** 2, dim=1)

def OrderedAngleMSE(outputs, target_cov, source_numbers, angles):
    rank = source_numbers[0]
    error = outputs[:,:rank] - angles[:,:rank]
    return torch.mean(error ** 2, dim=1)

# loss function of the gridless end-to-end approach
def BranchAngleMSE(outputs, target_cov, source_numbers, angles):
    rank = source_numbers[0]
    error = torch.sort(outputs[rank-1])[0] - angles[:,:rank]
    return torch.mean(error ** 2, dim=1)

def BranchOrderedAngleMSE(outputs, target_cov, source_numbers, angles):
    rank = source_numbers[0]
    error = outputs[rank-1] - angles[:,:rank]
    return torch.mean(error ** 2, dim=1)

# loss function of DCR-T
def ToepSquare(outputs, targets, source_numbers, angles):
    first_row_err = outputs[:,0,:] - targets[:,0,:]
    return 0.5 * torch.mean(torch.abs(first_row_err * first_row_err.conj()), dim=1)

# loss function of DCR-G-Fro
def FrobeniusNorm(outputs, targets, source_numbers, angles):
    A = outputs - targets
    return torch.linalg.matrix_norm(A,'fro')

# subspace representation learning
def NoiseSubspaceDist(outputs, targets, source_numbers, angles):
    rank = source_numbers[0]
    m = targets.size(-1) - rank
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-m:]
    B = BQ[:,:,:-rank]
    _, S, _= torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return torch.sqrt(torch.sum(theta[:,:m] ** 2, dim=1))

# the main loss function of the subspace representation learning approach, see Section IV in the paper
def SignalSubspaceDist(outputs, targets, source_numbers, angles):
    rank = source_numbers[0]
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _= torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return torch.sqrt(torch.sum(theta[:,:rank] ** 2, dim=1))

# subspace representation learning
def AvgSubspaceDist(outputs, targets, source_numbers, angles):
    rank = source_numbers[0]
    m = targets.size(-1) - rank
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A_s = AQ[:,:,-rank:]
    B_s = BQ[:,:,-rank:]
    A_n = AQ[:,:,:-rank]
    B_n = BQ[:,:,:-rank]
    _, S_s, _= torch.linalg.svd(A_s.conj().transpose(-2,-1) @ B_s)
    theta_s = torch.acos(-torch.nn.functional.threshold(-S_s,-1,-1))
    _, S_n, _= torch.linalg.svd(A_n.conj().transpose(-2,-1) @ B_n)
    theta_n = torch.acos(-torch.nn.functional.threshold(-S_n,-1,-1))
    return 0.5 * torch.sqrt(torch.sum(theta_s[:,:rank] ** 2, dim=1)) + 0.5 * torch.sqrt(torch.sum(theta_n[:,:m] ** 2, dim=1))

# subspace representation learning
def L2SubspaceDist(outputs, targets, source_numbers, angles):
    rank = source_numbers[0]
    m = targets.size(-1) - rank
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A_s = AQ[:,:,-rank:]
    B_s = BQ[:,:,-rank:]
    A_n = AQ[:,:,:-rank]
    B_n = BQ[:,:,:-rank]
    _, S_s, _= torch.linalg.svd(A_s.conj().transpose(-2,-1) @ B_s)
    theta_s = torch.acos(-torch.nn.functional.threshold(-S_s,-1,-1))
    _, S_n, _= torch.linalg.svd(A_n.conj().transpose(-2,-1) @ B_n)
    theta_n = torch.acos(-torch.nn.functional.threshold(-S_n,-1,-1))
    return torch.sqrt(torch.sum(theta_s[:,:rank] ** 2, dim=1) + torch.sum(theta_n[:,:m] ** 2, dim=1))

def logm(A: torch.Tensor):
    lam, V = torch.linalg.eig(A)
    V_inv = torch.inverse(V)
    log_A_prime = torch.diag(lam.log())
    return V @ log_A_prime @ V_inv

def inv_sqrtmh(A): # modified from https://github.com/pytorch/pytorch/issues/25481
    """Compute sqrtm(inv(A)) where A is a symmetric or Hermitian PD matrix (or a batch of matrices)"""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero) # zero out small components
    return (Q * (1/L.sqrt().unsqueeze(-2))) @ Q.mH

# loss function of DCR-G-Aff
def AffInvDist(outputs, targets, source_numbers, angles):
    delta = 1e-4
    I = torch.eye(outputs.size(-1),device=outputs.device).unsqueeze(0)
    targets = targets + delta * I
    targets_inv_sqrt = inv_sqrtmh(targets)
    A = torch.vmap(logm)(targets_inv_sqrt @ outputs @ targets_inv_sqrt)
    return torch.linalg.matrix_norm(A,'fro')

# loss function of DCR-G-Aff for the 6-element MRA
def AffInvDist3(outputs, targets, source_numbers, angles): # delta is 1e-3
    delta = 1e-3
    I = torch.eye(outputs.size(-1),device=outputs.device).unsqueeze(0)
    targets = targets + delta * I
    targets_inv_sqrt = inv_sqrtmh(targets)
    A = torch.vmap(logm)(targets_inv_sqrt @ outputs @ targets_inv_sqrt)
    return torch.linalg.matrix_norm(A,'fro')

loss_dict = {
    'AngleMSE': AngleMSE,
    'OrderedAngleMSE': OrderedAngleMSE,
    'BranchAngleMSE': BranchAngleMSE,
    'BranchOrderedAngleMSE': BranchOrderedAngleMSE,
    'ToepSquare': ToepSquare,
    'FrobeniusNorm': FrobeniusNorm,
    'NoiseSubspaceDist': NoiseSubspaceDist,
    'SignalSubspaceDist': SignalSubspaceDist,
    'AvgSubspaceDist': AvgSubspaceDist,
    'L2SubspaceDist': L2SubspaceDist,
    'AffInvDist': AffInvDist,
    'AffInvDist3': AffInvDist3
}

def is_EnEnH(loss):
    if 'Noise' in loss:
        return True
    else:
        return False