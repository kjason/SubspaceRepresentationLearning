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

# subspace representation learning | Geodesic distance
def NoiseSubspaceDist(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    m = targets.size(-1) - rank
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-m:]
    B = BQ[:,:,:-rank]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return torch.sqrt(torch.sum(theta[:,:m] ** 2, dim=1))

# the main loss function of the subspace representation learning approach | Geodesic distance | see Section IV in the paper
def SignalSubspaceDist(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return torch.sqrt(torch.sum(theta[:,:rank] ** 2, dim=1))

# subspace representation learning | Geodesic distance | without consistent rank sampling | direct approach
def SignalSubspaceDistNoCrsDirect(outputs, targets, source_numbers, angles):
    batch_size = outputs.size(0)
    l = []
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    for i in range(batch_size):
        rank = source_numbers[i]
        A = AQ[:,:,-rank:]
        B = BQ[:,:,-rank:]
        _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
        theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
        l.append(torch.sqrt(torch.sum(theta[:,:rank] ** 2, dim=1)))
    return torch.cat(l,dim=0)

# subspace representation learning | Geodesic distance | without consistent rank sampling | grouping approach
def SignalSubspaceDistNoCrsGroup(outputs, targets, source_numbers, angles):
    max_n_src = max(source_numbers).item()
    l = []
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    for i in range(1,max_n_src+1):
        x = source_numbers == i
        if not True in x:
            continue
        rank = source_numbers[x][0]
        A = AQ[x,:,-rank:]
        B = BQ[x,:,-rank:]
        _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
        theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
        l.extend(torch.sqrt(torch.sum(theta[:,:rank] ** 2, dim=1)))
    l = [j.reshape(1) for j in l]
    return torch.cat(l,dim=0)

# subspace representation learning | Chordal distance (or projection Frobenius norm distance) using principal angles
def SignalChordalDistPA(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return torch.sqrt(torch.sum(torch.sin(theta[:,:rank]) ** 2, dim=1))

# subspace representation learning | Chordal distance (or projection Frobenius norm distance) using orthonormal bases
def SignalChordalDistOB(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    C = A @ A.conj().transpose(-2,-1) - B @ B.conj().transpose(-2,-1)
    return torch.linalg.matrix_norm(C,'fro') / torch.sqrt(torch.tensor(2))

# subspace representation learning | Projection 2-norm using principal angles
def SignalProjectionDistPA(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return torch.sin(theta[:,rank-1])

# subspace representation learning | Projection 2-norm using orthonormal bases
def SignalProjectionDistOB(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    C = A @ A.conj().transpose(-2,-1) - B @ B.conj().transpose(-2,-1)
    return torch.linalg.matrix_norm(C,2)

# subspace representation learning | Fubini-Study distance using principal angles
def SignalFubiniStudyDistPA(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    C = torch.prod(S[:,:rank],dim=1)
    return torch.acos(-torch.nn.functional.threshold(-C,-1,-1))

# subspace representation learning | Fubini-Study distance using orthonormal bases
def SignalFubiniStudyDistOB(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    C = A.conj().transpose(-2,-1) @ B
    D = torch.abs(torch.linalg.det(C))
    return torch.acos(-torch.nn.functional.threshold(-D,-1,-1))

# subspace representation learning | Procrustes distance (or chordal Frobenius norm distance) using principal angles
def SignalProcrustesDistPA(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return 2 * torch.sqrt(torch.sum(torch.sin(theta[:,:rank] / 2) ** 2, dim=1))

# subspace representation learning | Procrustes distance (or chordal Frobenius norm distance) using orthonormal bases
# def SignalProcrustesDistOB(outputs, targets, source_numbers, angles):
#     rank = source_numbers[0] # assume consistent rank sampling is enabled
#     _, AQ = torch.linalg.eigh(outputs)
#     _, BQ = torch.linalg.eigh(targets)
#     A = AQ[:,:,-rank:]
#     B = BQ[:,:,-rank:]
#     U, _, V = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
#     C = A @ U - B @ V
#     return torch.linalg.matrix_norm(C,'fro')

# subspace representation learning | Spectral distance (or chordal 2-norm distance) using principal angles
def SignalSpectralDistPA(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return 2 * torch.sin(theta[:,rank-1] / 2)

# subspace representation learning | Spectral distance (or chordal 2-norm distance) using orthonormal bases
# def SignalSpectralDistOB(outputs, targets, source_numbers, angles):
#     rank = source_numbers[0] # assume consistent rank sampling is enabled
#     _, AQ = torch.linalg.eigh(outputs)
#     _, BQ = torch.linalg.eigh(targets)
#     A = AQ[:,:,-rank:]
#     B = BQ[:,:,-rank:]
#     U, _, V = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
#     C = A @ U - B @ V
#     return torch.linalg.matrix_norm(C,2)

# subspace representation learning
def AvgSubspaceDist(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
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
    rank = source_numbers[0] # assume consistent rank sampling is enabled
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

# What if only phi_1 is minimized?
def SignalPhi1(outputs, targets, source_numbers, angles):
    rank = source_numbers[0] # assume consistent rank sampling is enabled
    _, AQ = torch.linalg.eigh(outputs)
    _, BQ = torch.linalg.eigh(targets)
    A = AQ[:,:,-rank:]
    B = BQ[:,:,-rank:]
    _, S, _ = torch.linalg.svd(A.conj().transpose(-2,-1) @ B)
    theta = torch.acos(-torch.nn.functional.threshold(-S,-1,-1))
    return theta[:,0]

def scale_invariant_targets(outputs, targets):
    targets_ri = torch.cat((targets.real.unsqueeze(-1),targets.imag.unsqueeze(-1)),-1)
    outputs_ri = torch.cat((outputs.real.unsqueeze(-1),outputs.imag.unsqueeze(-1)),-1)
    alphas = torch.sum(targets_ri * outputs_ri, dim=[-3,-2,-1], keepdim=True) / torch.sum(targets_ri * targets_ri, dim=[-3,-2,-1], keepdim=True) # this is a real number
    return alphas.squeeze(-1) * targets

def SignalSubspaceTargets(A, source_numbers):
    rank = source_numbers[0]
    _,Q = torch.linalg.eigh(A)
    return Q[:,:,-rank:] @ Q[:,:,-rank:].transpose(-2,-1).conj()

# ICASSP SI-Cov
def SISDRFrobeniusNorm(outputs, targets, source_numbers, angles):
    targets = scale_invariant_targets(outputs, targets)
    return - 10 * torch.log10(torch.linalg.matrix_norm(targets,'fro') / FrobeniusNorm(outputs, targets, source_numbers, None) )

# ICASSP SI-Sig
def SignalSISDRFrobeniusNorm(outputs, targets, source_numbers, angles):
    targets = SignalSubspaceTargets(targets,source_numbers)
    targets = scale_invariant_targets(outputs, targets)
    return - 10 * torch.log10(torch.linalg.matrix_norm(targets,'fro') / FrobeniusNorm(outputs, targets, source_numbers, None) )

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
    'AffInvDist3': AffInvDist3,
    'SignalChordalDistPA': SignalChordalDistPA,
    'SignalChordalDistOB': SignalChordalDistOB,
    'SignalProjectionDistPA': SignalProjectionDistPA,
    'SignalProjectionDistOB': SignalProjectionDistOB,
    'SignalFubiniStudyDistPA': SignalFubiniStudyDistPA,
    'SignalFubiniStudyDistOB': SignalFubiniStudyDistOB,
    'SignalProcrustesDistPA': SignalProcrustesDistPA,
    'SignalSpectralDistPA': SignalSpectralDistPA,
    'SignalSubspaceDistNoCrsDirect': SignalSubspaceDistNoCrsDirect,
    'SignalSubspaceDistNoCrsGroup': SignalSubspaceDistNoCrsGroup,
    'SignalPhi1': SignalPhi1,
    'SISDRFrobeniusNorm': SISDRFrobeniusNorm,
    'SignalSISDRFrobeniusNorm': SignalSISDRFrobeniusNorm
}

def is_EnEnH(loss):
    if 'Noise' in loss:
        return True
    else:
        return False