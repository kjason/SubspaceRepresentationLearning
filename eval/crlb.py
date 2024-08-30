"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen

Implementations are based on the following two papers

Wang, Mianzhi, Zhen Zhang, and Arye Nehorai. "Further results on the Cramér–Rao bound for sparse linear arrays." IEEE Transactions on Signal Processing 67, no. 6 (2019): 1493-1507.
Wang, Mianzhi, and Arye Nehorai. "Coarrays, MUSIC, and the Cramér–Rao bound." IEEE Transactions on Signal Processing 65, no. 4 (2016): 933-946.
"""
import numpy as np
import torch
from utils import MRA
from scipy import linalg

def get_steering_matrix(N_sensors, source_angles, d, lam):
    source_angles = np.expand_dims(source_angles,0)
    # MRA
    MRA_sensor_locations, _, _ = MRA(N_sensors,d)
    p = (2*np.pi/lam)*np.expand_dims(np.array(MRA_sensor_locations,dtype=np.longdouble),1)
    # steering matrix
    imag = p @ np.cos(source_angles,dtype=np.longdouble)
    V = np.exp(1j*imag,dtype=np.complex128)
    # dV / dtheta
    imag_p = -p @ np.sin(source_angles,dtype=np.longdouble)
    dV = 1j*imag_p*V
    return V, dV

# Wang, Mianzhi, Zhen Zhang, and Arye Nehorai. "Further results on the Cramér–Rao bound for sparse linear arrays." IEEE Transactions on Signal Processing 67, no. 6 (2019): 1493-1507.
def uncorrelated_CRLB(source_angles, N_sensors, d, lam, SNR, n_snapshots, total_power_one):
    num_sources = len(source_angles)
    if total_power_one is True:
        noise_var = (num_sources/(10**(SNR/10)))
    else:
        noise_var = (1/(10**(SNR/10)))
    k = num_sources
    m = N_sensors
    p = np.ones(num_sources,dtype=np.longdouble)

    V, dV = get_steering_matrix(N_sensors,source_angles,d,lam)

    A = V
    DA = dV
    
    A_H = A.conj().T
    DA_H = DA.conj().T
    P = np.diag(np.ones(num_sources,dtype=np.longdouble))
    R = (A * p) @ A_H + noise_var * np.eye(m)
    R_inv = np.linalg.inv(R.astype(np.complex128))
    R_inv = 0.5 * (R_inv + R_inv.conj().T)

    DRD = DA_H @ R_inv @ DA
    DRA = DA_H @ R_inv @ A
    ARD = A_H @ R_inv @ DA
    ARA = A_H @ R_inv @ A

    FIM_tt = 2.0 * (DRD.conj() * (P @ ARA @ P) + DRA.conj() * (P @ ARD @ P)).real
    FIM_pp = (ARA.conj() * ARA).real
    R_inv2 = R_inv @ R_inv
    FIM_ss = np.trace(R_inv2).real
    FIM_tp = 2.0 * (DRA.conj() * (p[:, np.newaxis] * ARA)).real
    FIM_ts = 2.0 * (p * np.sum(DA.conj() * (R_inv2 @ A), axis=0)).real[:, np.newaxis]
    FIM_ps = np.sum(A.conj() * (R_inv2 @ A), axis=0).real[:, np.newaxis]
    FIM = np.block([
        [FIM_tt,          FIM_tp,          FIM_ts],
        [FIM_tp.conj().T, FIM_pp,          FIM_ps],
        [FIM_ts.conj().T, FIM_ps.conj().T, FIM_ss]
    ])
    CRB = np.linalg.inv(FIM.astype(np.float64))[:k, :k] / n_snapshots
    return 0.5 * (CRB + CRB.T)

# Wang, Mianzhi, and Arye Nehorai. "Coarrays, MUSIC, and the Cramér–Rao bound." IEEE Transactions on Signal Processing 65, no. 4 (2016): 933-946.
def uncorrelated_CRLB2(source_angles, N_sensors, d, lam, SNR, n_snapshots, total_power_one):
    num_sources = len(source_angles)
    if total_power_one is True:
        noise_var = (num_sources/(10**(SNR/10)))
    else:
        noise_var = (1/(10**(SNR/10)))
    k = num_sources
    m = N_sensors
    P = np.diag(np.ones(num_sources,dtype=np.longdouble))
    I = np.eye(N_sensors,dtype=np.longdouble)
    
    V, dV = get_steering_matrix(N_sensors,source_angles,d,lam)

    A = V
    DA = dV

    R = A @ P @ A.conj().T + noise_var * I
    i = np.expand_dims(I.flatten('F'),axis=1)
    A_d = linalg.khatri_rao(A.conj(),A)
    A_d_dot = linalg.khatri_rao(DA.conj(),A) + linalg.khatri_rao(A.conj(),DA)

    RtR_sqrt = np.linalg.inv(linalg.sqrtm(np.kron(R.T,R)).astype(np.complex128))
    M_theta = RtR_sqrt @ A_d_dot @ P

    M_s = RtR_sqrt @ np.concatenate((A_d,i),axis=1)

    M_sHM_s = M_s.conj().T @ M_s
    M_sHM_s_inv = np.linalg.inv(M_sHM_s.astype(np.complex128))
    M_sHM_s_inv = 0.5 * (M_sHM_s_inv + M_sHM_s_inv.conj().T)

    M_s_r = M_s @ M_sHM_s_inv @ M_s.conj().T
    P_M_s = np.eye(M_s_r.shape[0]) - M_s_r

    FIM = M_theta.conj().T @ P_M_s @ M_theta

    CRB = np.linalg.inv(FIM.astype(np.complex128)) / n_snapshots
    return 0.5 * (CRB + CRB.T).real

if __name__ == '__main__':
    from data import random_source_angles
    import matplotlib.pyplot as plt
    np.random.seed(0)
    torch.manual_seed(0)
    # source DoA
    d = 0.01
    lam = 0.02
    N_sensors = 5
    SNR = -10
    deg_range = [30,150]
    # (N_sensors,min_sep) (4,18) (5,12 or 13) (6,9)
    num_sources = 9
    min_sep = 13
    # the two CRLB implementations above will match if the minimum separation is sufficiently large
    total_power_one = False
    print_results = True
    n_snapshots = 50
    M = 10000
    j = 0
    tr_crb_vec = np.zeros(M)
    for i in range(M):
        source_angles = random_source_angles(deg_range,min_sep,num_sources) * np.pi / 180
        crb = uncorrelated_CRLB(source_angles=source_angles,N_sensors=N_sensors, d=d, lam=lam,SNR=SNR,n_snapshots=n_snapshots,total_power_one=total_power_one)
        crb2 = uncorrelated_CRLB2(source_angles=source_angles,N_sensors=N_sensors, d=d, lam=lam,SNR=SNR,n_snapshots=n_snapshots,total_power_one=total_power_one)
        tr_crb = np.mean(np.diag(crb))
        tr_crb_vec[i] = tr_crb
        tr_crb2 = np.mean(np.diag(crb2))
        crb_diag = np.diag(crb)
        crb2_diag = np.diag(crb2)
        diff_crb_max_ratio = np.max(np.abs(crb_diag-crb2_diag))/np.max(np.abs(crb_diag))
        is_less_than_0 = np.prod(crb_diag >= 0)
        if is_less_than_0 == 0:
            print("less than 0")
        if diff_crb_max_ratio > 0.02:
            j += 1
            if print_results is True:
                print(diff_crb_max_ratio)
                print(source_angles)
                print(crb_diag)
                print(crb2_diag)
                print(tr_crb)
                print(tr_crb2)
                print('-'*10)
    print(f'N_sensors={N_sensors}')
    print(f'mean tr_crb={np.mean(tr_crb_vec)} rad^2 ( RMSE= {np.sqrt(np.mean(tr_crb_vec))*180/np.pi} deg ) | {np.log10(np.mean(tr_crb_vec))}')
    print(f' min tr_crb={np.min(tr_crb_vec)} rad^2 ( RMSE= {np.sqrt(np.min(tr_crb_vec))*180/np.pi} deg ) | {np.log10(np.min(tr_crb_vec))}')
    print(f' max tr_crb={np.max(tr_crb_vec)} rad^2 ( RMSE= {np.sqrt(np.max(tr_crb_vec))*180/np.pi} deg ) | {np.log10(np.max(tr_crb_vec))}')
    print(f'Number of disagreements: {j}/{M}')
    crb_hist,bin_edges = np.histogram(tr_crb_vec,bins=5000)
    plt.semilogx(bin_edges[:-1],crb_hist)
    plt.grid()
    plt.xlabel('MSE (rad^2)')
    plt.ylabel('Probability density')
    plt.tight_layout()
    plt.show()