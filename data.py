"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen
"""
from datetime import datetime
from typing import List
import numpy as np
import scipy.linalg as la
import h5py
import os
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import MRA, cov_normalize, dir_path

def sample(a: float, b: float, min_sep: float):
    s = np.random.uniform(a,b)
    (l_a, l_b) = (a, s - min_sep) if s - min_sep > a else (None,None)
    (r_a, r_b) = (s + min_sep, b) if s + min_sep < b else (None,None)
    if l_a is None and r_a is None:
        return [s]
    elif l_a is None:
        return [s] + sample(r_a, r_b, min_sep)
    elif r_a is None:
        return [s] + sample(l_a, l_b, min_sep)
    else:
        return [s] + sample(l_a, l_b, min_sep) + sample(r_a, r_b, min_sep)

def random_source_angles(deg_range: List[float], min_sep: float, num_sources: int):
    candidates = sample(deg_range[0],deg_range[1],min_sep)
    while len(candidates) < num_sources:
        #print(f"candidates ({len(candidates)}) < num_sources ({num_sources}), resample")
        candidates = sample(deg_range[0],deg_range[1],min_sep)
    return np.random.permutation(np.random.choice(a=candidates, size=num_sources, replace=False).astype(np.float32))

class ArrayManifold:
    @torch.no_grad()
    def __init__(self, d: float, lam: float, N_sensors: int,gain_bias: List[float], phase_bias_deg: List[float],
                 position_bias: List[float],mc_mag_angle: List[float], device: str):
        self.d = d
        self.lam = lam
        self.N_sensors = N_sensors
        self.device = device
        # MRA and ULA
        MRA_sensor_locations, sensor_grid, N_a = MRA(N_sensors,d)
        ULA_sensor_locations = [(i-N_a/2)*d for i in range(N_a+1)]
        self.sensor_grid = sensor_grid
        self.ULA_M_sensors = len(ULA_sensor_locations)
        self.MRA_sensor_locations = torch.tensor(MRA_sensor_locations,device=device)
        self.ULA_sensor_locations = torch.tensor(ULA_sensor_locations,device=device)
        # imperfections
        if len(mc_mag_angle) != 2:
            raise ValueError("invalid mc_mag_angle, mc_mag_angle[0] is the magnitude and mc_mag_angle[1] is phase in degrees")
        if len(gain_bias) != self.ULA_M_sensors or len(phase_bias_deg) != self.ULA_M_sensors or len(position_bias) != self.ULA_M_sensors:
            raise ValueError("invalid gain_bias, phase_bias_deg, or position_bias, their length must be equal to M")
        self.gain_bias = torch.tensor(gain_bias,device=device,dtype=torch.complex64)
        self.phase_bias = torch.tensor(phase_bias_deg,device=device,dtype=torch.float32) * np.pi/180
        self.position_bias = torch.tensor(position_bias,device=device,dtype=torch.float32) * d
        gamma = mc_mag_angle[0]*np.exp(1j*mc_mag_angle[1]*np.pi/180)
        ula_gamma_vec = gamma ** np.arange(self.ULA_M_sensors)
        ula_gamma_vec[0] = 0
        self.ula_mcm = torch.from_numpy(la.toeplitz(ula_gamma_vec)).type(torch.complex64).to(device)
        mra_gamma_vec = ula_gamma_vec[self.sensor_grid]
        self.mra_mcm = torch.from_numpy(la.toeplitz(mra_gamma_vec)).type(torch.complex64).to(device)

    @torch.no_grad()
    def get_V(self, rho: float, source_angles: torch.Tensor, mix: bool, mode: str):
        # MRA_sensor_locations is of size N
        # source_angles is of size L x 1 x # of sources
        # V is of size L x N x # of sources
        if mode == 'MRA':
            if rho == 0:
                imag = 2*torch.pi*(1/self.lam)*torch.matmul(self.MRA_sensor_locations.unsqueeze(1).unsqueeze(0),torch.cos(source_angles))
                V = torch.exp(torch.complex(torch.zeros_like(imag),imag))
            else:
                if mix is True:
                    rho = rho * torch.rand(source_angles.shape[0],1,dtype=torch.float32)
                else:
                    rho = rho * torch.ones(source_angles.shape[0],1,dtype=torch.float32)
                e_gain = 1.0 + rho.type(torch.complex64) @ self.gain_bias[self.sensor_grid].unsqueeze(0)
                e_phase = torch.exp(1j * (rho @ self.phase_bias[self.sensor_grid].unsqueeze(0)))
                e_pos = rho @ self.position_bias[self.sensor_grid].unsqueeze(0)
                E_mc = torch.eye(self.N_sensors,dtype=torch.complex64,device=self.device).unsqueeze(0) + rho.type(torch.complex64).unsqueeze(2) * self.mra_mcm.unsqueeze(0)
                MRA_sensor_locations_e = self.MRA_sensor_locations.unsqueeze(0) + e_pos
                imag = 2*torch.pi*(1/self.lam)*torch.matmul(MRA_sensor_locations_e.unsqueeze(2),torch.cos(source_angles))
                temp = e_gain.unsqueeze(2) * e_phase.unsqueeze(2) * torch.exp(torch.complex(torch.zeros_like(imag),imag))
                V = torch.matmul(E_mc,temp)
        elif mode == 'ULA':
            if rho == 0:
                imag = 2*torch.pi*(1/self.lam)*torch.matmul(self.ULA_sensor_locations.unsqueeze(1).unsqueeze(0),torch.cos(source_angles))
                V = torch.exp(torch.complex(torch.zeros_like(imag),imag))
            else:
                if mix is True:
                    rho = rho * torch.rand(source_angles.shape[0],1,dtype=torch.float32)
                else:
                    rho = rho * torch.ones(source_angles.shape[0],1,dtype=torch.float32)
                e_gain = 1.0 + rho.type(torch.complex64) @ self.gain_bias.unsqueeze(0)
                e_phase = torch.exp(1j * (rho @ self.phase_bias.unsqueeze(0)))
                e_pos = rho @ self.position_bias.unsqueeze(0)
                E_mc = torch.eye(self.ULA_M_sensors,dtype=torch.complex64,device=self.device).unsqueeze(0) + rho.type(torch.complex64).unsqueeze(2) * self.ula_mcm.unsqueeze(0)
                ULA_sensor_locations_e = self.ULA_sensor_locations.unsqueeze(0) + e_pos
                imag = 2*torch.pi*(1/self.lam)*torch.matmul(ULA_sensor_locations_e.unsqueeze(2),torch.cos(source_angles))
                temp = e_gain.unsqueeze(2) * e_phase.unsqueeze(2) * torch.exp(torch.complex(torch.zeros_like(imag),imag))
                V = torch.matmul(E_mc,temp)
        else:
            raise TypeError(f"invalid mode={mode}, must be MRA or ULA")
        return V

@torch.no_grad()
def get_random_source_angles(deg_range: List[float], min_sep: float, num_sources: int, num_datapoints: int, mode: str, seed: int):
    filepath = os.path.join('./source_angles/',f"mode={mode}_source_angles_rg={str(deg_range)}_sep={min_sep}_nsrc={num_sources}_ndatapoints={num_datapoints}_seed={seed}.hdf5".replace(' ',''))
    if os.path.isfile(filepath):
        #print((f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] A dataset of random source angles already exists at {filepath}"
               #" (remove the existing dataset if you want to create a new one). Start loading..."))
        with h5py.File(filepath,'r') as file:
            source_angles = file["source_angles"][:]
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Finished loading the dataset")
    else:
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] No dataset found at {filepath}, generate a new dataset of random source angles")
        source_angles = np.zeros((num_datapoints,num_sources),dtype=np.float32)
        for i in tqdm(range(num_datapoints),leave=True):
            source_angles[i,:] = random_source_angles(deg_range,min_sep,num_sources) * np.pi/180
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Saving the dataset to path {filepath}")
        dir_path('./source_angles/')
        with h5py.File(filepath,'w') as file:
            file.create_dataset(name="source_angles",data=source_angles,compression='gzip')
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Dataset saved at {filepath}")
    return source_angles

@torch.no_grad()
def get_source_and_noise_random_base(base_L: int, num_sources: int, T_snapshots: int, M: int, seed: int, mode: str):
    filepath = os.path.join('./source_noise_random_base/',f"mode={mode}_sn_random_baseL={base_L}_M={M}_nsrc={num_sources}_Tsnapshots={T_snapshots}_seed={seed}.hdf5".replace(' ',''))
    if os.path.isfile(filepath):
        #print((f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] A dataset of source/noise random base already exists at {filepath}"
               #" (remove the existing dataset if you want to create a new one). Start loading..."))
        with h5py.File(filepath,'r') as file:
            source_base = file["source_base"][:]
            noise_base = file["noise_base"][:]
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Finished loading the dataset")
    else:
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] No dataset found at {filepath}, generate a new dataset of random source angles")
        source_base = torch.randn(base_L,num_sources,T_snapshots,dtype=torch.cfloat) # random source base
        noise_base = torch.randn(base_L,M,T_snapshots,dtype=torch.cfloat) # noise base
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Saving the dataset to path {filepath}")
        dir_path('./source_noise_random_base/')
        with h5py.File(filepath,'w') as file:
            file.create_dataset(name="source_base",data=source_base,compression='gzip')
            file.create_dataset(name="noise_base",data=noise_base,compression='gzip')
        #print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Dataset saved at {filepath}")
    return source_base, noise_base

@torch.no_grad()
def generate_batch_cov_MRA_ULA(array_manifold: ArrayManifold, rho: float, mix_rho: bool, source_base: torch.Tensor, noise_base: torch.Tensor, T_snapshots: int, p: torch.Tensor,
                               SNR: float, source_angles: List[float], use_variance: bool, noisy_ULA: bool = False, normalization: str = 'disabled',diag_src_cov=True,return_MRA_snapshots=False,total_power_one=False):
    # source
    if total_power_one is True:
        p = p / torch.sum(p,1).unsqueeze(1) # the given p is a 2d tensor (L x # of sources)
    # source_angles is a 2d tensor (L x # of sources) and sources is a 3d tensor (L x # of sources x T)
    sources = torch.sqrt(p).to(torch.cfloat).unsqueeze(2)*source_base  # T_snapshots complex zero-mean circularly-symmetric gaussian random vectors
    # noise (L x M x T)
    noise = (1/(10**(SNR/20)))*noise_base  # T_snapshots complex zero-mean circularly-symmetric gaussian random vectors
    # source_angles is converted to a 3d tensor (L x 1 x # of sources)
    source_angles = source_angles.unsqueeze(1)
    # ULA
    V_ULA = array_manifold.get_V(rho=0,source_angles=source_angles,mix=False,mode='ULA')
    # ULA sample covariance with or without noise
    Y_ULA = torch.matmul(V_ULA,sources)
    Y_ULA = Y_ULA + noise if noisy_ULA is True else Y_ULA
    if diag_src_cov is True:
        # ULA noise-free diagonal covariance matrix or diagonal sample covariance matrix
        if use_variance is True:
            cov_ULA = torch.matmul(torch.matmul(V_ULA,torch.vmap(torch.diag)(p).to(torch.cfloat)),V_ULA.conj().transpose(-2,-1))
        else:
            source_sample_cov = (1/T_snapshots)*torch.matmul(sources,sources.conj().transpose(-2,-1))
            source_sample_cov_diag = torch.vmap(torch.diag)(torch.diagonal(source_sample_cov,dim1=-2,dim2=-1))
            cov_ULA = torch.matmul(torch.matmul(V_ULA,source_sample_cov_diag),V_ULA.conj().transpose(-2,-1))
    else:
        cov_ULA = (1/T_snapshots)*torch.matmul(Y_ULA,Y_ULA.conj().transpose(-2,-1))
    # imperfect or perfect (depending on rho) ULA with holes or MRA (no zero padding)
    V = array_manifold.get_V(rho=rho,source_angles=source_angles,mix=mix_rho,mode='MRA')
    noise = noise[:,array_manifold.sensor_grid,:]
    Y_nopad = torch.matmul(V,sources) + noise
    if return_MRA_snapshots is True:
        return_MRA = Y_nopad
    else:
        cov_MRA = (1/T_snapshots)*torch.matmul(Y_nopad,Y_nopad.conj().transpose(-2,-1))
        # normalization
        return_MRA = cov_normalize(cov_MRA,normalization,array_manifold.N_sensors)
    # normalization
    cov_ULA = cov_normalize(cov_ULA,normalization,array_manifold.ULA_M_sensors)
    return return_MRA, cov_ULA, array_manifold.MRA_sensor_locations, array_manifold.ULA_sensor_locations

class CovMapDataset(Dataset):
    def __init__(self,
                 mode: str,
                 L: int,
                 d: float,
                 lam: float,
                 N_sensors: int,
                 T_snapshots: int,
                 num_sources: List[int],
                 snr_range: List[float],
                 snr_uniform: bool,
                 snr_list: List[float],
                 snr_prob: List[float],
                 seed: int,
                 deg_range: List[float],
                 min_sep: List[float],
                 diag_src_cov: bool,
                 use_variance: bool,
                 gain_bias: List[float],
                 phase_bias_deg: List[float],
                 position_bias: List[float],
                 mc_mag_angle: List[float],
                 rho: float,
                 mix_rho: bool,
                 base_L: int = 10000,
                 dynamic: bool = True,
                 random_power: bool = False,
                 power_range: List[float] = [0.1,1.0],
                 total_power_one: bool = False,
                 normalization: str = 'disabled',
                 device: str = 'cpu',
                 save_dataset: bool = False
                 ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.L = L
        self.d = d
        self.lam = lam
        self.N_sensors = N_sensors
        self.T_snapshots = T_snapshots
        self.num_sources = num_sources
        self.snr_range = snr_range
        self.snr_uniform = snr_uniform
        self.snr_list = np.array(snr_list)
        self.snr_prob = np.array(snr_prob)
        self.deg_range = deg_range
        self.min_sep = min_sep
        self.diag_src_cov = diag_src_cov
        self.use_variance = use_variance
        self.dynamic = dynamic
        self.random_power = random_power
        self.power_range = power_range
        self.total_power_one = total_power_one
        self.normalization = normalization
        self.device = device
        self.base_L = base_L
        self.N_datapoints_per_nsrc = self.base_L * L
        self.N_datapoints = self.N_datapoints_per_nsrc * len(num_sources)
        self.cov_in = None
        self.cov_out = None
        self.source_number = None
        self.rho = rho
        self.mix_rho = mix_rho
        self.pid180 = np.pi/180
        self.array_manifold = ArrayManifold(d=d,lam=lam,N_sensors=N_sensors,gain_bias=gain_bias,phase_bias_deg=phase_bias_deg,
                                            position_bias=position_bias,mc_mag_angle=mc_mag_angle,device=device)
        dataset_folder = f'./covaug_datasets_{mode}/'
        
        path = os.path.join(dataset_folder,(f"{mode}_d={d}_lam={lam}_L={L}_N={N_sensors}_T={T_snapshots}_nsrc={str(num_sources)}_snr={str(snr_range)}_uni={int(snr_uniform)}"
                f"_spr={round(snr_prob[-1]/snr_prob[0],1)}_seed={seed}_rg={str(deg_range)}_sep={str(min_sep)}_rho={rho}_mix={int(mix_rho)}_dg={int(diag_src_cov)}"
                f"_uv={int(use_variance)}_baseL={base_L}_rp={int(random_power)}_pr={str(power_range)}_tpo={int(total_power_one)}"
                f"_nor={normalization}.hdf5").replace(' ','').replace('.','').replace(',','_').replace('[','').replace(']',''))

        if dynamic is False:
            if os.path.exists(path):
                print((f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] A dataset already exists at {path}"
                       " (remove the existing dataset if you want to create a new one). Start loading..."))
                with h5py.File(path,'r') as file:
                    self.cov_out = torch.from_numpy(file["cov_out"][:])
                    self.cov_in = torch.from_numpy(file["cov_in"][:])
                    self.source_number = torch.from_numpy(file["source_number"][:])
                    self.angles = torch.from_numpy(file["angles"][:])
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Finished loading the dataset")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] No dataset found at {path}, generate a new dataset for CovMap")
                self.angles = torch.zeros(self.N_datapoints,max(self.num_sources))
                with torch.device(self.device):
                    with tqdm(total=self.N_datapoints,leave=True) as pbar:
                        for k in range(len(num_sources)):
                            source_angles = torch.from_numpy(get_random_source_angles(deg_range=deg_range,min_sep=min_sep[k],num_sources=num_sources[k],num_datapoints=self.N_datapoints_per_nsrc,mode=mode,seed=seed))
                            self.angles[k * self.base_L * L:(k+1) * self.base_L * L,:num_sources[k]] = torch.sort(source_angles)[0]
                            for j in range(self.L):
                                if self.random_power is True:
                                    p = (power_range[1] - power_range[0]) * torch.rand(self.base_L,num_sources[k]) + power_range[0]
                                    p = p * p.size(1) / torch.sum(p,dim=1,keepdim=True)
                                else:
                                    p = torch.ones(self.base_L,num_sources[k])
                                if snr_uniform is True:
                                    SNR = torch.rand(self.base_L,1,1) * (snr_range[1]-snr_range[0]) + snr_range[0]
                                else:
                                    SNR = torch.from_numpy(np.random.choice(a=self.snr_list,size=self.base_L,p=self.snr_prob).astype(np.float32)).unsqueeze(1).unsqueeze(2)
                                source_base = torch.randn(self.base_L,source_angles.shape[1],T_snapshots,dtype=torch.cfloat) # random source base
                                noise_base = torch.randn(self.base_L,self.array_manifold.ULA_M_sensors,T_snapshots,dtype=torch.cfloat) # noise base
                                cov_MRA, cov_ULA, _, _ = generate_batch_cov_MRA_ULA(self.array_manifold,self.rho,self.mix_rho,source_base,noise_base,T_snapshots,p,SNR,source_angles[j*self.base_L:(j+1)*self.base_L,:],
                                                                                    use_variance,False,normalization,diag_src_cov,False,total_power_one)
                                l = k * self.base_L * L + j * self.base_L
                                if self.cov_out is None:
                                    self.cov_out = torch.zeros(self.N_datapoints,cov_ULA.shape[1],cov_ULA.shape[2],dtype=torch.complex64)
                                    self.cov_out[:self.base_L,:,:] = cov_ULA
                                else:
                                    self.cov_out[l:l+self.base_L,:,:] = cov_ULA
                                if self.cov_in is None:
                                    self.cov_in = torch.zeros(self.N_datapoints,cov_MRA.shape[1],cov_MRA.shape[2],dtype=torch.complex64)
                                    self.cov_in[:self.base_L,:,:] = cov_MRA
                                else:
                                    self.cov_in[l:l+self.base_L,:,:] = cov_MRA
                                if self.source_number is None:
                                    self.source_number = torch.zeros(self.N_datapoints,dtype=torch.int16)
                                    self.source_number[:self.base_L] = num_sources[k]
                                else:
                                    self.source_number[l:l+self.base_L] = num_sources[k]
                                pbar.update(self.base_L)
                if save_dataset is True:
                    if not os.path.isdir(dataset_folder):
                        os.mkdir(dataset_folder)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Saving the dataset to path {path}")
                    with h5py.File(path,'w') as file:
                        file.create_dataset(name="cov_in",data=self.cov_in.numpy(),compression='gzip')
                        file.create_dataset(name="cov_out",data=self.cov_out.numpy(),compression='gzip')
                        file.create_dataset(name="source_number",data=self.source_number.numpy(),compression='gzip')
                        file.create_dataset(name="angles",data=self.angles.numpy(),compression='gzip')
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Dataset saved at {path}")

    def __len__(self):
        return self.N_datapoints
    
    def __getitem__(self, idx):
        if self.dynamic is True:
            s = np.random.choice(a=self.num_sources,size=1).item()
            source_number = s + 1
            self.angles = torch.zeros(max(self.num_sources))
            source_angles = torch.from_numpy(random_source_angles(self.deg_range,self.min_sep[s],source_number)*self.pid180).unsqueeze(0)
            self.angles[:source_number] = source_angles
            if self.random_power is True:
                p = (self.power_range[1] - self.power_range[0]) * torch.rand(1,source_number) + self.power_range[0]
                p = p * p.size(1) / torch.sum(p,dim=1,keepdim=True)
            else:
                p = torch.ones(1,source_number)
            source_base = torch.randn(1,source_angles.shape[1],self.T_snapshots,dtype=torch.cfloat) # random source base
            noise_base = torch.randn(1,self.array_manifold.ULA_M_sensors,self.T_snapshots,dtype=torch.cfloat) # noise base
            if self.snr_uniform is True:
                SNR = torch.rand(1,1,1) * (self.snr_range[1]-self.snr_range[0]) + self.snr_range[0]
            else:
                SNR = torch.from_numpy(np.random.choice(a=self.snr_list,size=1,p=self.snr_prob).astype(np.float32)).unsqueeze(1).unsqueeze(2)
            cov_MRA, cov_ULA, _, _ = generate_batch_cov_MRA_ULA(self.array_manifold,self.rho,self.mix_rho,source_base,noise_base,self.T_snapshots,p,SNR,source_angles,
                                                                self.use_variance,False,self.normalization,self.diag_src_cov,False,self.total_power_one)
            cov_out = cov_ULA[0,:,:]
            cov_in = cov_MRA[0,:,:]
        else:
            cov_out = self.cov_out[idx,:,:]
            cov_in = self.cov_in[idx,:,:]
            source_number = self.source_number[idx]
            angles = self.angles[idx,:]
        return cov_in, cov_out, source_number, angles

class Cov2DoADataset(Dataset):
    def __init__(self,
                 mode: str,
                 d: float,
                 lam: float,
                 N_sensors: int,
                 T_snapshots: int,
                 num_sources: int,
                 snr_range: List[float],
                 seed: int,
                 deg_range: List[float],
                 min_sep: float,
                 L: int,
                 base_L: int,
                 gain_bias: List[float],
                 phase_bias_deg: List[float],
                 position_bias: List[float],
                 mc_mag_angle: List[float],
                 rho: float,
                 mix_rho: bool,
                 provide_noise_var: bool = False,
                 random_power: bool = False,
                 power_range: List[float] = [0.1,1.0],
                 total_power_one: bool = False,
                 evenly_distributed: bool = False,
                 return_snapshots: bool = False,
                 device: str = 'cpu',
                 save_dataset: bool = False
                 ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.d = d
        self.lam = lam
        self.N_sensors = N_sensors
        self.T_snapshots = T_snapshots
        self.num_sources = num_sources
        self.snr_range = snr_range
        self.deg_range = deg_range
        self.min_sep = min_sep
        self.L = L
        self.base_L = base_L
        self.random_power = random_power
        self.power_range = power_range
        self.evenly_distributed = evenly_distributed
        if evenly_distributed is True:
            if self.L != 1:
                raise ValueError("L must be 1 because evenly_distributed is True (angles are not random)")
        self.device = device
        self.N_datapoints = self.base_L * L
        self.rho = rho
        self.mix_rho = mix_rho
        self.provide_noise_var = provide_noise_var # only meaningful when random_power is False
        if provide_noise_var is True and random_power is True:
            raise ValueError("provide_noise_var can only be True when random_power is False")
        self.array_manifold = ArrayManifold(d=d,lam=lam,N_sensors=N_sensors,gain_bias=gain_bias,phase_bias_deg=phase_bias_deg,
                                            position_bias=position_bias,mc_mag_angle=mc_mag_angle,device=device)
        if return_snapshots is True:
            self.data_in = torch.zeros(self.N_datapoints,N_sensors,T_snapshots,dtype=torch.complex64)
        else:
            self.data_in = torch.zeros(self.N_datapoints,N_sensors,N_sensors,dtype=torch.complex64)
        if provide_noise_var is True:
            self.noise_var = torch.zeros(self.N_datapoints,dtype=torch.float64)

        dataset_folder = f'./cov2DoA_datasets_{mode}/'
        
        path = (f"{dataset_folder}{mode}_d={d}_lam={lam}_N={N_sensors}_T={T_snapshots}_nsrc={num_sources}_snr={str(snr_range).replace(' ','')}"
                f"_seed={seed}_degr={str(deg_range).replace(' ','')}_sep={min_sep}_rho={rho}_mix={int(mix_rho)}_L={L}_baseL={base_L}"
                f"pnv={int(provide_noise_var)}_rp={int(random_power)}_tpo={int(total_power_one)}_ed={int(evenly_distributed)}_rsnap={int(return_snapshots)}.hdf5")

        if os.path.exists(path):
            tqdm.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] A dataset already exists at {path} (remove the existing dataset if you want to create a new one). Start loading...")
            with h5py.File(path,'r') as file:
                self.DoA = torch.from_numpy(file["DoA"][:])
                self.data_in = torch.from_numpy(file["data_in"][:])
                if provide_noise_var is True:
                    self.noise_var = torch.from_numpy(file["noise_var"][:])
            tqdm.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Finished loading the dataset")
        else:
            tqdm.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] No dataset found at {path}, generate a new dataset for DoA estimation")
            self.DoA = torch.zeros(self.N_datapoints,num_sources,dtype=torch.float64)
            with torch.device(self.device):
                if evenly_distributed is True:
                    source_angle = np.linspace(deg_range[0],deg_range[1],num_sources+2,dtype=np.float32)[1:-1] * np.pi/180
                else:
                    source_angles = get_random_source_angles(deg_range=deg_range,min_sep=min_sep,num_sources=num_sources,num_datapoints=self.L,mode=mode,seed=seed)
                with tqdm(total=self.N_datapoints,leave=True) as pbar:
                    for j in range(self.L):
                        if evenly_distributed is True:
                            src_angle = torch.from_numpy(source_angle)
                        else:
                            src_angle = torch.from_numpy(source_angles[j,:])
                        repeat_src_angles = src_angle.unsqueeze(0).repeat(self.base_L,1)
                        if self.random_power is True:
                            p = (self.power_range[1] - self.power_range[0]) * torch.rand(self.base_L,num_sources) + self.power_range[0]
                            p = p * p.size(1) / torch.sum(p,dim=1,keepdim=True)
                        else:
                            p = torch.ones(self.base_L,num_sources)
                        SNR = torch.rand(self.base_L,1,1) * (snr_range[1]-snr_range[0]) + snr_range[0]
                        source_base, noise_base = get_source_and_noise_random_base(self.base_L,num_sources,T_snapshots,self.array_manifold.ULA_M_sensors,seed,'eval')
                        data_in, _, _, _ = generate_batch_cov_MRA_ULA(self.array_manifold,self.rho,self.mix_rho,source_base,noise_base,T_snapshots,p,SNR,repeat_src_angles,
                                                                      False,True,'disabled',False,return_snapshots,total_power_one)
                        l = j * self.base_L
                        self.DoA[l:l+self.base_L,:] = torch.sort(src_angle)[0].unsqueeze(0).repeat(self.base_L,1)
                        self.data_in[l:l+self.base_L,:,:] = data_in
                        if provide_noise_var is True:
                            self.noise_var[l:l+self.base_L] = 1/(10**(SNR.squeeze()/10))
                        pbar.update(self.base_L)
            if save_dataset is True:
                if not os.path.isdir(dataset_folder):
                    os.mkdir(dataset_folder)
                tqdm.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Saving the dataset to path {path}")
                with h5py.File(path,'w') as file:
                    file.create_dataset(name="data_in",data=self.data_in.numpy(),compression='gzip')
                    file.create_dataset(name="DoA",data=self.DoA.numpy(),compression='gzip')
                    if provide_noise_var is True:
                        file.create_dataset(name="noise_var",data=self.noise_var.numpy(),compression='gzip')
                tqdm.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [data.py] Dataset saved at {path}")

    def __len__(self):
        return self.N_datapoints
    
    def __getitem__(self, idx):
        label = self.DoA[idx,:]
        data = self.data_in[idx,:,:]
        if self.provide_noise_var is True:
            noise_var = self.noise_var[idx]
            return data, noise_var, label
        else:
            return data, label

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    d = 0.01
    lam = 0.02
    N_sensors = 5
    T_snapshots = 50
    num_sources = 5
    snr_range = [10,20]
    seed = 0
    deg_range = [30,150]
    min_sep = 10
    L = 2
    diag_src_cov = True
    use_variance = True
    dynamic = False
    provide_noise_var = True
    random_power = False
    power_range = [0.1,1.0]
    return_snapshots = True
    normalization = 'disabled'
    mode = 'testdryrun'
    base_L = 100
    gain_bias = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    phase_bias_deg = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    position_bias = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    mc_mag_angle = [0.1,0.1]
    rho = 0
    mix_rho = False
    save_dataset = False
    
    DoA_dataset = Cov2DoADataset(mode,d,lam,N_sensors,T_snapshots,num_sources,snr_range,seed,deg_range,min_sep,L,base_L,gain_bias,phase_bias_deg,position_bias,mc_mag_angle,rho,mix_rho,provide_noise_var,random_power,power_range,return_snapshots,device='cpu',save_dataset=save_dataset)
    dataloader = DataLoader(DoA_dataset,batch_size=512,shuffle=True,num_workers=0,pin_memory=True,drop_last=False)

    print(dataloader)
    print(len(dataloader))
    print(len(DoA_dataset))

    tic = time.time()
    for idx, (data,noise_var,label) in enumerate(dataloader):
        print(idx)
        print(data.shape)
        print(data[0,:,:])
        print(noise_var)
        print(noise_var.shape)
        print(label.shape)
    toc = time.time()
    print(toc-tic)