"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen

SDP-based baselines used in the paper

https://arxiv.org/abs/2408.16605
"""
import numpy as np
import matlab.engine
import torch
from utils import MRA
from DoA import BasePredictor
import os

class SDPMRA2ULA(BasePredictor):
    def __init__(self, N_sensors: int):
        self.N_sensors = N_sensors
        _, sensor_grid, N_a = MRA(N_sensors,1)
        self.M_sensors = N_a + 1
        self.S = np.zeros((self.N_sensors,self.M_sensors),dtype=np.cfloat)
        for i in range(self.N_sensors):
            self.S[i,sensor_grid[i]] = 1

class SDPCovMRA2ULA_Wasserstein_SDPT3(SDPMRA2ULA):
    def __init__(self, N_sensors: int):
        super().__init__(N_sensors)
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.path.dirname(os.path.realpath(__file__)), nargout=0)

    def _get_one_ULA_cov(self,cov: torch.Tensor):
        return np.array(self.eng.Wasserstein(matlab.double(cov.tolist(),is_complex=True),matlab.double(self.S.tolist(),is_complex=True)))

class SDPCovMRA2ULA_SPA_SDPT3(SDPMRA2ULA):
    def __init__(self, N_sensors: int, use_noise_var: bool, remove_noise: bool = True):
        super().__init__(N_sensors)
        self.use_noise_var = use_noise_var
        self.remove_noise = remove_noise
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.path.dirname(os.path.realpath(__file__)), nargout=0)

    def _get_one_ULA_cov(self,cov: torch.Tensor, noise_var: torch.Tensor = None):
        if noise_var == None and self.use_noise_var is False:
            R = np.array(self.eng.SPA(matlab.double(cov.tolist(),is_complex=True),matlab.double(self.S.tolist(),is_complex=True)))
        elif noise_var != None and self.use_noise_var is True:
            R = np.array(self.eng.SPA_noisevar(matlab.double(cov.tolist(),is_complex=True),matlab.double(self.S.tolist(),is_complex=True),matlab.double(noise_var.tolist())))
        else:
            raise ValueError(f"Incorrect mode: self.use_noise_var={self.use_noise_var} and noise_var={noise_var}")
        if self.remove_noise is True:
            L, _ = np.linalg.eigh(R)
            R = R - min(L) * np.eye(L.shape[0])
        return R

class SDPSnapshotMRA2ULA_ProxCov_SDPT3(SDPMRA2ULA):
    def __init__(self, N_sensors: int, epsilon: float):
        super().__init__(N_sensors)
        self.need_snapshot = True
        self.epsilon = epsilon
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.path.dirname(os.path.realpath(__file__)), nargout=0)

    def _get_one_ULA_cov(self,Y: torch.Tensor):
        return np.array(self.eng.ProxCov(matlab.double(Y.tolist(),is_complex=True),matlab.double(self.S.tolist(),is_complex=True),matlab.double(self.epsilon)))

class SDPCovMRA2ULA_StructCovMLE_SDPT3(SDPMRA2ULA):
    def __init__(self, N_sensors: int, epsilon: float, max_iter: int, use_noise_var: bool):
        super().__init__(N_sensors)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.use_noise_var = use_noise_var
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.path.dirname(os.path.realpath(__file__)), nargout=0)

    def _get_one_ULA_cov(self, cov: torch.Tensor, noise_var: torch.Tensor = None):
        if noise_var == None and self.use_noise_var is False:
            return np.array(self.eng.StructCovMLE(matlab.double(cov.tolist(),is_complex=True),matlab.double(self.S.tolist(),is_complex=True),matlab.double(self.epsilon),matlab.int16(self.max_iter)))
        elif noise_var != None and self.use_noise_var is True:
            return np.array(self.eng.StructCovMLE_noisevar(matlab.double(cov.tolist(),is_complex=True),matlab.double(self.S.tolist(),is_complex=True),matlab.double(self.epsilon),matlab.int16(self.max_iter),matlab.double(noise_var.tolist())))
        else:
            raise ValueError(f"Incorrect mode: self.use_noise_var={self.use_noise_var} and noise_var={noise_var}")