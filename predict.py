"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen
"""
import os
import torch
from models import model_dict
from DoA import BasePredictor
from utils import data_in_preprocess, cov_normalize

class Predictor(BasePredictor):
    def __init__(self, model_folder: str, device: str ='cuda:0'):
        meta_path = os.path.join(model_folder,'meta_data.pt')
        model_path = os.path.join(model_folder,'best_model.pt')
        if not (os.path.isfile(meta_path) and os.path.isfile(model_path)):
            self.isfunctional = False
            return
        else:
            self.isfunctional = True
        self.model_folder = model_folder
        self.device = device
        array_data = torch.load(meta_path)
        self.name = array_data['model']
        self.N_sensors = array_data['N_sensors']
        self.normalization = array_data['normalization']
        self.EnEnH = array_data['EnEnH'] if 'EnEnH' in array_data else False

        pretrained_model = torch.load(model_path,map_location=device)
        self.net = model_dict[self.name]()
        self.out_type = self.net.out_type
        self.net.load_state_dict(pretrained_model,strict=True)
        self.net = self.net.to(self.device)
        self.net.eval()

    def get_ULA_cov(self, cov: torch.Tensor, is_snapshot: bool = False):
        cov, _ = data_in_preprocess(cov)
        cov = cov_normalize(cov,self.normalization,self.N_sensors)
        with torch.no_grad():
            outputs = self.net(cov.to(self.device)).cpu()
        return outputs

    def get_DoA(self, data_in: torch.Tensor, num_sources: int, is_snapshot: bool = False, noise_var: torch.Tensor = None):
        if self.out_type == 'direct':
            batch_size = data_in.shape[0]
            out = self.get_ULA_cov(data_in,False)
            DoA = torch.sort(out[:,:num_sources])[0]
            success = [True for _ in range(batch_size)]
            return DoA, success
        elif self.out_type == 'branch':
            batch_size = data_in.shape[0]
            cov, _ = data_in_preprocess(data_in)
            cov = cov_normalize(cov,self.normalization,self.N_sensors)
            with torch.no_grad():
                out = self.net(cov.to(self.device))
            DoA = torch.sort(out[num_sources-1].cpu())[0]
            success = [True for _ in range(batch_size)]
            return DoA, success
        else:
            return super().get_DoA(data_in,num_sources,is_snapshot,noise_var)