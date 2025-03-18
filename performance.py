"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen
"""
import argparse
import numpy as np
import torch
import os
import re
import scipy.io
from datetime import datetime
from tqdm import tqdm
from utils import dir_path, file_path, check_device
from eval.crlb import uncorrelated_CRLB
from data import Cov2DoADataset
from DoA import CovMRA2ULA_DA, CovMRA2ULA_SS
from SDP.SDP import SDPCovMRA2ULA_Wasserstein_SDPT3, SDPCovMRA2ULA_SPA_SDPT3, SDPSnapshotMRA2ULA_ProxCov_SDPT3, SDPCovMRA2ULA_StructCovMLE_SDPT3
from predict import Predictor

def get_name(s: str):
    patterns = [r'/([^;]*)_t=',r't=([^;]*)_v=',r'loss=([^;]*)_mu=',r'mu=([^;]*)_mo=',
                r'mo=([^;]*)_bs=',r'bs=([^;]*)_epoch=',r'epoch=([^;]*)_wd',r'wd=([^;]*)_seed=',r'spr=([^;]*)_dg=',r'sep=([^;]*)_rho=',r'nsrc=([^;]*)_T=',r'T=([^;]*)_rg=',
                r'rg=([^;]*)_sep=',r'rho=([^;]*)_mix=',r'mix=([^;]*)_snr=',r'snr=([^;]*)_uni=',r'uni=([^;]*)_spr=',r'dg=([^;]*)_uv=',
                r'uv=([^;]*)_crs=',r'crs=([^;]*)_dy=',r'rp=([^;]*)_pr=',r'_pr=([^;]*)_tpo=',r'tpo=([^;]*)_nor=',r'nor=([^;]*)_oc=']
    name = ""
    for i in patterns:
        x = re.findall(i,s)
        if len(x) != 0:
            name += x[0]
    name = name.replace('./','').replace('/','_').replace('.','_').replace(',','').replace('[','').replace(']','').replace('-','')
    return name

def display_evaluation_status(N_sensors: int, num_sources: int, T_snapshots: int, SNR: float, crb: float, trials: int, mse: np.ndarray, bias: np.ndarray, success: np.ndarray, num_random_thetas: int, j: int, i: int, k: int, rho: float):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tqdm.write(f'{t} [performance.py] N: {N_sensors} | # of trials: '+f'{trials}'.rjust(6)+' | # of sources: '+f'{num_sources}'.rjust(3)+' | T_snapshots: '+f'{T_snapshots}'.rjust(4)
               +' | SNR (dB): '+f'{SNR}'.rjust(6)+' | rho: '+f'{rho}'.rjust(3)+' '*46+' | mean CRB: '+f'{crb*(180/np.pi)**2:.3f}'.rjust(8)+' deg^2 ('+f'{crb:.7f}'.rjust(9)+' rad^2)')
    for m in mse.keys():
        if num_random_thetas == 1:
            bias_str = '| bias: '+f'{bias[m][j,i,k]:.7f}'.rjust(9)
        else:
            bias_str = ''
        tqdm.write(f'{t} [performance.py] '+f'{N_sensors}-MRA '.rjust(7)+m.rjust(135)+' | '.ljust(8)+'MSE: '+f'{mse[m][j,i,k]*(180/np.pi)**2:.3f}'.rjust(8)+' deg^2 ('+f'{mse[m][j,i,k]:.7f}'.rjust(9)+' rad^2) '+bias_str+' | success: '+f'{round(success[m][j,i,k])}/{trials}'.rjust(8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performance evaluation of DoA estimation methods',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results_folder', default='./results/', type=dir_path, help='path to the results folder')
    parser.add_argument('--resume_from', default=None, type=file_path, help='resume from the previous unfinished checkpoint file path')
    parser.add_argument('--device', default='cuda:0', type=check_device, help='specify a CUDA or CPU device, e.g., cuda:0, cuda:1, or cpu')
    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers of the dataloader')
    parser.add_argument('--seed', default=9, type=int, help='random seed')
    parser.add_argument('--num_random_thetas', default=100, type=int, help='Number of random angles for evaluation')
    parser.add_argument('--trials_per_theta', default=100, type=int, help='Number of trials per random theta (number of random source signals per theta)')
    parser.add_argument('--SNR_list', default=[i for i in range(-10,21,2)], nargs='+', type=float, help='List of SNRs for evaluation')
    parser.add_argument('--T_snapshots_list', default=[i for i in range(50,51)], nargs='+', type=int, help='List of snapshots for evaluation')
    parser.add_argument('--num_sources_list', default=[1,2,3,4,5,6,7,8,9], nargs='+', type=int, help='Number of sources for evaluation')
    parser.add_argument('--provide_noise_var', default=1, type=int, help='1 or 0. (1): provide the ground truth noise variance, (0): not provide the noise variance')
    parser.add_argument('--random_power', default=0, type=int, help='1 or 0. (1): random source power, (0): equal source power')
    parser.add_argument('--power_range', default=[0.1,1.0], nargs='+', type=float, help='range of the random power')
    parser.add_argument('--total_power_one', default=0, type=int, help='1 or 0. (1): normalize the power of sources such that the total source power is one, (0): no normalization')
    parser.add_argument('--evenly_distributed', default=0, type=int, help='1 or 0. (1): source angles are evenly distributed, (0): randomly distributed')
    parser.add_argument('--return_snapshots', default=0, type=int, help='1 or 0. (1): return snapshots as input, (0): return covariance matrices as input')
    parser.add_argument('--d', default=0.01, type=float, help='sensor spacing')
    parser.add_argument('--lam', default=0.02, type=float, help='wavelength lambda')
    parser.add_argument('--N_sensors', default=5, type=int, help='N-element MRA')
    parser.add_argument('--deg_range', default=[30,150], nargs='+', type=int, help='DoA estimation range in degrees (0 to 180)')
    parser.add_argument('--min_sep', default=[4,4,4,4,4,4,4,4,4], nargs='+', type=float, help='List of minimum separations in degrees for the n_sources_train/val (must be a positive number)')
    parser.add_argument('--save_dataset', default=1, type=int, help='1 or 0. (1): save all datasets that are going to be generated, (0): not save')
    parser.add_argument('--gain_bias', default=[0.0,0.2,0.2,0.2,0.2,0.2,-0.2,-0.2,-0.2,-0.2], nargs='+', type=float, help='Gain bias')
    parser.add_argument('--phase_bias_deg', default=[0,-30,-30,-30,-30,-30,30,30,30,30], nargs='+', type=float, help='Phase bias in degrees')
    parser.add_argument('--position_bias', default=[0.0,-0.2,-0.2,-0.2,-0.2,-0.2,0.2,0.2,0.2,0.2], nargs='+', type=float, help='Position bias')
    parser.add_argument('--mc_mag_angle', default=[0.3,60], nargs='+', type=float, help='magnitude and phase (in degrees) of the mutual coupling coefficient')
    parser.add_argument('--rho', default=0.0, type=float, help='A number in [0,1] describing the degree of array imperfections')
    parser.add_argument('--SPA', default=0, type=int, help='1 or 0. (1): evaluate the performance of SPA, (0): not evaluting SPA')
    parser.add_argument('--SPA_noisevar', default=0, type=int, help='1 or 0. (1): evaluate the performance of SPA using the noise variance, (0): not evaluting SPA using the noise variance')
    parser.add_argument('--Wasserstein', default=0, type=int, help='1 or 0. (1): evaluate the performance of Wasserstein, (0): not evaluting Wasserstein')
    parser.add_argument('--ProxCov', default=0, type=int, help='1 or 0. (1): evaluate the performance of ProxCov, (0): not evaluting ProxCov')
    parser.add_argument('--ProxCov_epsilon', default=1e-5, type=float, help='the epsilon parameter of ProxCov')
    parser.add_argument('--StructCovMLE', default=0, type=int, help='1 or 0. (1): evaluate the performance of StructCovMLE, (0): not evaluting StructCovMLE')
    parser.add_argument('--StructCovMLE_noisevar', default=0, type=int, help='1 or 0. (1): evaluate the performance of StructCovMLE using the noise variance, (0): not evaluting StructCovMLE using the noise variance')
    parser.add_argument('--StructCovMLE_epsilon', default=1e-3, type=float, help='the threshold of the relative change as the stopping criterion of StructCovMLE')
    parser.add_argument('--StructCovMLE_max_iter', default=100, type=int, help='the maximum number of iterations of StructCovMLE')
    parser.add_argument('--DA', default=0, type=int, help='1 or 0. (1): evaluate the performance of DA, (0): not evaluting DA')
    parser.add_argument('--SS', default=0, type=int, help='1 or 0. (1): evaluate the performance of SS, (0): not evaluting SS')
    parser.add_argument('--cov_models', nargs='+', type=str, help='Path to the DNN model checkpoint folder', default=None)

    args = parser.parse_args()
    save_dataset = bool(args.save_dataset)
    results_folder = args.results_folder
    seed = args.seed
    d = args.d
    lam = args.lam
    N_sensors = args.N_sensors
    deg_range = args.deg_range
    min_sep = args.min_sep
    # use 8,8,8,8,10,11,11,12,13 if meaningful CRBs are needed (the minimum separations need to be sufficiently large)
    provide_noise_var = bool(args.provide_noise_var)
    random_power = bool(args.random_power)
    power_range = args.power_range
    total_power_one = bool(args.total_power_one)
    evenly_distributed = bool(args.evenly_distributed)
    return_snapshots = bool(args.return_snapshots)
    num_sources_list = args.num_sources_list
    T_snapshots_list = args.T_snapshots_list
    SNR_list = args.SNR_list
    num_random_thetas = args.num_random_thetas
    trials_per_theta = args.trials_per_theta
    device=args.device
    batch_size = args.batch_size
    ProxCov_epsilon = args.ProxCov_epsilon
    StructCovMLE_epsilon = args.StructCovMLE_epsilon
    StructCovMLE_max_iter = args.StructCovMLE_max_iter

    if len(min_sep) != len(num_sources_list):
        raise ValueError(f"len(min_sep)={len(min_sep)} does not match len(num_sources_list)={len(num_sources_list)}")

    gain_bias = args.gain_bias
    phase_bias_deg = args.phase_bias_deg
    position_bias = args.position_bias
    mc_mag_angle = args.mc_mag_angle
    rho = args.rho

    if evenly_distributed is True:
        if num_random_thetas != 1:
            raise ValueError("num_random_thetas should be 1 because evenly_distributed is True")

    # DoA predictors
    methods = {}
    if args.DA:
        DA = CovMRA2ULA_DA()
        methods.update({'DA': DA})
    if args.SS:
        SS = CovMRA2ULA_SS()
        methods.update({'SS': SS})
    if args.Wasserstein:
        Wasserstein = SDPCovMRA2ULA_Wasserstein_SDPT3(N_sensors)
        methods.update({'Wasserstein': Wasserstein})
    if args.SPA:
        SPA = SDPCovMRA2ULA_SPA_SDPT3(N_sensors,False)
        methods.update({'SPA': SPA})
    if args.SPA_noisevar:
        SPA_noisevar = SDPCovMRA2ULA_SPA_SDPT3(N_sensors,True)
        methods.update({'SPA_noisevar': SPA_noisevar})
    if args.ProxCov:
        ProxCov = SDPSnapshotMRA2ULA_ProxCov_SDPT3(N_sensors,ProxCov_epsilon)
        methods.update({'ProxCov': ProxCov})
    if args.StructCovMLE:
        StructCovMLE = SDPCovMRA2ULA_StructCovMLE_SDPT3(N_sensors,StructCovMLE_epsilon,StructCovMLE_max_iter,False)
        methods.update({'StructCovMLE': StructCovMLE})
    if args.StructCovMLE_noisevar:
        StructCovMLE_noisevar = SDPCovMRA2ULA_StructCovMLE_SDPT3(N_sensors,StructCovMLE_epsilon,StructCovMLE_max_iter,True)
        methods.update({'StructCovMLE_noisevar': StructCovMLE_noisevar})
    if args.cov_models != None:
        for m in args.cov_models:
            name = get_name(m)
            cov_model = Predictor(m,device=device)
            if cov_model.isfunctional:
                methods.update({name: cov_model})
    if len(methods.keys()) == 0:
        raise ValueError("No method. Stop performance evaluation.")
    else:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [performance.py] methods to be evaluated: {list(methods.keys())}")

    # total number of trials
    trials_per_src_num_and_snr = num_random_thetas*trials_per_theta
    total_trials = trials_per_src_num_and_snr*len(num_sources_list)*len(T_snapshots_list)*len(SNR_list)

    # initialize result placeholders
    finished = False

    if args.resume_from != None:
        checkpoint = np.load(args.resume_from,allow_pickle=True).item()
        # check
        if (
            checkpoint['d'] != d or checkpoint['lam'] != lam or checkpoint['N_sensors'] != N_sensors or checkpoint['deg_range'] != deg_range or checkpoint['min_sep'] != min_sep or checkpoint['random_power'] != random_power or
            checkpoint['total_power_one'] != total_power_one or checkpoint['evenly_distributed'] != evenly_distributed or checkpoint['trials_per_theta'] != trials_per_theta or checkpoint['num_random_thetas'] != num_random_thetas or
            checkpoint['num_sources_list'] != num_sources_list or checkpoint['SNR_list'] != SNR_list or checkpoint['T_snapshots_list'] != T_snapshots_list or checkpoint['rho'] != rho or checkpoint['gain_bias'] != gain_bias or
            checkpoint['phase_bias_deg'] != phase_bias_deg or checkpoint['position_bias'] != position_bias or checkpoint['mc_mag_angle'] != mc_mag_angle or checkpoint['ProxCov_epsilon'] != ProxCov_epsilon or
            checkpoint['StructCovMLE_epsilon'] != StructCovMLE_epsilon or checkpoint['StructCovMLE_max_iter'] != StructCovMLE_max_iter
           ):
            raise ValueError("Hyperparameters from resume_from do not match with the current setting")
        # load
        prev_result_path = args.resume_from
        crb_accumulated = checkpoint['crb_accumulated']
        crb_total = checkpoint['crb_total']
        crb_mean = checkpoint['crb_mean']
        method_accumulated = {}
        method_total = {}
        method_success = {}
        method_mse = {}
        method_accu_doa = {}
        method_bias = {}
        for m in methods.keys():
            method_mse.update({m:checkpoint[m]})
            method_success.update({m:checkpoint[f'suc_{m}']})
            method_accumulated.update({m:np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
            method_total.update({m:np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
            method_accu_doa.update({m:checkpoint[f'adoa_{m}']})
            method_bias.update({m:checkpoint[f'bias_{m}']})
        t0 = checkpoint['t0']
    else:
        prev_result_path = None
        crb_accumulated = np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))
        crb_total = np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))
        crb_mean = np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))
        method_accumulated = {}
        method_total = {}
        method_success = {}
        method_mse = {}
        method_accu_doa = {}
        method_bias = {}
        for m in methods.keys():
            method_accumulated.update({m:np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
            method_total.update({m:np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
            method_success.update({m:np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
            method_mse.update({m:np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
            method_accu_doa.update({m:np.zeros((max(num_sources_list),len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
            method_bias.update({m:np.zeros((len(T_snapshots_list),len(num_sources_list),len(SNR_list)))})
        t0 = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

    # evaluation
    with tqdm(total=total_trials) as pbar:
        for j in range(len(T_snapshots_list)):
            for i in range(len(num_sources_list)):
                for k in range(len(SNR_list)):
                    # skip if the results are available in the checkpoint
                    if crb_total[j,i,k] != 0:
                        pbar.update(n=trials_per_src_num_and_snr)
                        continue
                    # create or load a dataset
                    eval_dataset = Cov2DoADataset(mode='eval',d=d,lam=lam,N_sensors=N_sensors,T_snapshots=T_snapshots_list[j],num_sources=num_sources_list[i],snr_range=[SNR_list[k],SNR_list[k]],
                                                  seed=seed,deg_range=deg_range,min_sep=min_sep[i],L=num_random_thetas,base_L=trials_per_theta,gain_bias=gain_bias,phase_bias_deg=phase_bias_deg,
                                                  position_bias=position_bias,mc_mag_angle=mc_mag_angle,rho=rho,mix_rho=False,provide_noise_var=provide_noise_var,random_power=random_power,power_range=power_range,
                                                  total_power_one=total_power_one,evenly_distributed=evenly_distributed,return_snapshots=return_snapshots,device='cpu',save_dataset=save_dataset)
                    dataloader = torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True,drop_last=False)

                    # evaluate each method on the given dataset
                    with torch.no_grad():
                        for idx, x in enumerate(dataloader):
                            if provide_noise_var is True:
                                data_in, noise_var, DoA_gt = x[0], x[1], x[2]
                            else:
                                data_in, DoA_gt = x[0], x[1]
                            for m in methods.keys():
                                if provide_noise_var is True:
                                    DoA_est, successes = methods[m].get_DoA(data_in,num_sources_list[i],return_snapshots,noise_var)
                                else:
                                    DoA_est, successes = methods[m].get_DoA(data_in,num_sources_list[i],return_snapshots)
                                method_accu_doa[m][:num_sources_list[i],j,i,k] += torch.sum(DoA_est[successes],dim=0).numpy()
                                method_success[m][j,i,k] += sum(successes)
                                method_total[m][j,i,k] += DoA_est.shape[1] * sum(successes)
                                method_accumulated[m][j,i,k] += torch.sum((DoA_est[successes] - DoA_gt[successes]) ** 2).numpy()
                            batch_size = DoA_gt.size(0)
                            for b in range(batch_size):
                                temp = np.diag(uncorrelated_CRLB(DoA_gt[b,:], N_sensors, d, lam, SNR_list[k], T_snapshots_list[j],total_power_one))
                                crb_total[j,i,k] += temp.size
                                crb_accumulated[j,i,k] += np.sum(temp)
                            pbar.update(n=batch_size)

                    # compute the MSE and bias
                    for m in methods.keys():
                        method_mse[m][j,i,k] = method_accumulated[m][j,i,k] / method_total[m][j,i,k]
                        method_bias[m][j,i,k] = np.sum(abs(method_accu_doa[m][:num_sources_list[i],j,i,k] / method_success[m][j,i,k] - DoA_gt[0,:].numpy())) / num_sources_list[i]
                    crb_mean[j,i,k] = crb_accumulated[j,i,k] / crb_total[j,i,k]

                    # display current results
                    display_evaluation_status(N_sensors,num_sources_list[i],T_snapshots_list[j],SNR_list[k],crb_mean[j,i,k],len(eval_dataset),method_mse,method_bias,method_success,num_random_thetas,j,i,k,rho)

                    # save settings and results
                    t = datetime.now().strftime('%m-%d_%H_%M_%S')
                    if j + 1 == len(T_snapshots_list) and i + 1 == len(num_sources_list) and k + 1 == len(SNR_list):
                        finished = True
                    result_path = os.path.join(results_folder,
                                               (f"N={N_sensors}_nM={len(methods.keys())}_sep={str(min_sep)}_rg={str(deg_range)}_rp={int(random_power)}_tpo={int(total_power_one)}"
                                                f"_ed={int(evenly_distributed)}_tpt={trials_per_theta}_nt={num_random_thetas}_nSrc={str(num_sources_list)}_SNR={str(SNR_list)}"
                                                f"_T={str(T_snapshots_list)}_rho={str(rho)}_pnv={int(provide_noise_var)}_t0={t0}_t={t}_k={k+1}"
                                                f"_fin={int(finished)}").replace(' ','').replace(',','_').replace('[','').replace(']',''))
                    result = {
                              'crb_accumulated': crb_accumulated,
                              'crb_total': crb_total,
                              'crb_mean': crb_mean,
                              'list_of_methods': list(method_mse.keys()),
                              'd': d,
                              'lam': lam,
                              'N_sensors': N_sensors,
                              'deg_range': deg_range,
                              'min_sep': min_sep,
                              'random_power': random_power,
                              'total_power_one': total_power_one,
                              'evenly_distributed': evenly_distributed,
                              'trials_per_theta': trials_per_theta,
                              'num_random_thetas': num_random_thetas,
                              'num_sources_list': num_sources_list,
                              'SNR_list': SNR_list,
                              'T_snapshots_list': T_snapshots_list,
                              'rho': rho,
                              'gain_bias': gain_bias,
                              'phase_bias_deg': phase_bias_deg,
                              'position_bias': position_bias,
                              'mc_mag_angle': mc_mag_angle,
                              'ProxCov_epsilon': ProxCov_epsilon,
                              'StructCovMLE_epsilon': StructCovMLE_epsilon,
                              'StructCovMLE_max_iter': StructCovMLE_max_iter,
                              't0': t0,
                              't': t
                              }
                    result.update(method_mse)
                    result.update({f'suc_{k}': v for k,v in method_success.items()})
                    result.update({f'adoa_{k}': v for k,v in method_accu_doa.items()})
                    result.update({f'bias_{k}': v for k,v in method_bias.items()})
                    scipy.io.savemat(result_path+'.mat', result)
                    np.save(result_path+'.npy', result)
                    tqdm.write(f'{t} [performance.py] Results saved at {result_path}.npy and {result_path}.mat')
                    if prev_result_path is not None and os.path.exists(prev_result_path+'.npy'):
                        os.remove(prev_result_path+'.npy')
                        os.remove(prev_result_path+'.mat')
                        tqdm.write(f'{t} [performance.py] Intermediate results at {prev_result_path}.npy and {prev_result_path}.mat were removed')
                    prev_result_path = result_path