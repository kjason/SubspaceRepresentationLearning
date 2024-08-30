"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen

Modified from https://github.com/kjason/DnnNormTimeFreq4DoA/tree/main/SpeechEnhancement
"""
import argparse
from datetime import datetime
from data import CovMapDataset
from train import TrainParam,TrainRegressor
from utils import dir_path, check_device
from models import model_dict
from loss import loss_dict, is_EnEnH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DNN model to estimate the covariance matrix of the corresponding ULA from a sample covariance of an MRA',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_resume_group = parser.add_mutually_exclusive_group()
    parser_resume_group.add_argument('--resume', dest='resume', action='store_true', help='resume from the last checkpoint',default=True)
    parser_resume_group.add_argument('--no-resume', dest='noresume', action='store_true', help='start a new training or overwrite the last one',default=False)
    parser.add_argument('--checkpoint_folder',default='./checkpoint/', type=dir_path, help='path to the checkpoint folder')
    parser.add_argument('--device', default='cuda:0', type=check_device, help='specify a CUDA or CPU device, e.g., cuda:0, cuda:1, or cpu')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD','AdamW'], help='optimizer')
    parser.add_argument('--mu', default=0.5, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.5, type=float, help='momentum')
    parser_nesterov_group = parser.add_mutually_exclusive_group()
    parser_nesterov_group.add_argument('--nesterov', dest='nesterov', action='store_true', help='enable Nesterov momentum',default=True)
    parser_nesterov_group.add_argument('--no-nesterov', dest='nonesterov', action='store_true', help='disable Nesterov momentum',default=False)
    parser.add_argument('--batch_size', default=4096, type=int, help='training batch size')
    parser.add_argument('--val_batch_size', default=4096, type=int, help='validation batch size')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--mu_scale', default=[1.0], nargs='+', type=float, help='learning rate scaling')
    parser.add_argument('--mu_epoch', default=[50], nargs='+', type=int, help='epochs to scale the learning rate (the last element is the total number of epochs)')
    parser.add_argument('--milestone', default=[5,10,20,30,40,50,80,100,150], nargs='+', type=int, help='the model trained after these epochs will be saved')
    parser.add_argument('--print_every_n_batch', default=10000, type=int, help='print the training status every n batch')
    parser.add_argument('--seed_list', default=[0], nargs='+', type=int, help='train models with different random seeds')
    parser.add_argument('--model', default='N5_M10_WRN_16_8', choices=list(model_dict.keys()), help='the DNN model')
    parser.add_argument('--loss', default='SignalSubspaceDist', choices=list(loss_dict.keys()), help='loss function')
    parser.add_argument('--train_L', default=200, type=int, help='train_L*base_L training datapoints for every number of sources')
    parser.add_argument('--val_L', default=60, type=int, help='val_L*base_L validation datapoints for every number of sources')
    parser.add_argument('--base_L', default=10000, type=int, help='base number of datapoints')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers of the dataloader')
    parser.add_argument('--snr_range', default=[-10,20], nargs='+', type=float, help='SNR range')
    parser.add_argument('--snr_uniform', default=0, type=int, help='1 or 0. (1): uniformly sample from the snr_range, (0): use the specified snr_list and snr_prob')
    parser.add_argument('--snr_list', default=[i for i in range(-11,23,2)], nargs='+', type=int, help='List of SNRs for training and validation')
    parser.add_argument('--snr_prob_ratio', default=1, type=float, help='the ratio given by snr_prob(last)/snr_prob(first) where snr_prob increases/descreases linearly')
    parser.add_argument('--N_sensors', default=5, type=int, help='N-element MRA')
    parser.add_argument('--deg_range', default=[30,150], nargs='+', type=float, help='DoA estimation range in degrees (0 to 180)')
    parser.add_argument('--min_sep', default=[3,3,3,3,3,3,3,3,3], nargs='+', type=float, help='List of minimum separations in degrees for the n_sources_train/val (must be a positive number)')
    parser.add_argument('--T_snapshots', default=50, type=int, help='T snapshots')
    parser.add_argument('--n_sources_train', default=[1,2,3,4,5,6,7,8,9], nargs='+', type=int, help='Number of sources for training')
    parser.add_argument('--n_sources_val', default=[1,2,3,4,5,6,7,8,9], nargs='+', type=int, help='Number of sources for validation')
    parser.add_argument('--diag_src_cov', default=1, type=int, help='1 or 0. target is (1): the diagonal sample covariance matrix, (0): sample covariance matrix')
    parser.add_argument('--use_variance', default=0, type=int, help='Use the covariance (1) or diagonal sample covariance (0) for the target (only effective if diag_src_cov=1)')
    parser.add_argument('--dynamic', default=0, type=int, help='1 or 0. (1): dynamically generate training data, (0): generate a fixed training dataset')
    parser.add_argument('--consistent_rank_sampling', default=0, type=int, help='1 or 0. (1): use ConsistentRankBatchSampler, (0): use the default random sampling')
    parser.add_argument('--fp16', default=0, type=int, help='1 or 0. (1): use mixed precision training float16 and float32, (0): use the default float32')
    parser.add_argument('--onecycle', default=1, type=int, help='1 or 0. (1): use OneCycleLR, (0): use LambdaLR')
    parser.add_argument('--normalization', default='disabled', choices=['disabled','max','sensors'], help='how to normalize the covariance matrix')
    parser.add_argument('--random_power', default=0, type=int, help='1 or 0. (1): random source power, (0): equal source power')
    parser.add_argument('--power_range', default=[0.1,1.0], nargs='+', type=float, help='range of the random power')
    parser.add_argument('--total_power_one', default=0, type=int, help='1 or 0. (1): normalize the power of sources such that the total source power is one, (0): no normalization')
    parser.add_argument('--d', default=0.01, type=float, help='sensor spacing')
    parser.add_argument('--lam', default=0.02, type=float, help='wavelength lambda')
    parser.add_argument('--gain_bias', default=[0.0,0.2,0.2,0.2,0.2,0.2,-0.2,-0.2,-0.2,-0.2], nargs='+', type=float, help='Gain bias')
    parser.add_argument('--phase_bias_deg', default=[0,-30,-30,-30,-30,-30,30,30,30,30], nargs='+', type=float, help='Phase bias in degrees')
    parser.add_argument('--position_bias', default=[0.0,-0.2,-0.2,-0.2,-0.2,-0.2,0.2,0.2,0.2,0.2], nargs='+', type=float, help='Position bias')
    parser.add_argument('--mc_mag_angle', default=[0.3,60], nargs='+', type=float, help='magnitude and phase (in degrees) of the mutual coupling coefficient')
    parser.add_argument('--rho', default=0.0, type=float, help='A number in [0,1] describing the degree of array imperfections')
    parser.add_argument('--mix_rho', default=0, type=int, help='1 or 0. (1): mix different rhos in [0,rho], (0): use the fixed given rho')
    parser.add_argument('--save_dataset', default=0, type=int, help='1 or 0. (1): save the datasets, (0): not saving')

    args = parser.parse_args()

    train_seed = 1000
    val_seed = 2000 # must be different from the train_seed

    save_dataset = bool(args.save_dataset)

    d = args.d
    lam = args.lam
    N_sensors = args.N_sensors
    T_snapshots = args.T_snapshots
    train_num_sources = args.n_sources_train
    validation_num_sources = args.n_sources_val
    snr_range = args.snr_range
    snr_uniform = bool(args.snr_uniform)
    snr_list = args.snr_list
    snr_prob_ratio = args.snr_prob_ratio
    snr_prob_inc = ((1+snr_prob_ratio) * len(snr_list))/2
    snr_prob = [1/snr_prob_inc+(i*(snr_prob_ratio-1)/(snr_prob_inc*(len(snr_list)-1))) for i in range(len(snr_list))]
    deg_range = args.deg_range
    min_sep = args.min_sep
    train_L = args.train_L
    base_L = args.base_L
    val_L = args.val_L
    diag_src_cov = bool(args.diag_src_cov)
    use_variance = bool(args.use_variance)
    dynamic = bool(args.dynamic)
    consistent_rank_sampling = bool(args.consistent_rank_sampling)
    fp16 = bool(args.fp16)
    onecycle = bool(args.onecycle)
    normalization = args.normalization
    random_power = bool(args.random_power)
    power_range = args.power_range
    total_power_one = bool(args.total_power_one)
    optimizer = args.optimizer
    gain_bias = args.gain_bias
    phase_bias_deg = args.phase_bias_deg
    position_bias = args.position_bias
    mc_mag_angle = args.mc_mag_angle
    rho = args.rho
    mix_rho = bool(args.mix_rho)

    if len(min_sep) != len(train_num_sources):
        raise ValueError(f"len(min_sep)={len(min_sep)} does not match len(num_sources_list)={len(train_num_sources)}")

    trainset = CovMapDataset(mode='train',L=train_L,d=d,lam=lam,N_sensors=N_sensors,T_snapshots=T_snapshots,num_sources=train_num_sources,
                             snr_range=snr_range,snr_uniform=snr_uniform,snr_list=snr_list,snr_prob=snr_prob,seed=train_seed,deg_range=deg_range,
                             min_sep=min_sep,diag_src_cov=diag_src_cov,use_variance=use_variance,gain_bias=gain_bias,phase_bias_deg=phase_bias_deg,
                             position_bias=position_bias,mc_mag_angle=mc_mag_angle,rho=rho,mix_rho=mix_rho,base_L=base_L,dynamic=dynamic,
                             random_power=random_power,power_range=power_range,total_power_one=total_power_one,normalization=normalization,
                             device='cpu',save_dataset=save_dataset)
    
    validationset = CovMapDataset(mode='validation',L=val_L,d=d,lam=lam,N_sensors=N_sensors,T_snapshots=T_snapshots,num_sources=validation_num_sources,
                                  snr_range=snr_range,snr_uniform=snr_uniform,snr_list=snr_list,snr_prob=snr_prob,seed=val_seed,deg_range=deg_range,
                                  min_sep=min_sep,diag_src_cov=diag_src_cov,use_variance=use_variance,gain_bias=gain_bias,phase_bias_deg=phase_bias_deg,
                                  position_bias=position_bias,mc_mag_angle=mc_mag_angle,rho=rho,mix_rho=mix_rho,base_L=base_L,dynamic=False,
                                  random_power=random_power,power_range=power_range,total_power_one=total_power_one,normalization=normalization,
                                  device='cpu',save_dataset=save_dataset)

    criterion = loss_dict[args.loss]

    for seed in args.seed_list:

        name = (f"{args.model}_t={train_L}_v={val_L}_n={N_sensors}_loss={args.loss}_mu={args.mu}_mo={args.momentum}_bs={args.batch_size}_epoch={args.mu_epoch[-1]}"
                f"_wd={args.weight_decay}_seed={seed}_nsrc={str(train_num_sources)}_T={T_snapshots}_rg={str(deg_range)}_sep={str([int(s) for s in min_sep])}_rho={rho}_mix={args.mix_rho}"
                f"_snr={str(snr_range)}_uni={args.snr_uniform}_spr={round(snr_prob_ratio,1)}_dg={args.diag_src_cov}_uv={args.use_variance}"
                f"_crs={args.consistent_rank_sampling}_dy={args.dynamic}_rp={args.random_power}_pr={str(power_range)}_tpo={int(total_power_one)}_nor={normalization}"
                f"_oc={args.onecycle}").replace(' ','').replace('.','').replace(',','').replace('[','').replace(']','')
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [main.py] start the training task {name}")

        array_setting = {'N_sensors': N_sensors, 'd': d, 'lam': lam, 'normalization': normalization, 'model': args.model,'EnEnH': is_EnEnH(args.loss)}

        tp = TrainParam(
            mu = args.mu,
            mu_scale = args.mu_scale,
            mu_epoch = args.mu_epoch,
            weight_decay = args.weight_decay,
            momentum = args.momentum,
            batch_size = args.batch_size,
            val_batch_size = args.val_batch_size,
            nesterov = args.nesterov and not args.nonesterov,
            onecycle = onecycle,
            optimizer = optimizer
            )
        
        r = TrainRegressor(
            name = name,
            net = model_dict[args.model],
            tp = tp,
            trainset = trainset,
            validationset = validationset,
            criterion = criterion,
            device = args.device,
            seed = seed,
            resume = args.resume and not args.noresume,
            checkpoint_folder = args.checkpoint_folder,
            num_workers = args.num_workers,
            consistent_rank_sampling = consistent_rank_sampling,
            milestone = args.milestone,
            print_every_n_batch = args.print_every_n_batch,
            fp16 = fp16,
            meta_data = array_setting
        ).train()

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [main.py] training task {name} is completed\n")