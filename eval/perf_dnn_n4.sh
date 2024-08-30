#!/bin/bash

DCRT=./checkpoint/N4_M7_toep_WRN_16_8_t=200_v=60_n=4_loss=ToepSquare_mu=005_mo=05_bs=4096_epoch=50_wd=00_seed=0_nsrc=123456_T=50_rg=30150_sep=333333_rho=00_mix=0_snr=-1020_uni=0_spr=1_dg=1_uv=0_crs=0_dy=0_rp=0_pr=0110_tpo=0_nor=disabled_oc=1
DCRGF=./checkpoint/N4_M7_WRN_16_8_t=200_v=60_n=4_loss=FrobeniusNorm_mu=001_mo=05_bs=4096_epoch=50_wd=00_seed=0_nsrc=123456_T=50_rg=30150_sep=333333_rho=00_mix=0_snr=-1020_uni=0_spr=1_dg=1_uv=0_crs=0_dy=0_rp=0_pr=0110_tpo=0_nor=disabled_oc=1
DCRGA=./checkpoint/N4_M7_WRN_16_8_t=200_v=60_n=4_loss=AffInvDist_mu=0005_mo=05_bs=4096_epoch=50_wd=00_seed=0_nsrc=123456_T=50_rg=30150_sep=333333_rho=00_mix=0_snr=-1020_uni=0_spr=1_dg=1_uv=0_crs=0_dy=0_rp=0_pr=0110_tpo=0_nor=disabled_oc=1
OURS=./checkpoint/N4_M7_WRN_16_8_t=200_v=60_n=4_loss=SignalSubspaceDist_mu=01_mo=05_bs=4096_epoch=50_wd=00_seed=0_nsrc=123456_T=50_rg=30150_sep=333333_rho=00_mix=0_snr=-1020_uni=0_spr=1_dg=1_uv=0_crs=1_dy=0_rp=0_pr=0110_tpo=0_nor=disabled_oc=1

MIXRHO_DCRGA=./checkpoint/N4_M7_WRN_16_8_t=200_v=60_n=4_loss=AffInvDist_mu=0005_mo=05_bs=4096_epoch=50_wd=00_seed=0_nsrc=123456_T=50_rg=30150_sep=333333_rho=10_mix=1_snr=-1020_uni=0_spr=1_dg=1_uv=0_crs=0_dy=0_rp=0_pr=0110_tpo=0_nor=disabled_oc=1
MIXRHO_OURS=./checkpoint/N4_M7_WRN_16_8_t=200_v=60_n=4_loss=SignalSubspaceDist_mu=01_mo=05_bs=4096_epoch=50_wd=00_seed=0_nsrc=123456_T=50_rg=30150_sep=333333_rho=10_mix=1_snr=-1020_uni=0_spr=1_dg=1_uv=0_crs=1_dy=0_rp=0_pr=0110_tpo=0_nor=disabled_oc=1

# SNR vs. MSE
python3 performance.py --results_folder dnn_results --N_sensors 4 --num_sources_list 1 2 3 4 5 6 --min_sep 4 4 4 4 4 4 --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2 --cov_models $DCRT $DCRGF $DCRGA $OURS

# number of snapshots vs. MSE
python3 performance.py --results_folder dnn_results --N_sensors 4 --num_sources_list 1 4 6 --min_sep 4 4 4 --SNR_list 20 --T_snapshots_list 10 20 30 40 50 60 70 80 90 100 --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2 --cov_models $DCRT $DCRGF $DCRGA $OURS

# array imperfection parameter rho vs. MSE
for i in 0.0 0.1 0.2 0.5 1.0
do
    python3 performance.py --results_folder dnn_results --N_sensors 4 --num_sources_list 1 4 6 --min_sep 4 4 4 --SNR_list 20 --rho $i --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2 --cov_models $MIXRHO_DCRGA $MIXRHO_OURS
done