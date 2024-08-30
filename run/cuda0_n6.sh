#!/bin/bash

# standard assumptions, 6-element MRA

python3 main.py --train_L 200 --loss ToepSquare --model N6_M14_toep_WRN_16_8 --mu 0.05 --save_dataset 1 --N_sensors 6 --n_sources_train 1 2 3 4 5 6 7 8 9 10 11 12 13 --n_sources_val 1 2 3 4 5 6 7 8 9 10 11 12 13 --min_sep 3 3 3 3 3 3 3 3 3 3 3 3 3 --gain_bias 0.0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 -30 -30 -30 -30 30 30 30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 0.2 0.2 0.2 0.2 0.2 0.2

python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.01 --N_sensors 6 --n_sources_train 1 2 3 4 5 6 7 8 9 10 11 12 13 --n_sources_val 1 2 3 4 5 6 7 8 9 10 11 12 13 --min_sep 3 3 3 3 3 3 3 3 3 3 3 3 3 --model N6_M14_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 -30 -30 -30 -30 30 30 30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 0.2 0.2 0.2 0.2 0.2 0.2

python3 main.py --train_L 200 --loss AffInvDist3 --mu 0.005 --N_sensors 6 --n_sources_train 1 2 3 4 5 6 7 8 9 10 11 12 13 --n_sources_val 1 2 3 4 5 6 7 8 9 10 11 12 13 --min_sep 3 3 3 3 3 3 3 3 3 3 3 3 3 --model N6_M14_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 -30 -30 -30 -30 30 30 30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 0.2 0.2 0.2 0.2 0.2 0.2

python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1 --N_sensors 6 --n_sources_train 1 2 3 4 5 6 7 8 9 10 11 12 13 --n_sources_val 1 2 3 4 5 6 7 8 9 10 11 12 13 --min_sep 3 3 3 3 3 3 3 3 3 3 3 3 3 --model N6_M14_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 -30 -30 -30 -30 30 30 30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 0.2 0.2 0.2 0.2 0.2 0.2

# imperfect arrays, 6-element MRA, top 2

python3 main.py --train_L 200 --loss AffInvDist3 --mu 0.005 --rho 1.0 --mix_rho 1 --save_dataset 1 --N_sensors 6 --n_sources_train 1 2 3 4 5 6 7 8 9 10 11 12 13 --n_sources_val 1 2 3 4 5 6 7 8 9 10 11 12 13 --min_sep 3 3 3 3 3 3 3 3 3 3 3 3 3 --model N6_M14_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 -30 -30 -30 -30 30 30 30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 0.2 0.2 0.2 0.2 0.2 0.2

python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1 --rho 1.0 --mix_rho 1  --N_sensors 6 --n_sources_train 1 2 3 4 5 6 7 8 9 10 11 12 13 --n_sources_val 1 2 3 4 5 6 7 8 9 10 11 12 13 --min_sep 3 3 3 3 3 3 3 3 3 3 3 3 3 --model N6_M14_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 -30 -30 -30 -30 30 30 30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 0.2 0.2 0.2 0.2 0.2 0.2