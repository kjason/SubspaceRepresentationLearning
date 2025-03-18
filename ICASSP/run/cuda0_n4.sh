#!/bin/bash

# standard assumptions, 4-element MRA

python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.01 --N_sensors 4 --n_sources_train 1 2 3 4 5 6 --n_sources_val 1 2 3 4 5 6 --min_sep 3 3 3 3 3 3 --model N4_M7_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2 --save_dataset 1

python3 main.py --train_L 200 --loss SISDRFrobeniusNorm --mu 0.05 --N_sensors 4 --n_sources_train 1 2 3 4 5 6 --n_sources_val 1 2 3 4 5 6 --min_sep 3 3 3 3 3 3 --model N4_M7_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2

python3 main.py --train_L 200 --loss SignalSISDRFrobeniusNorm --mu 0.2 --N_sensors 4 --n_sources_train 1 2 3 4 5 6 --n_sources_val 1 2 3 4 5 6 --min_sep 3 3 3 3 3 3 --model N4_M7_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2