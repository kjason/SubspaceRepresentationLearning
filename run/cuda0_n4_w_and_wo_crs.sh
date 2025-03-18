#!/bin/bash

# standard assumptions, 4-element MRA, w/ consistent rank sampling vs. w/o

# batch size 4096: 172.30 vs. 356.67 seconds/epoch (w crs vs. w/o crs using grouping): Best validation loss: 2.1322e-01 vs. 2.1312e-01 (w crs vs. w/o crs using grouping)
python3 main.py --train_L 200 --val_L 60 --batch_size 4096 --val_batch_size 4096 --print_every_n_batch 10000 --loss SignalSubspaceDistNoCrsGroup --mu 0.1 --consistent_rank_sampling 0 --N_sensors 4 --n_sources_train 1 2 3 4 5 6 --n_sources_val 1 2 3 4 5 6 --min_sep 3 3 3 3 3 3 --model N4_M7_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2
python3 main.py --train_L 200 --val_L 60 --batch_size 4096 --val_batch_size 4096 --print_every_n_batch 10000 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1 --N_sensors 4 --n_sources_train 1 2 3 4 5 6 --n_sources_val 1 2 3 4 5 6 --min_sep 3 3 3 3 3 3 --model N4_M7_WRN_16_8 --gain_bias 0.0 0.2 0.2 0.2 -0.2 -0.2 -0.2 --phase_bias_deg 0 -30 -30 -30 30 30 30 --position_bias 0.0 -0.2 -0.2 -0.2 0.2 0.2 0.2