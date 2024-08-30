#!/bin/bash

# standard assumptions, 5-element MRA

python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.05 --save_dataset 1

python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.01

python3 main.py --train_L 200 --loss AffInvDist --mu 0.005

python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1

# imperfect arrays, 5-element MRA, top 2

python3 main.py --train_L 200 --loss AffInvDist --mu 0.005 --rho 1.0 --mix_rho 1 --save_dataset 1

python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1 --rho 1.0 --mix_rho 1

# gridless end-to-end approach
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.2 --consistent_rank_sampling 1