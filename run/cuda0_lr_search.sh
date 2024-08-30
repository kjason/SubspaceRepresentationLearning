#!/bin/bash

# search the best maximum learning rate

python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 2.0 --save_dataset 1
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 1.0
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.5
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.2
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.1
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.05
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.02
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.01
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.005
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.002
python3 main.py --train_L 200 --loss ToepSquare --model N5_M10_toep_WRN_16_8 --mu 0.001

python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.5
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.2
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.1
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.05
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.02
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.01
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.005
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.002
python3 main.py --train_L 200 --loss FrobeniusNorm --mu 0.001

python3 main.py --train_L 200 --loss AffInvDist --mu 0.2
python3 main.py --train_L 200 --loss AffInvDist --mu 0.1
python3 main.py --train_L 200 --loss AffInvDist --mu 0.05
python3 main.py --train_L 200 --loss AffInvDist --mu 0.02
python3 main.py --train_L 200 --loss AffInvDist --mu 0.01
python3 main.py --train_L 200 --loss AffInvDist --mu 0.005
python3 main.py --train_L 200 --loss AffInvDist --mu 0.002
python3 main.py --train_L 200 --loss AffInvDist --mu 0.001
python3 main.py --train_L 200 --loss AffInvDist --mu 0.0005

python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 2.0 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 1.0 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.5 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.2 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.05 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.02 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.01 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.005 --consistent_rank_sampling 1

python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 1.0 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.5 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.2 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.1 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.05 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.02 --consistent_rank_sampling 1
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.01 --consistent_rank_sampling 1