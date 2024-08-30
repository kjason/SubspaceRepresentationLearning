#!/bin/bash
python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 2 3 4 5 6 7 8 9 --rho 0 --results_folder results --min_sep 4 4 4 4 4 4 4 4 4

python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 6 9 --SNR_list 20 --T_snapshots_list 10 20 30 40 50 60 70 80 90 100 --rho 0 --results_folder results --min_sep 4 4 4

python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 6 9 --SNR_list 20 --rho 0.0 --results_folder results --min_sep 4 4 4
python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 6 9 --SNR_list 20 --rho 0.1 --results_folder results --min_sep 4 4 4
python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 6 9 --SNR_list 20 --rho 0.2 --results_folder results --min_sep 4 4 4
python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 6 9 --SNR_list 20 --rho 0.5 --results_folder results --min_sep 4 4 4
python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 6 9 --SNR_list 20 --rho 1.0 --results_folder results --min_sep 4 4 4