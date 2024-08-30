# Subspace Representation Learning for Sparse Linear Arrays to Localize More Sources than Sensors: A Deep Learning Methodology

This repository is the official implementation of the paper submitted to the IEEE Transactions on Signal Processing, [Subspace Representation Learning for Sparse Linear Arrays to Localize More Sources than Sensors: A Deep Learning Methodology](https://arxiv.org/abs/2408.16605).

- Download the paper from [arXiv](https://arxiv.org/abs/2408.16605).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> Please ensure Python is installed before running the above setup command. The code was tested on Python 3.9.13 and 3.10.12.

If you are interested in running SDP-based baselines such as SPA and StructCovMLE, then you will need to install MATLAB because all SDP problems will be solved by the SDPT3 solver in CVX. For the implementation of all SDP-based baselines, please see the "SDP" folder.

## Training DNN models for DoA estimation

To reproduce the numerical results in the paper, we will need to train DNN models before evaluation. To train all the models in the experiment of the 4-element MRA, run:

```train
bash run/cuda0_n4.sh
```

To replicate results for the 5-element and 6-element MRAs, simply run `bash run/cuda0_n5.sh` and `bash run/cuda0_n6.sh`.

> The best learning rate can be found by a simple grid search using `cuda0_lr_search.sh`. See Appendix C in the paper for more details about learning rates and the empirical risk on the validation set.

If one is only interested in subspace representation learning for the 5-element MRA, then one can run:

```train_subspace
python main.py --train_L 200 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1
```

> This will train a model with 200x10,000=2,000,000 data points per source number, leading to a dataset of 18,000,000 data points in total since the 5-element MRA can resolve up to 9 sources. The base of 10,000 can be configured by the option `base_L` of `main.py` and one can also specify the size of the validation set via the option `val_L`. Consistent rank sampling is enabled, the learning rate is 0.1, and the loss function is `SignalSubspaceDist` which will apply subspace representation learning.

The above command will train a model for a perfect 5-element MRA. To train a model for imperfect arrays, run:

```train_subspace_for_imperfect_arrays
python3 main.py --train_L 200 --loss SignalSubspaceDist --mu 0.1 --consistent_rank_sampling 1 --rho 1.0 --mix_rho 1
```

> Note that here we enable the option `mix_rho` to randomly select the degree of imperfections, rho, in the interval [0, 1.0], for each data point. If the option `rho` is set to 0.5, then the interval becomes [0, 0.5]. This `mix_rho` allows us to create a dataset of different imperfect arrays.

To train a model for the gridless end-to-end approach, run:

```train_gridless_end2end_for_imperfect_arrays
python3 main.py --train_L 200 --loss BranchAngleMSE --model Branch_N5_M10_WRN_16_8 --mu 0.2 --consistent_rank_sampling 1
```

## Evaluation

To evaluate performance of all of the SDP-based methods in the case of 5-element MRA, run:

```eval_SDP
bash eval/perf_opt_n5.sh
```

Performance evaluation can be customized by using the options of `performance.py`. For example, you can change the number of total random trials by specifying the number of random angles and the number of random trials per random angle. You can also change the minimum separation constraint, the degree of array imperfections, SNRs, numbers of sources, the methods you want to evaluate, etc. For instance, the following command evaluates the direct augmentation approach, spatial smoothing, Wasserstein distance minimization, and SPA, using a total of 10,000 random trials and a minimum separation of 4 degrees on the perfect 5-element MRA. One can even specify different minimum separations for different source numbers. See the options of `performance.py` for more details.

```eval_SDP_MRA5
python3 performance.py --DA 1 --SS 1 --Wasserstein 1 --SPA 1 --SPA_noisevar 1 --num_random_thetas 100 --trials_per_theta 100 --num_sources_list 1 2 3 4 5 6 7 8 9 --rho 0 --results_folder results --min_sep 4 4 4 4 4 4 4 4 4
```

To evaluate performance of all of the DNN-based methods in the case of 5-element MRA, run:

```eval_DNN
bash eval/perf_dnn_n5.sh
```

You can also evaluate any number of DNN-based approaches by specifying a list of paths to the models after the option `cov_models`. For example, the following command will evaluate 4 different models.

```eval_DNN_single
python3 performance.py --results_folder dnn_results --cov_models $DCRT $DCRGF $DCRGA $OURS
```

To evaluate methods on other MRAs, simply switch the script to other cases as indicated in the folder `eval`.

## BibTeX

Please feel free to cite our paper if you find this repository useful in your work.

```
@article{chen2024subspace,
  title={Subspace Representation Learning for Sparse Linear Arrays to Localize More Sources than Sensors: A Deep Learning Methodology},
  author={Chen, Kuan-Lin and Rao, Bhaskar D.},
  journal={arXiv preprint arXiv:2408.16605},
  year={2024}
}
```