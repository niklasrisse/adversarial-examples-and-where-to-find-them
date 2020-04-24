# Adversarial examples and where to find them

Authors: **Niklas Risse, Christina Göpfert and Jan Philip Göpfert**

Institution: **Bielefeld University**

Paper: [https://arxiv.org/abs/2004.10882](https://arxiv.org/abs/2004.10882)


## What is in this repository?
+ The code to calculate robustness curves for a chosen model and dataset
+ The code to calculate perturbation cost trajectories for a chosen model and dataset
+ The code to reproduce all experiments (with figures) from the paper, including the following:
  + 5 experiments on inter and intra class distances for different datasets and norms
  + 5 experiments on perturbation cost trajectories for different models and datasets
  + 7 experiments on robustness curves for different models and datasets
  + 1 visualization of adversarial examples for different models and norms
  
## Main idea of the paper
<p align="center"><img src="images/readme_gif.gif" width="500"></p>
Adversarial robustness of trained models has attracted considerable attention
over recent years, within and beyond the scientific community. This is not only
because of a straight-forward desire to deploy reliable systems, but also
because of how adversarial attacks challenge our beliefs about deep neural
networks. Demanding more robust models seems to be the obvious solution --
however, this requires a rigorous understanding of how one should judge
adversarial robustness as a property of a given model. In this work, we analyze
where adversarial examples occur, in which ways they are peculiar, and how they
are processed by robust models. We use robustness curves to show that
l_infty threat models are surprisingly effective in improving robustness
for other l_p norms; we introduce perturbation cost trajectories to provide
a broad perspective on how robust and non-robust networks perceive adversarial
perturbations as opposed to random perturbations; and we explicitly examine the
scale of certain common data sets, showing that robustness thresholds must be
adapted to the data set they pertain to. This allows us to provide concrete
recommendations for anyone looking to train a robust model or to estimate how
much robustness they should require for their operation.

## How to generate robustness curves
The python script `generate_robustness_curves.py` contains methods to calculate robustness curves. You can either directly execute the script or import the methods from the file. If you directly execute the script, you can define parameters via arguments. Example of usage (estimated runtime: 4 Minutes):

`python generate_robustness_curves.py --dataset=mnist --n_points=10 --model_path='provable_robustness_max_linear_regions/models/mmr+at/2019-02-17 01:54:16 dataset=mnist nn_type=cnn_lenet_small p_norm=inf lmbd=0.5 gamma_rb=0.2 gamma_db=0.2 ae_frac=0.5 epoch=100.mat' --nn_type='cnn' --norms 2 1 inf --save=True --plot=True`

This calculates and plots the robustness curves for a model trained by Croce et al. for 10 datapoints of the MNIST test set for the l_2, l_1 and l_\infty norms.

The datasets are available in the folder `provable_robustness_max_linear_regions/datasets`. You can choose between the following: `mnist`, `fmnist`, `gts` and `cifar10`. The models are available in the folder `provable_robustness_max_linear_regions/models`. You can execute `python generate_robustness_curves.py --help` to get more information about the different arguments of the script.
## How to generate perturbation cost trajectories
The python script `generate_perturbation_cost_trajectories.py` contains methods to calculate perturbation cost trajectories. You can either directly execute the script or import the methods from the file. If you directly execute the script, you can define parameters via arguments. Example of usage (estimated runtime: 3 Minutes):

`python generate_perturbation_cost_trajectories.py --dataset=mnist --n_points=10 --model_path='provable_robustness_max_linear_regions/models/mmr+at/2019-02-17 01:54:16 dataset=mnist nn_type=cnn_lenet_small p_norm=inf lmbd=0.5 gamma_rb=0.2 gamma_db=0.2 ae_frac=0.5 epoch=100.mat' --nn_type='cnn' --norms inf 2 --noise_types gaussian --plot=True`

This calculates and plots the perturbation cost trajectories for a model trained by Croce et al. for 10 datapoints of the MNIST test set for the l_2, l_\infty norms, for adversarial and random noise of type `gaussian`.

The datasets are available in the folder `provable_robustness_max_linear_regions/datasets`. You can choose between the following: `mnist`, `fmnist`, `gts` and `cifar10`. The models are available in the folder `provable_robustness_max_linear_regions/models`. You can execute `python generate_perturbation_cost_trajectories.py --help` to get more information about the different arguments of the script.
## Installation
The model and data files in this repository are stored with git lfs. Install git lfs on Linux Ubuntu with:
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```
If your system is not based on Ubuntu, consider https://git-lfs.github.com/ for installation.
After installing git-lfs, you can just clone the repository with `git clone`, and all the files will be downloaded automatically (may take some time, based on your download speed).

We manage python dependencies with anaconda. You can find information on how to install anaconda at: https://docs.anaconda.com/anaconda/install/. After installing, create the environment with executing `conda env create` in the root directory of the repository. This automatically finds and uses the file `environment.yml`, which creates an environment called `robustness` with
everything needed to run our python files and notebooks. Activate the environment with `conda activate robustness`.

We use tensorflow-gpu 2.1 to calculate adversarial examples. To correctly set up tensorflow for your GPU, follow the instructions from: https://www.tensorflow.org/install/gpu.

We use the julia package [MIPVerify](https://github.com/vtjeng/MIPVerify.jl) with [Gurobi](https://www.gurobi.com/documentation/quickstart.html) to calculate exact minimal adversarial examples in the notebook `experiments/rob_curves_true_vs_approximative.ipynb`. To install julia, follow the instructions from: https://julialang.org/downloads/. To install gurobi, follow the instructions from  https://www.gurobi.com/documentation/quickstart.html (free academic licenses available). You need to install the following julia packages: MIPVerify, Gurobi, JuMP, Images, Printf, MAT, CSV and NPZ. More information on MIPVerify can be found here: https://vtjeng.github.io/MIPVerify.jl/latest/#Installation-1.

## Contact
If you have a problem or question regarding the code, please contact [Niklas Risse](https://github.com/niklasrisse).
## Citation
```
@misc{risse2020adversarial,
    title={Adversarial examples and where to find them},
    author={Niklas Risse and Christina Göpfert and Jan Philip Göpfert},
    year={2020},
    eprint={2004.10882},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
