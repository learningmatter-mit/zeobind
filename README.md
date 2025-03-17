# zeobind

This repository contains code for the computational screening workflow used in the paper "Exhaustive mapping of the zeolite-template chemical space" by Xie et. al. If you use our code and data, please cite our cite our manuscript: 

```
@misc{xie_exhaustive_2024,
	title = {An exhaustive mapping of zeolite-template chemical space},
	copyright = {https://creativecommons.org/licenses/by-nc/4.0/},
	url = {https://chemrxiv.org/engage/chemrxiv/article-details/66f8658812ff75c3a1cb235d},
	doi = {10.26434/chemrxiv-2024-d74sw},
	language = {en},
	urldate = {2024-10-10},
	author = {Xie, Mingrou and Schwalbe-Koda, Daniel and Semanate-Esquivel, Yolanda M. and Bello-Jurado, Estafanía and Hoffman, Alexander and Santiago-Reyes, Omar and Paris, Cecilia and Moliner, Manuel and Gómez-Bombarelli, Rafael},
	month = oct,
	year = {2024},
}
```

## Installation

To install the code, clone the repository and install the dependencies using the following commands:

```bash
mamba create -f environment.yml
mamba activate zeobind
```

For Apple Silicon systems, install `mamba` first with homebrew, then run `bash environment_m1_install.sh`. Note that there might be some libraries with different versions as this repo was originally developed on Linux. Furthermore, when running the bash scripts described below for various tasks, `source ~/.zshrc` should be added to the top of the bash script, and the `device` flag should be changed to `mps`.

The data used in the paper, dataset of hypothetical molecules, as well as predictions across all the IZA framework - hypothetical molecule chemical space, can be found at [Materials Data Facility](https://materialsdatafacility.org/detail/2cd2e3f3-12d3-4cd9-8ef0-efd91c0f8e3a-1.0?type=dataset). Download the data and place it in the `zeobind/data` directory.

## Usage 


### Screening molecules from predictions 

We provide a bash script for screening for molecules for a targeted framework using the pre-generated predictions stored inside `data/predictions`. 

Example bash scripts, which are also the ones used in the paper's CHA and ERI case studies, can be found at `run_scripts/inference/screen_{cha,eri}.sh`. Run

```bash
bash run_scripts/inference/screen_cha.sh
```

### Predicting on new molecules

If you have a new set of molecules you would like to predict on, an example bash script can be found at `run_scripts/predict.sh`.

```bash
bash run_scripts/inference/predict.sh
```


### Training models 

To train models, template bash scripts can be found at `run_scripts/training/`.

```bash
# example
bash run_scripts/training/binary/template.sh
```

### Reproducing results and figures 

The notebooks in the `notebooks` directory generate metrics and figures reported in the paper. 