# zeobind

This repository contains code for the computational screening workflow used in the paper "Exhaustive mapping of the zeolite-template chemical space" by Xie et. al. If you use our code and data, please cite our paper: [TODO insert paper citation]

## Installation

To install the code, clone the repository and install the dependencies using the following commands:

```bash
mamba create -f environment.yml
mamba activate zeobind
```

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