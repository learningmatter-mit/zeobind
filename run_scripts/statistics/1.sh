#!/bin/bash

source ~/.bashrc
mamba activate zeo_bind

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
run_output_dir=$repo_dir/data/runs
cd $parent_dir 

hyp_mols_pred_dir=$data_dir/predictions/hyp_mols

python zeobind/src/utils/statistics.py \
    --pred_dir $hyp_mols_pred_dir/diq \
    --op_dir $data_dir/publication/statistics/diq \
    --charge diq \


python zeobind/src/utils/statistics.py \
    --pred_dir $hyp_mols_pred_dir/mono \
    --op_dir $data_dir/publication/statistics/mono \
    --charge monoq \