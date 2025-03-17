source ~/.bashrc
# source ~/.zshrc # For Apple M1. Also change device (below) to mps
mamba activate zeobind

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
model_dir=$repo_dir/data/runs
preds_dir=$repo_dir/data/predictions
cd $parent_dir 

binary_models=("$model_dir/mlp/binary/2/1" "$model_dir/mlp/binary/2/2" "$model_dir/mlp/binary/2/3" "$model_dir/mlp/binary/2/4" "$model_dir/mlp/binary/2/5")
energy_models=("$model_dir/mlp/energy/2/1" "$model_dir/mlp/energy/2/2" "$model_dir/mlp/energy/2/3" "$model_dir/mlp/energy/2/4" "$model_dir/mlp/energy/2/5")
mclass_models=("$model_dir/mlp/mclass/2/1" "$model_dir/mlp/mclass/2/2" "$model_dir/mlp/mclass/2/3" "$model_dir/mlp/mclass/2/4" "$model_dir/mlp/mclass/2/5")

task=all
folder=example
python zeobind/src/predict.py \
    --binary_models ${binary_models[@]} \
    --binary_model_type mlp_classifier \
    --energy_models ${energy_models[@]} \
    --energy_model_type mlp_regressor \
    --loading_models ${mclass_models[@]} \
    --loading_model_type mlp_classifier \
    --device "cuda:1" \
    --num_processes 1 \
    --osda_prior_folders null \
    --osda_prior_files $repo_dir/run_scripts/inference/example_osda \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_folders null \
    --zeolite_prior_files $repo_dir/run_scripts/inference/example_zeolite \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --full_matrix \
    --task $task \
    --output $preds_dir/$folder/ \
    --batch_size 256 \
    --condense \
