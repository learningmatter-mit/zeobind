source ~/.bashrc
# source ~/.zshrc # For Apple M1. Also change device (below) to mps
mamba activate zeobind

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
model_dir=$data_dir/runs
preds_dir=$data_dir/predictions
cd $parent_dir 

energy_models=("$model_dir/mlp/multitask/1/0") 

task=multitask
folder=multitask
python zeobind/src/predict.py \
    --multitask_models ${energy_models[@]} \
    --multitask_model_type mlp_multitask \
    --model_file "model_final.pt" \
    --device "cuda:2" \
    --num_processes 1 \
    --osda_prior_folders null \
    --osda_prior_files $data_dir/datasets/heldout_hyp_mols/osda_priors_0.pkl \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_folders null \
    --zeolite_prior_files $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --full_matrix \
    --task $task \
    --output $preds_dir/$folder/ \
    --batch_size 256 \
    --condense \
