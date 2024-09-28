source ~/.bashrc
mamba activate zeo_bind

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
model_dir=$repo_dir/data/runs
preds_dir=$repo_dir/data/predictions
cd $parent_dir 

task=all
folder=$preds_dir/example
python zeobind/src/predict.py \
    --binary_models $model_dir/mlp/binary/2 \
    --binary_model_type mlp_classifier \
    --energy_models $model_dir/mlp/energy/2 \
    --energy_model_type mlp_regressor \
    --mclass_models $model_dir/mlp/mclass/2 \
    --mclass_model_type mlp_classifier \
    --device 0 \
    --num_processes 4 \
    --osda_prior_folders null \
    --osda_prior_files example_osda \
    --zeolite_prior_folders null \
    --zeolite_prior_files example_zeolite \
    --full_matrix \
    --task $task \
    --condense \
    --output $preds_dir/$folder/ \
    --batch_size 256 \
