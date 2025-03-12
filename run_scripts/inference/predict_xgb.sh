source ~/.bashrc
mamba activate zeo_bind

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
model_dir=$data_dir/runs
preds_dir=$data_dir/predictions
cd $parent_dir 

# Best models from hyperparameter tuning
binary_models=("$model_dir/xgb/binary/tune/hp_tune_run_4")
energy_models=("$model_dir/xgb/energy/tune/hp_tune_run_1")
folder="xgb"

# Predict for heldout mols 
osda_prior_files=$data_dir/datasets/heldout_hyp_mols/osda_priors_0.pkl
sub_op_folder=heldout_hyp_mols
echo "Predicting for" $sub_op_folder `date`

python zeobind/src/predict.py \
    --binary_models ${binary_models[@]} \
    --binary_model_type xgb_classifier \
    --device "cuda:1" \
    --num_processes 1 \
    --osda_prior_folders null \
    --osda_prior_files $osda_prior_files \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_folders null \
    --zeolite_prior_files $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --full_matrix \
    --task binary_classification \
    --output $preds_dir/$folder/$sub_op_folder \
    --batch_size 256 \
    --condense \

python zeobind/src/predict.py \
    --energy_models ${energy_models[@]} \
    --energy_model_type xgb_regressor \
    --device "cuda:1" \
    --num_processes 1 \
    --osda_prior_folders null \
    --osda_prior_files $osda_prior_files \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_folders null \
    --zeolite_prior_files $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --full_matrix \
    --task energy_regression \
    --output $preds_dir/$folder/$sub_op_folder \
    --batch_size 256 \
    --condense \


############################################################################################################
############################################################################################################


# Predict for training data
osda_prior_files=$data_dir/datasets/training_data/osda_priors_0.pkl
sub_op_folder=training_data
echo "Predicting for" $sub_op_folder `date`

python zeobind/src/predict.py \
    --binary_models ${binary_models[@]} \
    --binary_model_type xgb_classifier \
    --device "cuda:1" \
    --num_processes 1 \
    --osda_prior_folders null \
    --osda_prior_files $osda_prior_files \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_folders null \
    --zeolite_prior_files $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --full_matrix \
    --task binary_classification \
    --output $preds_dir/$folder/$sub_op_folder \
    --batch_size 256 \
    --condense \

python zeobind/src/predict.py \
    --energy_models ${energy_models[@]} \
    --energy_model_type xgb_regressor \
    --device "cuda:1" \
    --num_processes 1 \
    --osda_prior_folders null \
    --osda_prior_files $osda_prior_files \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_folders null \
    --zeolite_prior_files $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --full_matrix \
    --task energy_regression \
    --output $preds_dir/$folder/$sub_op_folder \
    --batch_size 256 \
    --condense \
