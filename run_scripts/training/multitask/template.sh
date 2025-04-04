source ~/.bashrc
# source ~/.zshrc # For Apple M1. Also change device (below) to mps
mamba activate zeobind

# If using sigopt, set the SIGOPT_API_TOKEN environment variable in .bashrc

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
run_output_dir=$data_dir/runs
cd $parent_dir 

task=multitask
run=0
split=1
binary_weight=1.0
energy_weight=1.0
load_weight=1.0
# Maintain relative weighting of the 2 loss terms for loading prediction as 0.1
load_weight_2=$(printf "%.1f" "$(echo "$load_weight / 10.0" | bc -l)")
echo "Loss weights: $binary_weight $energy_weight $load_weight $load_weight_2"

python zeobind/src/train.py \
    --output $run_output_dir/template/$task/$run \
    --seed 12934 \
    --device "cuda:1" \
    --osda_prior_file $data_dir/datasets/training_data/osda_priors_0.pkl \
    --zeolite_prior_file $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --truth $data_dir/datasets/training_data/training_data.csv \
    --split_by osda \
    --split_folder $data_dir/datasets/training_data/splits/$split/ \
    --trainer_type mlp \
    --model_type mlp_multitask \
    --ip_scaler standard \
    --op_scaler standard \
    --optimizer adam \
    --epochs 2000 \
    --batch_size 256 \
    --early_stopping \
    --patience 10 \
    --min_delta 0.05 \
    --scaler standard \
    --loss_1 celoss \
    --weight_1 $binary_weight \
    --loss_2 mse \
    --weight_2 $energy_weight \
    --loss_3 celoss \
    --weight_3 $load_weight \
    --loss_4 mse \
    --weight_4 $load_weight_2 \
    --lr 0.00005 \
    --input_length 35 \
    --layers 4 \
    --neurons 256 \
    --dropout 0.2 \
    --class_op_size 256 \
    --binary_l_sizes 256 512 512 \
    --binary_dropout 0.4 \
    --energy_l_sizes 256 256 256 256 256 \
    --energy_dropout 0.2 \
    --load_l_sizes 256 1024 1024 \
    --load_dropout 0.5 \
    --task $task \
    --lr_patience 20 \
    --shuffle_batch \
    --save_model \
    --save \
    >> $logfile 2>&1
