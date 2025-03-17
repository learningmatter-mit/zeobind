source ~/.bashrc
# source ~/.zshrc # For Apple M1. Also change device (below) to mps
mamba activate zeobind

# If using sigopt, set the SIGOPT_API_TOKEN environment variable in .bashrc

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
run_output_dir=$repo_dir/data/runs
cd $parent_dir 

task=loading_classification
run=0
split=1
python zeobind/src/train.py \
    --output $run_output_dir/template/$task/$run \
    --seed 12934 \
    --device "cuda" \
    --osda_prior_file $data_dir/datasets/training_data/osda_priors_0.pkl \
    --zeolite_prior_file $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --truth $data_dir/datasets/training_data/training_data.csv \
    --split_by osda \
    --split_folder $data_dir/datasets/training_data/splits/$split/ \
    --trainer_type mlp \
    --model_type mlp_classifier \
    --ip_scaler standard \
    --optimizer adam \
    --epochs 1 \
    --batch_size 256 \
    --patience 10 \
    --min_delta 0.05 \
    --loss_1 celoss \
    --weight_1 1.0 \
    --loss_2 mse \
    --weight_2 0.1 \
    --lr 0.0001 \
    --input_length 35 \
    --layers 2 \
    --neurons 512 \
    --dropout 0.4 \
    --num_classes 46 \
    --task $task \
    --lr_patience 20 \
    --shuffle_batch \
    --save_model \
    --save \
