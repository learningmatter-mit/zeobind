source ~/.bashrc
mamba activate zeo_bind

# If using sigopt, set the SIGOPT_API_TOKEN environment variable in .bashrc

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
run_output_dir=$repo_dir/data/runs
cd $parent_dir 

task=binary
run=0
split=0
python zeobind/src/train.py \
    --output $run_output_dir/$task/$run/ \
    --seed 12934 \
    --device 2 \
    --osda_prior_file $data_dir/datasets/training_data/osda_priors_0.pkl \
    --zeolite_prior_file $data_dir/datasets/training_data/zeolite_priors_0.pkl \
    --osda_prior_map $repo_dir/src/configs/osda_v1_phys.json \
    --zeolite_prior_map $repo_dir/src/configs/zeolite_v1_phys_short.json \
    --truth $data_dir/datasets/training_data/training_data.csv \
    --split_by smiles \
    --split_folder $data_dir/datasets/training_data/splits/$split/ \
    --model_type nn \
    --input_scaler standard \
    --optimizer adam \
    --epochs 500 \
    --batch_size 256 \
    --patience 10 \
    --min_delta 0.05 \
    --loss_1 celoss \
    --lr 0.0001 \
    --layers 2 \
    --neurons 512 \
    --batch_norm false \
    --softmax false \
    --dropout 0.4 \
    --num_classes 2 \
    --task $task \
    --scheduler false \
    --lr_patience 20 \
    --early_stopping false \
    --shuffle_batch true \
    --save_truths false \
    --save_preds false \
    --save_ips false \
    --save_mask false \
