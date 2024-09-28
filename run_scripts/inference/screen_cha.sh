source ~/.bashrc
mamba activate zeo_bind

parent_dir=~/projects/affinity 
repo_dir=$parent_dir/zeobind
data_dir=$repo_dir/data
model_dir=$repo_dir/data/runs
preds_dir=$repo_dir/data/predictions
filtered_dir=$repo_dir/data/filtered
cd $parent_dir 

fw=CHA
python zeobind/src/screen.py \
    --parallel \
    --num_threads 24 \
    --preds_dir $preds_dir/hyp_mols/diq \
    --opriors_dir hyp_mols_data/diq \
    --ofile_root osda_priors \
    --zfile_root zeolite_priors \
    --output $filtered_dir/$fw \
    --charge diq \
    --fw $fw \
    --be_cutoff -10.0 \
    --ce_cutoff 4.0 \
    --vol_lower 100 \
    --vol_upper 250 \
    --rot_lower 0 \
    --rot_upper 5 \
    --sim_cutoff 0.8 \
    --sa_cutoff 5 \
    --px_cutoff 50000 \
    --remove_aromatic_n \
    --remove_3mr \
    --remove_4mr \
    --remove_double_bond \
    --remove_neighboring_n \
    --remove_stereocenters \
    --num_c_between_n_min 2 \
