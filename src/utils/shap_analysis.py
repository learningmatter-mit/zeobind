import time
import shap
import torch
from math import floor
import os 
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from zeobind.src.models.models import get_model
from zeobind.src.utils.logger import log_msg

##########################################
##### FILE PATHS AND OUTPUT DIRECTORY ####
model_folder = "../../data/runs/mlp/energy/2/1/"
o_config_file = "../configs/osda_v1_phys.json"
z_config_file = "../configs/zeolite_v1_phys_short.json"

split_dir = "../../data/datasets/training_data/splits/1/"
truth_file = "../../data/datasets/training_data/training_data.csv"
oprior_file = "../../data/datasets/training_data/osda_priors_0.pkl"
zprior_file = "../../data/datasets/training_data/zeolite_priors_0.pkl"

op_dir = "../../data/publication/shap/split_1/"
##########################################
##########################################


# get model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = dict() 
kwargs["saved_model"] = model_folder
model, _ = get_model("mlp_regressor", kwargs, device=device)

# get feature column names
with open(o_config_file, "r") as f:
    o_X_cols = json.load(f).keys()
with open(z_config_file, "r") as f:
    z_X_cols = json.load(f).keys()

# read prior files to make X. Also for sampling background dataset with
opriors = pd.read_pickle(oprior_file)
zpriors = pd.read_pickle(zprior_file)

# read truth to get index to make X
truth = pd.read_csv(truth_file)
truth = truth[['Zeolite', 'SMILES', 'Binding (SiO2)']]

log_msg("truth:", truth.shape)
truth = truth[truth['Binding (SiO2)'].lt(0.0)] 
log_msg("truth:", truth.shape)

o = opriors.loc[truth['SMILES'].values][o_X_cols]
z = zpriors.loc[truth['Zeolite'].values][z_X_cols]

o = o.reset_index().rename(columns={'index': 'SMILES'})
z = z.reset_index().rename(columns={'index': 'Zeolite'})

X = pd.concat([o, z], axis=1)
X = X.set_index(['SMILES', 'Zeolite'])

# scale X 
scaler = StandardScaler()
with open(f"{model_folder}/input_scaling.json", "r") as f:
    op_scaling_dict = json.load(f) 
scaler.mean_ = op_scaling_dict['mean']
scaler.var_ = op_scaling_dict['var']
scaler.scale_ = np.sqrt(scaler.var_)

X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# split into train and test 
smiles_train = np.load(f"{split_dir}/smiles_train.npy")
smiles_test = np.load(f"{split_dir}/smiles_test.npy")

X_train_scaled = X_scaled[X_scaled.index.get_level_values('SMILES').isin(smiles_train)]
X_test_scaled = X_scaled[X_scaled.index.get_level_values('SMILES').isin(smiles_test)]

# Take 100% of the molecules for each framework by sorting by mol_volume then free_sasa and taking equally spaced
fraction = 1.0

fws = sorted(set(X_train_scaled.index.get_level_values('Zeolite')))
log_msg("number of frameworks:", len(fws))

for fw in fws:
    log_msg(f'\n{fw}====================================')
    df = X_train_scaled[X_train_scaled.index.get_level_values('Zeolite') == fw]
    
    if df.shape[0] == 0:
        log_msg("No data found for:", fw)
        continue
    
    idx = np.linspace(0, df.shape[0]-1, floor(df.shape[0] * fraction), dtype=int)

    sort_by = opriors.loc[df.index.get_level_values('SMILES')]
    sort_by = sort_by.sort_values(['mol_volume', 'free_sasa'])
    df = df.reindex(sort_by.index.tolist(), axis=0, level='SMILES')
    background_data = df.iloc[idx]

    log_msg("background data:", background_data.shape[0])
    log_msg("bg data fraction of train data:", background_data.shape[0] / X_train_scaled.shape[0])

    # analysis set
    analysis_data = X_test_scaled[X_test_scaled.index.get_level_values('Zeolite') == fw]
    log_msg("analysis data:", analysis_data.shape[0])

    # make torch friendly
    background_datat = torch.tensor(background_data.values, device=device).float()
    analysis_datat = torch.tensor(analysis_data.values, device=device).float()

    start = time.time()

    # run shap
    model.to(device)
    background_datat = background_datat.to(device)
    analysis_datat = analysis_datat.to(device)

    deep_explainer = shap.DeepExplainer(model, background_datat) # model, background dataset
    deep_shap_values = deep_explainer.shap_values(analysis_datat)
    log_msg("deep explainer expected value:", deep_explainer.expected_value)

    compute_time = time.time() - start
    log_msg("compute time:", compute_time / 60, "minutes")

    # format deep_shap_values
    deep_shap_values = pd.DataFrame(deep_shap_values, index=analysis_data.index, columns=analysis_data.columns)

    # save
    deep_shap_values.to_csv(os.path.join(op_dir, f'deep_shap_values_{fw}.csv'))
    
    save_time = time.time() - start - compute_time
    log_msg("save time:", save_time / 60, "minutes")
