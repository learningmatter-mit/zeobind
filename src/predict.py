from ast import literal_eval
from copy import deepcopy
from itertools import product
import numpy as np 
import json
import multiprocessing
import argparse
import os
import time
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from zeobind.src.utils.utils import get_load_norm_bins

from zeobind.src.utils.pred_tasks import (
    PREDICTION_TASK_DICT,
    MCLASS_TASK,
    BINARY_TASK,
    ENERGY_TASK,
    COL_DICT,
)
from zeobind.src.utils.utils import (
    PAIR_COLS,
    INPUT_SCALER_FILE,
    OUTPUT_SCALER_FILE,
    create_inputs,
    get_prior_files,
    rescale_data,
    scale_data,
)
from zeobind.src.utils.logger import log_msg
from zeobind.src.models.models import get_model


def set_model_path_and_type(kwargs, task):
    if task == ENERGY_TASK:
        model_paths = kwargs["energy_models"]
        model_type = kwargs["energy_model_type"]
    elif task == BINARY_TASK:
        model_paths = kwargs["binary_models"]
        model_type = kwargs["binary_model_type"]
    elif task == MCLASS_TASK:
        model_paths = kwargs["loading_models"]
        model_type = kwargs["loading_model_type"]
    return model_paths, model_type


def single_model_preds(task, input_kwargs, condense=False):
    """
    Predicts on a single model.

    Inputs:
        task: str, task to predict on
        input_kwargs: dict of input arguments
        condense: If True, saves the idxmax value instead of the probabilities.
    
    Returns:
        y_preds: pd.DataFrame, predictions
    """
    single_start_time = time.time()
    kwargs = deepcopy(input_kwargs)
    kwargs["task"] = task
    kwargs["model_paths"], kwargs["model_type"] = set_model_path_and_type(kwargs, task)

    log_msg("debug", kwargs["osda_prior_file"], kwargs["zeolite_prior_file"])

    # Get list of framework-molecule pair identities
    if kwargs.get("full_matrix", True):
        if ".pkl" in kwargs["osda_prior_file"]:
            osda_priors = pd.read_pickle(kwargs["osda_prior_file"])
            osdas = osda_priors.index
        else:
            with open(kwargs["osda_prior_file"], "r") as f:
                osdas = f.readlines() 
            osdas = [o.strip() for o in osdas]
        
        if ".pkl" in kwargs["zeolite_prior_file"]:
            zeos = pd.read_pickle(kwargs["zeolite_prior_file"]).index
        else:
            with open(kwargs["zeolite_prior_file"], "r") as f:
                zeos = f.readlines() 
            zeos = [z.strip() for z in zeos]

        pairs = list(product(osdas, zeos))

    else:
        pairs = pd.read_csv(kwargs["pairs_file"], index_col=0)[PAIR_COLS].values.tolist()

    pairs = pd.DataFrame(pairs, columns=PAIR_COLS).set_index(PAIR_COLS)

    # create inputs 
    X = create_inputs(kwargs, pairs.index, save=True)
    kwargs["input_length"] = len(X.columns)
    log_msg("single_model_preds", "Number of pairs to predict on:", len(pairs))

    # apply scaling to prior
    with open(os.path.join(kwargs["saved_model"], INPUT_SCALER_FILE), "r") as f:
        input_scaler = json.load(f)
    X_scaled = scale_data(X, input_scaler)

    # make it torch friendly
    X_scaled = torch.tensor(X_scaled.astype("float32").values)

    # make it into a dataloader
    X_scaled_dataloader = DataLoader(
        X_scaled, batch_size=kwargs["batch_size"], shuffle=False
    )

    # get model
    model = get_model(kwargs["model_type"], kwargs, kwargs["device"])
    model.to(kwargs["device"])

    # predict
    y_preds = []
    with torch.no_grad():
        model.eval()
        for X_batch in X_scaled_dataloader:
            X_batch = X_batch.to(kwargs["device"])
            y_preds.append(model(X_batch))

    # format outputs
    y_preds = torch.vstack(y_preds)
    y_preds = y_preds.cpu().numpy()

    # unscale predictions
    truth_scaler_file = os.path.join(kwargs["saved_model"], OUTPUT_SCALER_FILE)
    if os.path.exists(truth_scaler_file):
        with open(truth_scaler_file, "r") as f:
            truth_scaler = json.load(f)
        y_preds = rescale_data(y_preds, truth_scaler)

    if task == BINARY_TASK or task == MCLASS_TASK:
        if condense: # save class instead of proba
            log_msg("single_model_preds", f"Condensing probabilities to class for {task}")
            y_preds = np.argmax(y_preds, axis=1)
            if task == BINARY_TASK:
                kwargs["label"] = "binding"
            elif task == MCLASS_TASK:
                kwargs["label"] = "load_norm_class"
        else:
            if task == BINARY_TASK:
                kwargs["label"] = ["nb", "b"]
            elif task == MCLASS_TASK:
                _, bins_dict, _ = get_load_norm_bins()
                kwargs["label"] = [f"{COL_DICT[MCLASS_TASK]}_{i}" for i in bins_dict.keys()]

    y_preds = pd.DataFrame(y_preds, columns=[kwargs["label"]], index=pairs.index)
    log_msg("single_model_preds", "Total time:", "{:.2f}".format((time.time() - single_start_time) / 60), "mins")
    return y_preds

def ensemble_preds(task, kwargs, condense=False):
    '''
    Saves a single CSV file where columns are labelled by model number.
    condense: If True, saves the idxmax value instead of the probabilities.
    '''
    y_preds = [] 
    cols = [] 
    for idx, model_path in enumerate(kwargs["model_paths"]): 
        single_kwargs = deepcopy(kwargs)
        single_kwargs["saved_model"] = model_path
        single_y_preds = single_model_preds(task, single_kwargs, condense)

        if single_kwargs["task"] == ENERGY_TASK:
            lab = COL_DICT[single_kwargs["task"]]
            cols.append(f"{lab} ({idx})")
        else:
            if condense: # class output
                lab = COL_DICT[single_kwargs["task"]]
                cols.append(f"{lab} ({idx})")
            else: # proba output of all classes
                single_cols = [c[0] + f" ({idx})" for c in list(single_y_preds.columns)]
                cols.extend(single_cols)

        y_preds.append(single_y_preds)

    y_preds = pd.concat(y_preds, axis=1)
    y_preds.columns = cols
    return y_preds

def predict(files, kwargs, condense=False):
    """
    Make and save predictions. This loops over prediction tasks. 

    Inputs:
        files: 
            - tuple of (osda_file, zeolite_file) if kwargs["full_matrix"] is True, or a single file containing osda-zeolite pairs if kwargs["full_matrix"] is False
    kwargs: dictionary of arguments

    Returns:
        None
    """
    # make prediction filenames
    if kwargs.get("full_matrix", True):
        kwargs["osda_prior_file"] = files[0]
        kwargs["zeolite_prior_file"] = files[1]
        osda_file_root = os.path.splitext(os.path.basename(files[0]))[0]
        zeolite_file_root = os.path.splitext(os.path.basename(files[1]))[0]
        kwargs["filename"] = f"{osda_file_root}_{zeolite_file_root}_preds.csv"
    else:
        kwargs["pairs_file"] = files
        kwargs["filename"] = f"{os.path.splitext(os.path.basename(files))[0]}_preds.csv"

    # predict
    for task in kwargs["tasks"]:
        log_msg("predict", "Predicting for", task)
        os.makedirs(f"{kwargs['output']}/{task}", exist_ok=True)
        task_kwargs = deepcopy(kwargs)
        task_obj = PREDICTION_TASK_DICT[task]()
        task_kwargs.update(task_obj.__dict__)
        task_kwargs["model_paths"] = kwargs[f"{task.split('_')[0]}_models"]


        if not kwargs["ensemble"][task]:
            y_preds = single_model_preds(task, task_kwargs, task_kwargs.get('condense', condense))
        else:
            y_preds = ensemble_preds(task, task_kwargs, task_kwargs.get('condense', condense))

        y_preds.reset_index(inplace=True)
        filename = f"{kwargs['output']}/{task}/{kwargs['filename']}"
        y_preds.to_csv(filename)
        log_msg("predict", "Predicted for", filename)

def main(kwargs):
    """Preprocess input arguments and handle parallelisation of predictions. Parallelisation is carried out over files."""
    start_time = time.time()  # seconds

    # get list of osda and zeolite prior files for prediction
    kwargs["osda_prior_files"] = get_prior_files(
        kwargs["osda_prior_files"], kwargs["osda_prior_folders"]
    )
    kwargs["zeolite_prior_files"] = get_prior_files(
        kwargs["zeolite_prior_files"], kwargs["zeolite_prior_folders"]
    )

    # get zeolite-osda pairs if not full matrix
    if not kwargs.get("full_matrix", True):
        kwargs["pairs_files"] = get_prior_files(
            kwargs.get("pairs_files", None),
            kwargs.get("pairs_folders", None),
        )

    # get zeolite-osda file pairs
    if kwargs.get("full_matrix", True):
        files = sorted(
            product(kwargs["osda_prior_files"], kwargs["zeolite_prior_files"])
        )
    else:
        files = kwargs["pairs_files"]
    
    log_msg("main", "Number of file (pairs) to predict on:", len(files))

    # format output folder name
    log_msg("main", "Output folder is", kwargs["output"])
    os.makedirs(kwargs["output"], exist_ok=True)

    # ensemble settings
    kwargs["ensemble"] = {ENERGY_TASK: False, BINARY_TASK: False, MCLASS_TASK: False}
    if len(kwargs['binary_models']) > 1:
        kwargs['ensemble'][BINARY_TASK] = True
    if len(kwargs['energy_models']) > 1:
        kwargs['ensemble'][ENERGY_TASK] = True
    if len(kwargs['loading_models']) > 1:
        kwargs['ensemble'][MCLASS_TASK] = True

    # task settings 
    if kwargs["task"] == "all":
        kwargs["tasks"] = [ENERGY_TASK, BINARY_TASK, MCLASS_TASK]
    else:
        kwargs["task"] = [kwargs["task"]]

    # start predictions
    if kwargs["num_processes"] == 1:
        for file in files:
            predict(file, kwargs)
    else:
        with multiprocessing.Pool(processes=kwargs["num_processes"]) as pool:
            pool.starmap(predict, product(files, [kwargs]))

    log_msg(
        "main", "Total time:", "{:.2f}".format((time.time() - start_time) / 60), "mins"
    )

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="zeobind prediction")

    # models 
    parser.add_argument("--binary_models", nargs='+', help="Binary classification model filepaths. If more than one is provided, it is assumed that ensemble predictions will be made")
    parser.add_argument("--binary_model_type", help="Binary model type", choices=["xgb_classifier", "mlp_classifier"])
    parser.add_argument("--energy_models", nargs='+', help="Energy regression model filepaths. If more than one is provided, it is assumed that ensemble predictions will be made")
    parser.add_argument("--energy_model_type", help="Energy model type", choices=["xgb_regressor", "mlp_regressor"])
    parser.add_argument("--loading_models", nargs='+', help="Loading multiclassification model filepaths. If more than one is provided, it is assumed that ensemble predictions will be made")
    parser.add_argument("--loading_model_type", help="Loading model type", choices=["xgb_classifier", "mlp_classifier"])

    # set up 
    parser.add_argument("--device", help="Device", default="cpu")
    parser.add_argument("--num_processes", help="Number of processes. If more than 1, prediction is parallelised.", type=int, default=1)

    # input
    parser.add_argument("--osda_prior_folders", nargs="+", help="OSDA input folders", default=[])
    parser.add_argument("--osda_prior_files", nargs="+", help="OSDA input files", default=[])
    parser.add_argument("--osda_prior_map", help="OSDA features to use", default=None)
    parser.add_argument("--zeolite_prior_folders", nargs="+", help="Zeolite input folders", default=[])
    parser.add_argument("--zeolite_prior_files", nargs="+", help="Zeolite input files", default=[])
    parser.add_argument("--zeolite_prior_map", help="Zeolite features to use", default=None)
    parser.add_argument("--full_matrix", help="Predict on full matrix. If this is True, pairs_files is ignored", action="store_true")
    parser.add_argument("--pairs_files", nargs="+", help="File containing framework-molecule pairs. This is used if full_matrix is False", default=[])
    parser.add_argument("--pairs_folders", nargs="+", help="Folders containing framework-molecule pairs. This is used if full_matrix is False", default=[])

    # predictions 
    parser.add_argument("--task", help="Prediction task", choices=["all", BINARY_TASK, ENERGY_TASK, MCLASS_TASK], required=True)
    parser.add_argument("--condense", help="Condense classification outputs from probabilities", action="store_true")
    parser.add_argument("--output", help="Output folder", required=True)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=256)

    args = parser.parse_args()
    kwargs = args.__dict__
    main(kwargs)
