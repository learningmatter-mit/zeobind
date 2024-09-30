import argparse
from datetime import datetime
import numpy as np
import random
import torch
import os
import yaml
from zeobind.src.models.xgb import XGBTrainer
from zeobind.src.models.mlp import MLPTrainer
from utils.pred_tasks import PREDICTION_TASK_DICT
from utils.logger import log_msg


def preprocess(args: argparse.Namespace) -> dict:
    # Default parameters TODO: remove because these were for compatibility with the old code
    kwargs = {
        "batch_norm": False,
        "softmax": True,
        "lr": 1e-2,
        "load_models": None,
        "first_class": 0,
    }

    # Load config file
    config_file = args.__dict__["config"]
    with open(config_file, "rb") as file:
        kwargs_config = yaml.load(file, Loader=yaml.Loader)
        kwargs.update(kwargs_config)
    kwargs["config"] = config_file

    # Format output folder name 
    if os.path.isdir(kwargs["output"]):
        now = "_%d%d%d_%d%d%d" % (
            datetime.now().year,
            datetime.now().month,
            datetime.now().day,
            datetime.now().hour,
            datetime.now().minute,
            datetime.now().second,
        )
        kwargs["output"] = kwargs["output"] + now
    os.makedirs(kwargs["output"], exist_ok=True)

    # Check device and threads
    if "device" not in kwargs.keys():
        try:
            kwargs["device"] = torch.cuda.current_device()
        except RuntimeError:
            kwargs["device"] = "cpu"
    log_msg(
        "preprocess", 
        "kwargs device:",
        kwargs["device"],
        "of type",
        type(kwargs["device"]),
    )
    torch_threads = torch.get_num_threads()
    os_threads = os.cpu_count()
    log_msg("preprocess", "Number of threads:", torch_threads, os_threads)

    # Set seed
    if not kwargs.get("seed"):
        kwargs["seed"] = 422
        log_msg("preprocess", "Seed not set, using default seed 422")
    set_seeds(kwargs["seed"])

    # Set prediction task parameters 
    pred_task = PREDICTION_TASK_DICT[kwargs["task"]]
    kwargs.update(pred_task().__dict__)

    # Dump arguments TODO remove this because it was useful for debugging but messy 

    log_msg("preprocess", "Output folder:", kwargs["output"])
    return kwargs


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


TRAINER_DICT = dict(
    xgb=XGBTrainer,
    nn=MLPTrainer,
)

def get_trainer(kwargs):
    return TRAINER_DICT[kwargs["model_type"]](kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("zeobind")
    # parser.add_argument("--config", help="path to config file", required=True)

    # set up 
    parser.add_argument("--output", type=str, help="output folder", required=True)
    parser.add_argument("--seed", type=int, help="seed", default=12934)
    parser.add_argument("--device", help="device", choices=["cpu", "cuda"], default="cuda")

    # data
    parser.add_argument("--osda_prior_file", type=str, help="osda prior file", default=None)
    parser.add_argument("--zeolite_prior_file", type=str, help="zeolite prior file", default=None)
    parser.add_argument("--osda_prior_map", type=str, help="osda prior map")
    parser.add_argument("--zeolite_prior_map", type=str, help="zeolite prior map")
    parser.add_argument("--truth", type=str, help="truth file", default=None)
    parser.add_argument("--split_folder", type=str, help="split folder", default=None)
    parser.add_argument("--split_by", type=str, help="split", choices=['osda', 'zeolite'], default='osda')
    parser.add_argument("--drop_fws", nargs="+", help="frameworks to drop", default=[])

    # model and training parameters
    parser.add_argument("--model_type", type=str, help="model type", choices=["xgb", "nn"], default="nn")
    parser.add_argument("--loss_1", type=str, help="loss 1", default="celoss")
    parser.add_argument("--loss_2", type=str, help="loss 2", default=None)
    parser.add_argument("--weight_1", type=float, help="weight 1", default=1.0)
    parser.add_argument("--weight_2", type=float, help="weight 2", default=0.0)
    # parser.add_argument("l_sizes", type=int, nargs="+", help="layer sizes", default=[])
    parser.add_argument("--input_scaler", type=str, help="input scaler", default="standard")
    parser.add_argument("--optimizer", type=str, help="optimizer", default="adam")
    parser.add_argument("--scheduler", type=str, help="scheduler", default=False)
    parser.add_argument("--epochs", type=int, help="epochs", default=500)
    parser.add_argument("--batch_size", type=int, help="batch size", default=256)
    parser.add_argument("--patience", type=int, help="patience", default=10)
    parser.add_argument("--min_delta", type=float, help="min delta", default=0.05)
    parser.add_argument("--drop_last", action="store_true", help="drop last")
    parser.add_argument("--scaler", type=str, help="output scaler", default=None)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--lr_patience", type=int, help="lr patience", default=20)
    parser.add_argument("--layers", type=int, help="layers", default=2)
    parser.add_argument("--neurons", type=int, help="neurons", default=512)
    parser.add_argument("--batch_norm", action="store_true", help="batch norm")
    parser.add_argument("--softmax", action="store_true", help="softmax")
    parser.add_argument("--dropout", type=float, help="dropout", default=0.4)
    parser.add_argument("--num_classes", type=int, help="number of classes", default=2)
    parser.add_argument("--task", help="prediction task", choices=PREDICTION_TASK_DICT.keys(), default="binary")
    parser.add_argument("--shuffle_batch", action="store_true", help="shuffle batch")
    parser.add_argument("--save", action="store_true", help="save ground truth, predictions, and inputs at end of experiment.")

    # hyperparameter tuning 
    parser.add_argument("--tune", action="store_true", help="tune hyperparameters")
    parser.add_argument("--hp_file", type=str, help="hyperparameter file", default=None)
    parser.add_argument("--sigopt_api_token", type=str, help="sigopt api token", default=None)
    parser.add_argument("--sigopt_project_id", type=str, help="sigopt project id", default=None)
    parser.add_argument("--sigopt_budget", type=int, help="sigopt budget", default=20)

    args = parser.parse_args()
    kwargs = preprocess(args)
    trainer = get_trainer(kwargs)
    trainer.train()
