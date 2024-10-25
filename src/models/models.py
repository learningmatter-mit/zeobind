from copy import deepcopy
import math
import time
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sigopt
from sigopt import Connection
import socket
import torch
import yaml
from zeobind.src.utils.loss import get_loss_fn
from zeobind.src.utils.default_utils import RESULTS_FILE
from zeobind.src.utils.logger import log_msg
from zeobind.src.utils.utils import (
    PAIR_COLS,
    MultiTaskTensorDataset,
    INPUT_SCALER_FILE,
    OUTPUT_SCALER_FILE,
    SCALERS,
    cluster_isomers,
    create_inputs,
    save_scaler,
)
from xgboost import XGBClassifier, XGBRegressor
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

OPTIMIZERS = dict(
    adam=torch.optim.Adam,
    sgd=torch.optim.SGD,
)

MLP_MODEL_FILE = "model.pt"
XGB_MODEL_FILE = "model.json"

def get_model(model_type, kwargs, device="cpu"):
    """
    Get a model. If a saved model is provided, load it assuming the model parameters are saved.

    Inputs:
    - model_type: str, the type of model to get
    - kwargs: dict containing parameters
    - device: str, the device to use

    Returns:
    - model: the model
    """
    # saved model
    if kwargs.get("saved_model"):
        if "xgb" in model_type:
            model = MODELS[model_type]
            model.load_model(f"{kwargs['saved_model']}/{XGB_MODEL_FILE}")
        elif "mlp" in model_type:
            saved_dict = torch.load(f"{kwargs['saved_model']}/{MLP_MODEL_FILE}", map_location=device)
            model = MODELS[model_type](**saved_dict["model_args"])
            model.load_state_dict(saved_dict["model_state_dict"])
        else:
            raise ValueError(f"[get_model] Invalid model type: {model_type}")
        log_msg("get_model", f"Loaded model\n{model}")
        return model, saved_dict

    # new model
    if "mlp" in model_type:
        kwargs["l_sizes"] = get_l_sizes(
            kwargs["input_length"],
            kwargs["layers"],
            kwargs["neurons"]
        )

    model_argnames = MODELS[model_type].__init__.__code__.co_varnames
    model_kwargs = {k: v for k, v in kwargs.items() if k in model_argnames}
    model = MODELS[model_type](**model_kwargs)

    if "mlp" in model_type:
        model.to(device)

    log_msg("get_model", f"Loaded model\n{model}")
    return model, model_kwargs


def get_l_sizes(input_length, layers, neurons):
    l_sizes = [input_length]
    for _ in range(layers):
        l_sizes.append(neurons)
    return l_sizes
        

class Trainer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.loss_fn = get_loss_fn(
            kwargs["loss_1"],
            kwargs.get("loss_2", None),
            kwargs.get("weight_1", 1.0),
            kwargs.get("weight_2", 0.0),
        )        

    def get_model(self, model_kwargs=None):
        """If model_kwargs are provided, they override the model parameters specified in self.kwargs."""
        if not model_kwargs:
            model_kwargs = deepcopy(self.kwargs)
        model, model_kwargs = get_model(
            self.kwargs["model_type"], model_kwargs, self.kwargs["device"]
        )
        self.model_args = model_kwargs
        return model

    def _train(self):
        """If needed, use self.train_kwargs, not self.kwargs."""
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def train(self):
        """Main function for training the model. Uses self.kwargs."""
        start = time.time()

        # Load data
        self.truth = pd.read_csv(self.kwargs["truth"])
        all_osdas = sorted(set(self.truth[PAIR_COLS[0]]))
        all_zeolites = sorted(set(self.truth[PAIR_COLS[1]]))
        log_msg("train", "Number of molecules:", len(all_osdas))
        log_msg("train", "Number of zeolites:", len(all_zeolites))
        self.truth = self.truth.set_index(PAIR_COLS)
        self.label = self.kwargs["label"]
        self.truth = self.truth[self.label]

        # Load features
        X = create_inputs(self.kwargs, self.truth.index)
        self.kwargs["input_length"] = X.shape[1]

        # Split data
        split_by = self.kwargs.get("split_by", "osda")
        split_folder = self.kwargs.get("split_folder", None)
        splits = dict()
        if split_folder:
            for split in ["train", "val", "test"]:
                if split_by == "osda":
                    file_root = "smiles"
                elif split_by == "zeolite":
                    file_root = "fws"
                splits[split] = np.load(
                    os.path.join(self.kwargs["split_folder"], f"{file_root}_{split}.npy")
                ).tolist()
        else:
            rng = np.random.default_rng(self.kwargs["seed"])
            if split_by == "osda":
                log_msg(
                    "train",
                    "WARNING Splitting by isomer groups. Note that the train/val/test split fractions specify the fraction of isomer groups, so the actual number of molecules in each split may not follow the same ratios. If this is a concern, manually specify the splits and use `split_folder`.",
                )
                nonisomeric_smiles_lookup = cluster_isomers(all_osdas)
                isomer_groups = list(nonisomeric_smiles_lookup.keys())
                rng.shuffle(isomer_groups)
            elif split_by == "zeolite":
                rng.shuffle(all_zeolites)
            else:
                raise ValueError(f"[train] Invalid split_by value: {split_by}")

            train_idx, test_idx = train_test_split(
                all_zeolites,
                test_size=self.kwargs["val_frac"] + self.kwargs["test_frac"],
                random_state=self.kwargs["seed"],
            )[0]
            val_idx, test_idx = train_test_split(
                test_idx,
                test_size=self.kwargs["test_frac"]
                / (self.kwargs["val_frac"] + self.kwargs["test_frac"]),
                random_state=self.kwargs["seed"],
            )

            if split_by == "osda":
                for idx, split in zip(
                    [train_idx, val_idx, test_idx], ["train", "val", "test"]
                ):
                    splits[split] = []
                    for group in idx:
                        splits[split] += list(nonisomeric_smiles_lookup[group])
            elif split_by == "zeolite":
                for idx, split in zip(
                    [train_idx, val_idx, test_idx], ["train", "val", "test"]
                ):
                    splits[split] = idx

        splits[split] = sorted(splits[split])

        if split_by == "osda":
            self.truth = self.truth.reset_index(PAIR_COLS[1])
        else:
            self.truth = self.truth.reset_index(PAIR_COLS[0])
        truths = dict()
        for split in ["train", "val", "test"]:
            truths[split] = (
                self.truth[self.truth.index.isin(splits[split])]
                .reset_index()
                .set_index(PAIR_COLS)
            )

        log_msg(
            "train",
            "Split sizes:",
            truths["train"].size,
            truths["val"].size,
            truths["test"].size,
        )
        if self.kwargs.get("shuffle_data", False):
            log_msg("train", "Data is shuffled.")
            for split in ["train", "val", "test"]:
                truths[split] = truths[split].sample(
                    frac=1, random_state=self.kwargs["seed"]
                )

        # Scale truth
        if self.kwargs.get("op_scaler", None):
            self.op_scaler = SCALERS[self.kwargs["op_scaler"]]()
            self.op_scaler.fit(truths["train"][self.label])

            for split in ["train", "val", "test"]:
                scaled = self.op_scaler.transform(truths[split][self.label])
                truths[split][self.label] = pd.DataFrame(
                    scaled, index=truths[split].index, columns=[self.label]
                )

            save_scaler(self.op_scaler, self.kwargs["op_scaler"], self.kwargs["output"], OUTPUT_SCALER_FILE)

        # Scale inputs
        X = dict(
            train=X.loc[truths["train"].index],
            val=X.loc[truths["val"].index],
            test=X.loc[truths["test"].index],
        )
        if self.kwargs.get("ip_scaler", None):
            self.ip_scaler = SCALERS[self.kwargs["ip_scaler"]]()
            self.ip_scaler.fit(X["train"])
            for split in ["train", "val", "test"]:
                X[f"X_{split}"] = pd.DataFrame(
                    self.ip_scaler.transform(X[f"X_{split}"]),
                    index=X[f"X_{split}"].index,
                    columns=X[f"X_{split}"].columns,
                )
            save_scaler(self.ip_scaler, self.kwargs["ip_scaler"], self.kwargs["output"], INPUT_SCALER_FILE)

        # Further processing
        self.X, self.truths = self.process(X, truths)
        prep_time = time.time() - start
        log_msg("train", f"Preprocessing time: {prep_time} seconds")

        # Hyperparameter tuning
        if self.kwargs["tune"]:
            # Set up
            conn = Connection(client_token=self.kwargs["sigopt_api_token"])
            sigopt.set_project(self.kwargs["sigopt_project_id"])
            os.environ["SIGOPT_API_TOKEN"] = self.kwargs["sigopt_api_token"]
            os.environ["SIGOPT_PROJECT"] = self.kwargs["sigopt_project_id"]
            hostname = socket.gethostname()

            # Load hyperparameter file
            with open(self.kwargs["hyp_file"], "r") as f:
                hyp_params = yaml.load(f, Loader=yaml.Loader)

            # Create experiment
            experiment = sigopt.create_experiment(
                name=hyp_params["name"],
                type=hyp_params["type"],
                parameters=hyp_params["parameters"],
                metrics=hyp_params["metrics"],
                parallel_bandwidth=self.kwargs["sigopt_parallel_bandwidth"],
                budget=self.kwargs["sigopt_budget"],
            )
            log_msg("train", "Sigopt experiment id", experiment.id)

            # Loop over budget
            run_i = 0
            for run in experiment.loop():
                with run:
                    self.train_kwargs = deepcopy(self.kwargs)
                    self.train_kwargs.update(run.params)
                    self.model = self.get_model(self.train_kwargs)
                    self._train()

                    if math.isinf(self.val_losses[-1]):
                        sigopt.log_failture()
                        log_msg("train", "Inf loss, skipping")
                        continue

                    # Log metrics
                    run.log_metadata("hyperparams", run.params)
                    run.log_metric("overall_loss", self.val_losses[-1])
                    log_msg(
                        "train",
                        f"Run {run_i}",
                        "train",
                        "{:.3f}".format(self.train_losses[-1]),
                        "val",
                        "{:.3f}".format(self.val_losses[-1]),
                    )

                    # Save model
                    model_dir = os.path.join(
                        self.kwargs["output"], f"hp_tune_run_{run_i}"
                    )
                    self.save_model(model_dir)

                run_i += 1

            # Extract best run
            best_runs = experiment.get_best_runs()
            best_runs_ls = []
            for r in best_runs:
                best_runs_ls.append(
                    [r.model, r.id, r.values["overall_loss"].value, r.assignments]
                )
            log_msg("train", "best_runs_ls:\n", best_runs_ls)
            best_run_params = {}
            for key, val in best_runs_ls[0][-1].items():
                best_run_params[key] = val
            self.kwargs.update(best_run_params)

        # Train model with best or specified hyperparameters
        self.train_kwargs = deepcopy(self.kwargs)
        self.model = self.get_model(self.train_kwargs)
        self._train()
        train_time = time.time() - prep_time
        log_msg("train", f"Time to train: {train_time} seconds")

        # Evaluate on all splits
        self.preds = dict()
        for split in ["train", "val", "test"]:
            loss, preds = self.evaluate(split, True)
            self.__setattr__(f"{split}_loss", loss)
            self.preds[split] = preds

        # Evaluate on unscaled data
        if self.kwargs.get("op_scaler", None):
            for split in ["train", "val", "test"]:
                unscaled = self.op_scaler.inverse_transform(
                    self.truths[split][self.label]
                )
                self.truths[split] = pd.DataFrame(
                    unscaled, index=self.truths[split].index, columns=[self.label]
                )
                self.preds[split] = self.op_scaler.inverse_transform(self.preds[split])

                loss = self.evaluate(split, False)
                self.__setattr__(f"{split}_loss_unscaled", loss)
        else:
            for split in ["train", "val", "test"]:
                self.__setattr__(
                    f"{split}_loss_unscaled", self.__getattribute__(f"{split}_loss")
                )

        # For easy reading, also write to results file
        self.write_results()

        # Dump kwargs
        with open(os.path.join(self.kwargs["output"], "final_kwargs.yml"), "w") as f:
            yaml.dump(self.kwargs, f)

        # Save model
        if self.kwargs.get("save_model"):
            self.save_model(self.kwargs["output"])
        log_msg("train", "Output folder:", self.kwargs["output"])
        log_msg("train", "Done")

    def process(self, X, truths):
        return X, truths

    def save_model(self, model_dir):
        raise NotImplementedError

    def write_results(self):
        dp = 8
        newline = ""
        newline += self.kwargs["task"]
        newline += ","
        newline += self.kwargs["run"]
        newline += ","
        newline += str(round(self.train_loss, dp))
        newline += ","
        newline += str(round(self.val_loss, dp))
        newline += ","
        newline += str(round(self.test_loss, dp))
        newline += ","
        newline += str(round(self.train_loss_unscaled, dp))
        newline += ","
        newline += str(round(self.val_loss_unscaled, dp))
        newline += ","
        newline += str(round(self.test_loss_unscaled, dp))
        newline += ","
        newline += "\n"

        results_file = self.kwargs.get("results_file", RESULTS_FILE)
        with open(results_file, "a") as f:
            f.write(newline)
        log_msg("write_results", f"Written to {results_file}:\n{newline}")



class XGBTrainer(Trainer):
    def __init__(self, kwargs):
        super().__init__(kwargs)

    def _train(self, verbose=True):
        """If needed, use self.train_kwargs, not self.kwargs."""
        self.model.fit(
            self.X["train"].values,
            self.truths["train"].values,
            eval_set=[(self.X["val"].values, self.truths["val"].values)],
            verbose=True,
        )

    def evaluate(self, split="val", return_preds=False):
        if self.train_kwargs["task"].contains("regression"):
            preds = self.model.predict(self.X[split].values)
        elif self.train_kwargs["task"].contains("classification"):
            preds = self.model.predict_proba(self.X[split].values)
        else:
            raise ValueError(f"[evaluate] Invalid task: {self.train_kwargs['task']}")
        loss = self.loss_fn(preds, self.truths[split].values)
        if not return_preds:
            preds = None
        return loss, preds

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_model(os.path.join(model_dir, "model.json"))


class NClass(nn.Module):
    def __init__(
        self,
        l_sizes=(16, 8, 4),
        class_op_size=2,
        batch_norm=False,
        softmax=True,
        dropout=0.5,
    ):
        super(NClass, self).__init__()
        self.l_sizes = l_sizes
        self.batch_norm = batch_norm
        self.softmax = softmax
        self.dropout = dropout
        self.class_op_size = class_op_size

        num_layers = len(self.l_sizes)
        layers_list = []
        for i in range(0, num_layers - 1):
            layers_list.append(nn.Linear(self.l_sizes[i], self.l_sizes[i + 1]))
            layers_list.append(nn.ReLU())
            if self.batch_norm:
                layers_list.append(nn.BatchNorm1d(self.l_sizes[i + 1]))
            if self.dropout:
                if i != num_layers - 2:
                    # https://stats.stackexchange.com/questions/425610/why-massive-random-spikes-of-validation-loss recommends no dropout before last layer
                    layers_list.append(nn.Dropout(p=self.dropout))

        class_modules = [
            *layers_list,
            # nn.Linear(c_sizes[-1], 1)
            nn.Linear(self.l_sizes[-1], self.class_op_size),
        ]
        if self.softmax:
            class_modules.append(nn.Softmax(dim=1))
        self.classifier = nn.Sequential(*class_modules)

    def forward(self, x):
        return self.classifier(x)


class Regressor(nn.Module):
    def __init__(
        self, l_sizes=(16, 8, 4), class_op_size=1, batch_norm=False, dropout=0.5
    ):
        super(Regressor, self).__init__()
        self.l_sizes = l_sizes
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.class_op_size = class_op_size

        num_layers = len(self.l_sizes)
        layers_list = []
        for i in range(0, num_layers - 1):
            layers_list.append(nn.Linear(self.l_sizes[i], self.l_sizes[i + 1]))
            layers_list.append(nn.ReLU())  
            if self.batch_norm:
                layers_list.append(nn.BatchNorm1d(self.l_sizes[i + 1]))
            if self.dropout:
                if i != num_layers - 2:
                    layers_list.append(nn.Dropout(p=self.dropout))
        self.regressor = nn.Sequential(
            *layers_list, nn.Linear(self.l_sizes[-1], self.class_op_size)
        )

    def forward(self, x):
        return self.regressor(x)


class MLPTrainer(Trainer):  
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.scheduler = None 

    def _train(self):
        """
        Initialise the model, and run the training loop.
        Uses self.train_kwargs, not self.kwargs.
        """

        params = self.model.parameters()
        self.optimizer = OPTIMIZERS[self.train_kwargs["optimizer"]](
            params=params, lr=self.train_kwargs["lr"]
        )

        if self.train_kwargs.get("scheduler", None):
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, "min", patience=self.train_kwargs.get("lr_patience", 10)
            )

        while self.train_kwargs["batch_size"] > 0:
            try:
                self.train_losses = []
                self.val_losses = []
                self.epochs = []
                self.epoch = 0
                self.patience = deepcopy(self.train_kwargs["patience"])
                self.best_val_loss = float("inf")
                self.best_model_state = None
                self.best_epoch = 0

                for epoch in range(self.train_kwargs["epochs"]):
                    self.epoch = epoch
                    self.train_losses.append(self.train_single_epoch())
                    self.val_losses.append(self.evaluate("val", False))

                    if self.train_kwargs.get("early_stopping", False):
                        if self.val_losses[-1] < self.best_val_loss:
                            self.best_val_loss = self.val_losses[-1]
                            self.best_model_state = self.model.state_dict()
                            self.best_epoch = epoch
                            self.patience = deepcopy(self.train_kwargs["patience"])
                        else:
                            self.patience -= 1
                            if self.patience == 0:
                                break
                    else:
                        self.best_model_state = self.model.state_dict()
                        self.best_epoch = epoch

            except RuntimeError as e:
                if "out of memory" in str(e):
                    log_msg(
                        "_train",
                        "WARNING: Out of memory with batch_size=",
                        self.train_kwargs["batch_size"],
                    )
                    log_msg("_train", "Reducing batch_size by half")
                    self.train_kwargs["batch_size"] //= 2
                    if self.train_kwargs["model"] == "mlp":
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                else:
                    raise e
            else:
                break

    def train_single_epoch(self):
        losses = []
        self.model.train()
        for i, (X, y, _) in enumerate(self.dataloaders["train"]):
            # log_msg("train_single_epoch", f"Batch {i}")
            X = X.to(self.train_kwargs["device"])
            y = y.to(self.train_kwargs["device"])
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def evaluate(self, split="val", return_preds=False):
        losses = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (X, y, _) in enumerate(self.dataloaders[split]):
                # log_msg("evaluate", f"Batch {i}")
                X = X.to(self.train_kwargs["device"])
                y = y.to(self.train_kwargs["device"])
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                losses.append(loss.item())
                if return_preds:
                    preds.append(pred)

        if self.scheduler:
            self.scheduler.step(np.mean(losses))
        if return_preds:
            preds = torch.cat(preds, dim=0)
        else:
            preds = None

        return np.mean(losses), preds

    def process(self, X, truths):
        self.dataloaders = dict()
        for split in ["train", "val", "test"]:
            indices = X[split].index.to_list()
            X[split] = torch.tensor(X[split].values.astype('float64')).float()
            truths[split] = torch.tensor(truths[split].values.astype('float64')).float()

            dataset = MultiTaskTensorDataset(X[split], truths[split], indices)
            self.dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.kwargs["batch_size"],
                shuffle=self.kwargs.get("shuffle_batch", False),
                drop_last=self.kwargs["drop_last"],
            )

        return X, truths

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "opt_state_dict": self.optimizer.state_dict(),
                "epoch_losses": self.train_losses,
                "val_losses": self.val_losses,
                "model_args": self.model_args,
            },
            os.path.join(model_dir, MLP_MODEL_FILE),
        )

MODELS = dict(
    xgb_regressor=XGBRegressor,
    xgb_classifier=XGBClassifier,
    mlp_regressor=Regressor,
    mlp_classifier=NClass,
)