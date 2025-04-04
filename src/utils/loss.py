import torch
import numpy as np
import pandas as pd
from torch import nn

from zeobind.src.utils.logger import log_msg
from zeobind.src.utils.pred_tasks import BINARY_TASK, ENERGY_TASK, MCLASS_TASK


class BaseLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, yhat, y):
        y = self._format(y, yhat)
        yhat = self._format(yhat, is_yhat=True)
        return self.loss(yhat, y)

    def _format(self, data, yhat=None, is_yhat=False):
        data = [data] if not isinstance(data, list) else data
        formatter = self._format_single_yhat if is_yhat else self._format_single_y
        data = [formatter(d, yhat) for d in data]
        return data[0] if len(data) == 1 else data

    def _format_single_y(self, y, yhat):
        y = torch.tensor(y, device=self.device) if not isinstance(y, torch.Tensor) else y
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def _format_single_yhat(self, yhat, _):
        yhat = torch.tensor(yhat, device=self.device) if not isinstance(yhat, torch.Tensor) else yhat
        return yhat.reshape(-1, 1) if yhat.ndim == 1 else yhat

    def loss(self, yhat, y):
        try:
            return self.loss_fn(yhat, y)
        except Exception:
            return self.loss_fn(yhat, y.squeeze().long())


class MSELoss(BaseLoss):
    def __init__(self, device="cpu"):
        super().__init__()
        self.loss_fn = nn.MSELoss()


class RMSELoss(BaseLoss):
    def __init__(self, eps=1e-16, device="cpu"):
        super().__init__()
        self.eps = eps
        self.loss_fn = lambda yhat, y: torch.sqrt(nn.MSELoss()(yhat, y) + self.eps)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, device="cpu"):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def format_y(self, y, yhat):
        if type(y) != torch.Tensor:
            y = torch.tensor(y, device=self.device)
        if y.shape[1] == 1 and yhat.shape[1] == 2:
            # Otherwise, CrossEntropyLoss runs into segmentation error for binary classification tasks on Mac M1's GPU
            y = torch.concatenate([1-y, y], 1)
        return y.squeeze()


class MultiTaskLoss(BaseLoss):
    def __init__(self, 
                 loss_1, loss_2=None, loss_3=None, loss_4=None, 
                 weight_1=1.0, weight_2=0.0, weight_3=0.0, weight_4=0.0,
                 device="cpu"):
        """Multitask loss. This is hard coded for the 3 tasks, and specifically such that the multiclassification loss has 2 weighted loss terms."""
        super().__init__()
        self.binary_loss_fn = get_loss_fn(loss_1=loss_1, weight_1=weight_1, device=device, task=BINARY_TASK)
        self.energy_loss_fn = get_loss_fn(loss_1=loss_2, weight_1=weight_2, device=device, task=ENERGY_TASK)
        self.load_loss_fn = get_loss_fn(loss_1=loss_3, weight_1=weight_3, loss_2=loss_4, weight_2=weight_4, device=device, task=MCLASS_TASK)
    
    def loss(self, yhat, y):
        binary_loss = self.binary_loss_fn(yhat[0], y[:, 0])
        energy_loss = self.energy_loss_fn(yhat[1], y[:, 1])
        load_loss = self.load_loss_fn(yhat[2], y[:, 2:])
        return binary_loss + energy_loss + load_loss


LOSS_DICT = dict(
    rmse=RMSELoss, 
    mse=MSELoss, 
    celoss=CrossEntropyLoss,
    multitask=MultiTaskLoss,
    )

def get_loss_fn(
        loss_1, loss_2=None, loss_3=None, loss_4=None,
        weight_1=1.0, weight_2=0.0, weight_3=0.0, weight_4=0.0,
        task="binary",
        **kwargs):
    """
    Returns:
        A weighted loss function for single or multi-task prediction.
    """
    device = kwargs.get("device", "cpu")
    log_msg("get_loss_fn", f"Getting loss function for task: {task}")
    if task == "multitask":
        return LOSS_DICT["multitask"](
            loss_1=loss_1, loss_2=loss_2, loss_3=loss_3, loss_4=loss_4,
            weight_1=weight_1, weight_2=weight_2, weight_3=weight_3, weight_4=weight_4,
            device=device
        )
    loss_fns = [
            (LOSS_DICT[loss_1](device=device), weight_1),
            (LOSS_DICT[loss_2](device=device), weight_2) if loss_2 else None,
            (LOSS_DICT[loss_3](device=device), weight_3) if loss_3 else None,
            (LOSS_DICT[loss_4](device=device), weight_4) if loss_4 else None,
        ]
    loss_fns = [item for item in loss_fns if item is not None]

    return lambda yhat, y: sum(weight * fn(yhat, y) for fn, weight in loss_fns)