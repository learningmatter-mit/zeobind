import torch
import numpy as np
import pandas as pd
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, yhat, y):
        y = self.format_y(y, yhat)
        yhat = self.format_yhat(yhat)
        return self.loss(yhat, y)

    def format_y(self, y, yhat):
        if type(y) != torch.Tensor:
            y = torch.tensor(y, device=self.device)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def format_yhat(self, yhat):
        if type(yhat) != torch.Tensor:
            yhat = torch.tensor(yhat, device=self.device)
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1, 1)
        return yhat

    def loss(self, yhat, y):
        try:
            return self.loss_fn(yhat, y)
        except Exception as e:
            return self.loss_fn(yhat, y.long())


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


LOSS_DICT = dict(
    rmse=RMSELoss, 
    mse=MSELoss, 
    celoss=CrossEntropyLoss
    )

def get_loss_fn(loss_1, loss_2=None, weight_1=1.0, weight_2=0.0, **kwargs):
    """
    Returns:
        Loss function for a single task prediction."""
    loss_fn_1 = LOSS_DICT[loss_1](device=kwargs.get("device", "cpu"))

    if loss_2 is not None:
        loss_fn_2 = LOSS_DICT[loss_2](device=kwargs.get("device", "cpu"))
        return lambda yhat, y: weight_1 * loss_fn_1(yhat, y) + weight_2 * loss_fn_2(
            yhat, y
        )

    return lambda yhat, y: weight_1 * loss_fn_1(yhat, y)
