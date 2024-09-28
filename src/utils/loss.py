import torch
import numpy as np
import pandas as pd
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, yhat, y):
        y = self.format_y(y)
        yhat = self.format_yhat(yhat)
        return self.loss(yhat, y)

    def format_y(self, y):
        y = torch.Tensor(y, device=self.device)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def format_yhat(self, yhat):
        yhat = torch.Tensor(yhat, device=self.device)
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1, 1)
        return yhat

    def loss(self, yhat, y):
        raise NotImplementedError


class MSELoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def loss(self, yhat, y):
        loss = nn.MSELoss(yhat, y)
        return loss


class RMSELoss(BaseLoss):
    def __init__(self, eps=1e-16):
        super().__init__()
        self.eps = eps

    def loss(self, yhat, y):
        loss = torch.sqrt(nn.MSELoss(yhat, y) + self.eps)
        return loss


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def loss(self, yhat, y):
        loss = nn.CrossEntropyLoss(yhat, y)
        return loss


LOSS_DICT = dict(rmse=RMSELoss, mse=MSELoss, cross_entropy=CrossEntropyLoss)


def get_loss_fn(loss_1, loss_2=None, weight_1=1.0, weight_2=0.0, device="cpu"):
    """
    Returns:
        Loss function for a single task prediction."""
    loss_fn_1 = LOSS_DICT[loss_1](device=device)

    if loss_2 is not None:
        loss_fn_2 = LOSS_DICT[loss_2](device=device)
        return lambda yhat, y: weight_1 * loss_fn_1(yhat, y) + weight_2 * loss_fn_2(
            yhat, y
        )

    return lambda yhat, y: weight_1 * loss_fn_1(yhat, y)
