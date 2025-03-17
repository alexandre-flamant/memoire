from typing import List

import torch
from torch import nn


class CombinedLoss(nn.Module):
    def __init__(self, *losses: List[nn.Module], factor=None):
        super(CombinedLoss, self).__init__()
        if factor is not None:
            factor = torch.tensor([i / len(losses) for i in range(len(losses))])
        if len(factor) != len(losses):
            raise ValueError("losses and factor must have the same length")

        self.losses = nn.ModuleList(losses)
        self.factor = factor

    def forward(self, x):
        loss = torch.tensor(0.0)
        for loss_fn, factor in zip(self.losses, self.factor):
            loss += loss_fn(x) * factor

        return loss


__all__ = ["CombinedLoss"]
