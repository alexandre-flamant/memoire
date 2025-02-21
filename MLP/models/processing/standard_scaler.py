from copy import deepcopy
from torch import nn

import torch

class StandardScaler(nn.Module):
    def __init__(self, n_features):
        super(StandardScaler, self).__init__()
        self.n_features = n_features
        self.register_buffer("sum", torch.zeros(n_features))
        self.register_buffer("sum2", torch.zeros(n_features))
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.zeros(n_features))
        self.n = 0  # Not a tensor since it's a scalar integer

    def fit(self, x: torch.Tensor):
        """Resets and fits the scaler to the dataset."""
        self.n = x.shape[0]
        self._reset_buffers()
        self._update(x)

    def partial_fit(self, x: torch.Tensor):
        """Updates the scaler with additional data."""
        self.n += x.shape[0]
        self._update(x)

    def _reset_buffers(self):
        """Resets internal sum statistics."""
        self.sum.zero_()
        self.sum2.zero_()
        self.mean.zero_()
        self.std.zero_()

    def _update(self, x: torch.Tensor):
        """Updates the mean and std with new data."""
        self.sum += x.sum(dim=0)
        self.sum2 += (x ** 2).sum(dim=0)
        self.mean.copy_(self.sum / self.n)
        self.std.copy_(torch.sqrt(self.sum2 / self.n - self.mean ** 2))

    def transform(self, x: torch.Tensor):
        """Standardizes the input tensor."""
        if self.n == 0:
            raise RuntimeError("StandardScaler has not been fit yet.")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor):
        """Inverse transformation to original scale."""
        if self.n == 0:
            raise RuntimeError("StandardScaler has not been fit yet.")
        return x * self.std + self.mean

    def __str__(self):
        return f"StandardScaler(n_features={self.n_features})\nMean: {self.mean}\nStd: {self.std}"

    def __repr__(self):
        return self.__str__()


__all__ = ['StandardScaler']
