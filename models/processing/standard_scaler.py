from copy import deepcopy
from torch import nn

import torch


class StandardScaler(nn.Module):
    """
    A PyTorch module implementation of StandardScaler for normalizing data.

    This module keeps track of the running mean and standard deviation
    of a dataset and uses them to standardize input data during training
    and inference. It can be integrated into a PyTorch model pipeline.

    Parameters
    ----------
    n_features : int
        The number of features/dimensions in the input data.

    Attributes
    ----------
    n_features : int
        The number of input features.
    sum : torch.Tensor
        Running sum for each feature.
    sum2 : torch.Tensor
        Running sum of squares for each feature.
    mean : torch.Tensor
        Computed mean for each feature.
    std : torch.Tensor
        Computed standard deviation for each feature.
    n : int
        Total number of samples seen.

    Examples
    --------
    >>> scaler = StandardScaler(n_features=10)
    >>> # Fit the scaler with some data
    >>> scaler.fit(train_data)
    >>> # Transform new data
    >>> normalized_data = scaler.transform(test_data)
    >>> # Convert back to original scale
    >>> original_scale_data = scaler.inverse_transform(normalized_data)
    """

    def __init__(self, n_features):
        super(StandardScaler, self).__init__()
        self.n_features = n_features
        self.register_buffer("sum", torch.zeros(n_features))
        self.register_buffer("sum2", torch.zeros(n_features))
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.zeros(n_features))
        self.n = 0

    def fit(self, x: torch.Tensor):
        """
        Resets and fits the scaler to the dataset.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        """
        self.n = x.shape[0]
        self._reset_buffers()
        self._update(x)

    def partial_fit(self, x: torch.Tensor):
        """
        Updates the scaler with additional data.

        This method allows for incremental fitting, which is useful
        for large datasets or online learning.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        """
        self.n += x.shape[0]
        self._update(x)

    def _reset_buffers(self):
        """
        Resets internal sum statistics.

        This is called at the beginning of fit to reset all accumulated statistics.
        """
        self.sum.zero_()
        self.sum2.zero_()
        self.mean.zero_()
        self.std.zero_()

    def _update(self, x: torch.Tensor):
        """
        Updates the mean and std with new data.

        This internal method calculates statistics from input data and updates
        the internal state.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        """
        self.sum += x.sum(dim=0)
        self.sum2 += (x ** 2).sum(dim=0)
        self.mean.copy_(self.sum / self.n)
        self.std.copy_(torch.sqrt(self.sum2 / self.n - self.mean ** 2))

    def transform(self, x: torch.Tensor):
        """
        Standardizes the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Standardized tensor with zero mean and unit variance.

        Raises
        ------
        RuntimeError
            If the scaler has not been fit yet.
        """
        if self.n == 0:
            raise RuntimeError("StandardScaler has not been fit yet.")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor):
        """
        Inverse transformation to original scale.

        Parameters
        ----------
        x : torch.Tensor
            Standardized tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Tensor in the original scale.

        Raises
        ------
        RuntimeError
            If the scaler has not been fit yet.
        """
        if self.n == 0:
            raise RuntimeError("StandardScaler has not been fit yet.")
        return x * self.std + self.mean

    def __str__(self):
        return f"StandardScaler(n_features={self.n_features})\nMean: {self.mean}\nStd: {self.std}"

    def __repr__(self):
        return self.__str__()


__all__ = ['StandardScaler']
