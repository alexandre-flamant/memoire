import torch
from torch import nn


class MinMaxScaler(nn.Module):
    """
    A PyTorch module implementation of MinMaxScaler for normalizing data.

    This module keeps track of the minimum and maximum values in a dataset
    and uses them to scale input data to a range of [0, 1] during training
    and inference. It can be integrated into a PyTorch model pipeline.

    Parameters
    ----------
    n_features : int
        The number of features/dimensions in the input data.

    Attributes
    ----------
    n_features : int
        The number of input features.
    min : torch.Tensor
        Minimum value for each feature.
    max : torch.Tensor
        Maximum value for each feature.

    Examples
    --------
    >>> scaler = MinMaxScaler(n_features=10)
    >>> # Fit the scaler with some data
    >>> scaler.fit(train_data)
    >>> # Transform new data to [0, 1] range
    >>> normalized_data = scaler.transform(test_data)
    >>> # Convert back to original scale
    >>> original_scale_data = scaler.inverse_transform(normalized_data)
    """

    def __init__(self, n_features):
        super(MinMaxScaler, self).__init__()
        self.n_features = n_features
        self.register_buffer("min", None)
        self.register_buffer("max", None)

    def fit(self, x: torch.Tensor):
        """
        Fits the scaler to the dataset.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features) or
            a tensor that can be reshaped to that.
        """
        x = x.reshape((-1, self.n_features))
        self.max = x.max(dim=0).values
        self.min = x.min(dim=0).values

    def partial_fit(self, x: torch.Tensor):
        """
        Updates the scaler with additional data.

        This method allows for incremental fitting, which is useful
        for large datasets or online learning.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features) or
            a tensor that can be reshaped to that.
        """
        x = x.reshape((-1, self.n_features))
        if self.max is None:
            self.max = x.max(dim=0).values
            self.min = x.min(dim=0).values
        else:
            self.max = torch.maximum(self.max, x.max(dim=0).values)
            self.min = torch.minimum(self.min, x.min(dim=0).values)

    def transform(self, x: torch.Tensor):
        """
        Scales the input tensor to the [0, 1] range.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_samples, n_features) or
            a tensor that can be reshaped to that.

        Returns
        -------
        torch.Tensor
            Scaled tensor with values in the range [0, 1].

        Raises
        ------
        RuntimeError
            If the scaler has not been fit yet.
        """
        if self.max is None:
            raise RuntimeError("MinMaxScaler has not been fit yet.")
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x: torch.Tensor):
        """
        Inverse transformation to original scale.

        Parameters
        ----------
        x : torch.Tensor
            Scaled tensor of shape (n_samples, n_features) or
            a tensor that can be reshaped to that.

        Returns
        -------
        torch.Tensor
            Tensor in the original scale.

        Raises
        ------
        RuntimeError
            If the scaler has not been fit yet.
        """
        if self.max is None:
            raise RuntimeError("MinMaxScaler has not been fit yet.")
        return x * (self.max - self.min) + self.min

    def __str__(self):
        return f"MinMaxScaler(n_features={self.n_features})\nMin: {self.min}\nMax: {self.max}"

    def __repr__(self):
        return self.__str__()


__all__ = ['MinMaxScaler']
