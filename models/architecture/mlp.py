from typing import Iterable, Optional, Any, Dict

from torch import nn
import torch


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) implementation with configurable layers and activations.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : Iterable[int]
        Dimensions of the hidden layers.
    output_dim : int
        Dimensionality of the output features.
    activation : str, default="relu"
        Activation function to use in the hidden layers.
        Supported values: 'relu', 'leaky_relu', 'leakyrelu', 'elu', 'gelu', 'selu', 'tanh', 'sigmoid'.
    activation_params : Optional[Dict[str, Any]], default=None
        Additional parameters to pass to the activation function.
    dropout : float, default=0.0
        Dropout probability. If 0, no dropout is applied.
    batch_norm : bool, default=False
        Whether to use batch normalization after each hidden layer.
    layer_norm : bool, default=False
        Whether to use layer normalization after each hidden layer.
        Note: Only one of batch_norm or layer_norm should be True.
    output_activation : Optional[str], default=None
        Activation function for the output layer. If None, no activation is applied.
        Supports the same values as 'activation'.
    output_activation_params : Optional[Dict[str, Any]], default=None
        Additional parameters to pass to the output activation function.

    Attributes
    ----------
    hidden_layers : nn.Sequential
        The sequence of hidden layers including linear transformations,
        normalization, activation, and dropout.
    output_layer : nn.Linear
        The final linear transformation layer.
    activation_fn : nn.Module
        The activation function used in hidden layers.
    output_activation_fn : Optional[nn.Module]
        The activation function used in the output layer, or None.

    Examples
    --------
    >>> # Simple MLP for classification
    >>> model = MLP(input_dim=784, hidden_dims=[512, 256], output_dim=10,
    ...             dropout=0.2, batch_norm=True)
    >>>
    >>> # MLP for regression with custom activation parameters
    >>> model = MLP(input_dim=10, hidden_dims=[64, 32], output_dim=1,
    ...             activation='leaky_relu', activation_params={'negative_slope': 0.1})
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 output_dim: int,
                 activation: str = "relu",
                 activation_params: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 normalization_params: Optional[Dict[str, Any]] = None,
                 output_activation: Optional[str] = None,
                 output_activation_params: Optional[Dict[str, Any]] = None):

        super(MLP, self).__init__()
        if normalization_params is None:
            normalization_params = {}

        self.activation_fn = self._get_activation_fn(activation, activation_params)

        if output_activation:
            self.output_activation_fn = self._get_activation_fn(output_activation, output_activation_params)
        else:
            self.output_activation_fn = None

        layers = []

        previous_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((nn.Linear(previous_dim, hidden_dim)))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim, **normalization_params))
            elif layer_norm:
                layers.append(nn.LayerNorm(hidden_dim, **normalization_params))

            layers.append(self.activation_fn)

            if dropout > 0.:
                layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        x = self.hidden_layers(x)

        if self.output_activation_fn:
            x = self.output_activation_fn(x)
        return x

    def _get_activation_fn(self, activation, activation_params):
        """
        Creates and returns the specified activation function.

        Parameters
        ----------
        activation : str
            Name of the activation function.
        activation_params : Optional[Dict[str, Any]]
            Parameters to pass to the activation function.

        Returns
        -------
        nn.Module
            The activation module.

        Raises
        ------
        ValueError
            If the activation function name is not supported.
        """
        activation = activation.lower()
        if activation_params is None: activation_params = {}

        match activation:
            case 'relu':
                return nn.ReLU(**activation_params)
            case 'leaky_relu' | 'leakyrelu':
                return nn.LeakyReLU(**activation_params)
            case 'elu':
                return nn.ELU(**activation_params)
            case 'gelu':
                return nn.GELU(**activation_params)
            case 'selu':
                return nn.SELU(**activation_params)
            case 'tanh':
                return nn.Tanh(**activation_params)
            case 'sigmoid':
                return nn.Sigmoid(**activation_params)
            case 'softmax':
                return nn.Softmax(**activation_params)
            case 'softmin':
                return nn.Softmin(**activation_params)
            case 'softplus':
                return nn.Softplus(**activation_params)
            case 'softsign':
                return nn.Softsign(**activation_params)
            case _:
                raise ValueError(f"Unsupported activation function: {activation}")


__all__ = ["MLP"]
