from typing import Iterable

import torch
from torch import nn


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int, n_hidden_layers: int,
                 activation_class: nn.Module = nn.LeakyReLU,
                 activation_param: dict = None,
                 output_activation_class: nn.Module = nn.Identity,
                 output_activation_param: dict = None
                 ):

        super(MultiLayerPerceptron, self).__init__()
        if activation_param is None: activation_param = {}
        if output_activation_param is None: output_activation_param = {}

        inner_layers_sizes = [hidden_size for _ in range(n_hidden_layers)]

        in_sizes = [input_size, *inner_layers_sizes]
        out_sizes = [*inner_layers_sizes, output_size]

        for i, (size_in, size_out) in enumerate(zip(in_sizes, out_sizes)):
            self.add_module(f"layer_{i + 1}", nn.Linear(size_in, size_out))
            if i == n_hidden_layers:
                self.add_module(f"activation_{i + 1}", output_activation_class(**output_activation_param))
            else:
                self.add_module(f"activation_{i + 1}", activation_class(**activation_param))

    __all__ = ["MultiLayerPerceptron"]
