from typing import Iterable

import torch
from torch import nn


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, input_size: int, hidden_size: Iterable[int], output_size: int):
        super(MultiLayerPerceptron, self).__init__()
        a = [input_size, *hidden_size]
        b = [*hidden_size, output_size]

        for i, (size_in, size_out) in enumerate(zip(a, b)):
            self.add_module(f"layer_{i + 1}", nn.Linear(size_in, size_out, dtype=torch.float64))
            self.add_module(f"GELU_{i + 1}", nn.GELU())


__all__ = ["MultiLayerPerceptron"]
