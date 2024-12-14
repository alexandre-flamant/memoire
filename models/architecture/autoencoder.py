from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, size_in, size_out, activation_cls):
        super(Block, self).__init__()
        self.layer = nn.Linear(size_in, size_out, dtype=torch.float64)
        self.activation = activation_cls()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, size_in, size_layers: Iterable[int], encoder_cls=nn.GELU):
        super(Autoencoder, self).__init__()
        a = [size_in, *size_layers[:-1]]
        b = size_layers

        self.encoder = nn.Sequential(OrderedDict([(f"Encoder_{i}", Block(size_in, size_out, encoder_cls))
                                                  for i, (size_in, size_out) in enumerate(zip(a, b))])
                                     )

        self.decoder = nn.Sequential(OrderedDict([(f"Decoder_{i}", Block(size_in, size_out, encoder_cls))
                                                  for i, (size_in, size_out) in enumerate(zip(b[::-1], a[::-1]))])
                                     )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


__all__ = ["Autoencoder"]
