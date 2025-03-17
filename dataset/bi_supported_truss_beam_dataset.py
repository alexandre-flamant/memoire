from typing import List

import numpy as np
import torch

from .abstract_truss_dataset import AbstractTrussDataset


class BiSupportedTrussBeamDataset(AbstractTrussDataset):
    def __getitems__(self, idx: List[int]):
        n_nodes = 10
        data = np.hstack([
            self.truss_height[idx].reshape(-1, 1),
            self.truss_length[idx].reshape(-1, 1),
            self.bars_length_init[idx][:, [0,8,13]],
            self.nodes_displacement[idx][:, [i for i in range(10)] + [i for i in range(12,18)]],
            self.nodes_load[idx][:, [5]],
            self.bars_strain[idx]
        ])

        data = torch.tensor(data, dtype=self.dtype)
        target = torch.tensor(self.bars_area[idx] * self.bars_young[idx], dtype=self.dtype)
        nodes = torch.tensor(self.nodes_coordinate[idx].reshape((-1, n_nodes, 2)), dtype=self.dtype)
        load = torch.tensor(self.nodes_load[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)
        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)

        return [[data[i], target[i], nodes[i], displacements[i], load[i]] for i in range(len(idx))]


class BiSupportedTrussBeamSingleEADataset(BiSupportedTrussBeamDataset):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]
