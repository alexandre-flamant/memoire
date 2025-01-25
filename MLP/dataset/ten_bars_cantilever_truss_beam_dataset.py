from typing import List

import numpy as np
import torch

from .abstract_truss_dataset import AbstractTrussDataset


class TenBarsCantileverTrussDataset(AbstractTrussDataset):
    def __getitems__(self, idx: List[int]):
        n_nodes = 6
        data = np.hstack([
            self.truss_height[idx].reshape((-1, 1)),
            self.truss_length[idx].reshape((-1, 1)),
            self.bars_length_init[idx],
            self.nodes_displacement[idx][:, [2, 3, 4, 5, 8, 9, 10, 11]],
            self.nodes_load[idx][:, 9:10],
            self.bars_strain[idx]
        ])

        data = torch.tensor(data, dtype=self.dtype)
        target = torch.tensor(self.bars_area[idx] * self.bars_young[idx], dtype=self.dtype)
        nodes = torch.tensor(self.nodes_coordinate[idx].reshape((-1, n_nodes, 2)), dtype=self.dtype)
        load = torch.tensor(self.nodes_load[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)
        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)

        return [[data[i], target[i], nodes[i], displacements[i], load[i]] for i in range(len(idx))]


class TenBarsCantileverTrussSingleEADataset(TenBarsCantileverTrussDataset):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]
