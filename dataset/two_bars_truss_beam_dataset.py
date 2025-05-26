from typing import List

import numpy as np
import torch

from .abstract_truss_dataset import AbstractTrussDataset


class TwoBarsTrussDataset(AbstractTrussDataset):
    """
    Dataset class for a simple truss structure with two bars.

    This class loads structured data from a truss model consisting of only two
    bars and three nodes. It returns geometric features, loads, displacements,
    strains, and stiffness targets suitable for regression or analysis.

    Methods
    -------
    __getitems__(idx)
        Retrieves structured input-target samples for a batch of indices.
    """

    def __getitems__(self, idx: List[int]):
        """
        Retrieves a batch of structured samples from the two-bar truss dataset.

        Parameters
        ----------
        idx : List[int]
            Indices of the samples to retrieve.

        Returns
        -------
        samples : List[List[torch.Tensor]]
            A list of samples, where each sample is a list containing:

            - data : torch.Tensor, shape (n_features,)
                Input features: [height, length, bar 0 length, displacement at DOF 3,
                load at DOF 3, all bar strains].
            - target : torch.Tensor, shape (n_bars,)
                Axial stiffness of bars (area × Young’s modulus).
            - nodes : torch.Tensor, shape (n_nodes, 2)
                Node coordinates.
            - displacements : torch.Tensor, shape (2 * n_nodes, 1)
                Nodal displacements.
            - load : torch.Tensor, shape (2 * n_nodes, 1)
                Nodal load vectors.
        """
        n_nodes = 3
        data = np.hstack([
            self.truss_height[idx].reshape((-1, 1)),
            self.truss_length[idx].reshape((-1, 1)),
            self.bars_length_init[idx][:, 0].reshape((-1, 1)),
            self.nodes_displacement[idx][:, [3]],
            self.nodes_load[idx][:, [3]],
            self.bars_strain[idx]
        ])

        data = torch.tensor(data, dtype=self.dtype)
        target = torch.tensor(self.bars_area[idx] * self.bars_young[idx], dtype=self.dtype)
        nodes = torch.tensor(self.nodes_coordinate[idx].reshape((-1, n_nodes, 2)), dtype=self.dtype)
        load = torch.tensor(self.nodes_load[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)
        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)

        return [[data[i], target[i], nodes[i], displacements[i], load[i]] for i in range(len(idx))]


class TwoBarsTrussSingleEADataset(TwoBarsTrussDataset):
    """
    Variant of `TwoBarsTrussDataset` that returns only a single stiffness (EA) target per sample.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file containing the two-bar truss dataset.
    """

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]
