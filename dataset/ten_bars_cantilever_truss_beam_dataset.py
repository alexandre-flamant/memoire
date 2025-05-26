from typing import List

import numpy as np
import torch

from .abstract_truss_dataset import AbstractTrussDataset


class TenBarsCantileverTrussDataset(AbstractTrussDataset):
    """
    Dataset for 10-bar cantilever truss simulations.

    Loads structural and mechanical data for a cantilever truss model with 10 bars.
    Returns formatted features and targets suitable for regression or surrogate modeling,
    including geometry, displacements, loads, strains, and axial stiffness.

    Methods
    -------
    __getitems__(idx)
        Retrieve structured input-target samples for the specified indices.
    """

    def __getitems__(self, idx: List[int]):
        """
        Retrieves a batch of structured cantilever truss samples.

        Parameters
        ----------
        idx : List[int]
            List of indices for the samples to retrieve.

        Returns
        -------
        samples : List[List[torch.Tensor]]
            A list of samples, where each sample is a list containing:

            - data : torch.Tensor, shape (n_features,)
                Concatenated features: [height, length, bar lengths, selected displacements, load, strain].
            - target : torch.Tensor, shape (n_bars,)
                Axial stiffness per bar (area × Young’s modulus).
            - nodes : torch.Tensor, shape (n_nodes, 2)
                Node coordinates.
            - displacements : torch.Tensor, shape (2 * n_nodes, 1)
                Nodal displacements.
            - load : torch.Tensor, shape (2 * n_nodes, 1)
                Nodal load vectors.
        """
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
    """
    Variant of `TenBarsCantileverTrussDataset` using a single EA (stiffness) target per sample.

    Base target is assumed to be from first bar of each sample.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file containing the truss dataset.
    """

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]