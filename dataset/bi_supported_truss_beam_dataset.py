from typing import List

import numpy as np
import torch

from .abstract_truss_dataset import AbstractTrussDataset


class BiSupportedTrussBeamDataset(AbstractTrussDataset):
    """
    Dataset class for bi-supported truss beam simulations.

    This dataset constructs feature-target pairs for supervised learning or
    simulation tasks involving truss beams with bi-supported boundary conditions.
    It retrieves structured input features (geometry, strain, displacement, load)
    and learning targets (axial stiffness: area × Young’s modulus) for each sample.

    In addition, reshaped versions of the node coordinates, loads, and displacements
    are returned to assist in visualization or downstream physics computations.

    Methods
    -------
    __getitems__(idx)
        Returns a list of data and targets for the specified indices.
    """

    def __getitems__(self, idx: List[int]):
        """
        Retrieves a batch of structured truss beam data for the given indices.

        Parameters
        ----------
        idx : List[int]
            List of integer indices representing the samples to fetch.

        Returns
        -------
        samples : List[List[torch.Tensor]]
            A list of samples, where each sample is a list containing:

            - data : torch.Tensor, shape (n_features,)
                Concatenated input features including:
                [height, length, selected bar lengths, selected displacements, load, strain]

            - target : torch.Tensor, shape (n_bars,)
                Element-wise product of bar area and Young’s modulus.

            - nodes : torch.Tensor, shape (n_nodes, 2)
                Node coordinates.

            - displacements : torch.Tensor, shape (2 * n_nodes, 1)
                Nodal displacement vectors.

            - load : torch.Tensor, shape (2 * n_nodes, 1)
                Applied nodal forces.
        """
        n_nodes = 10
        data = np.hstack([
            self.truss_height[idx].reshape(-1, 1),
            self.truss_length[idx].reshape(-1, 1),
            self.bars_length_init[idx][:, [0, 8, 13]],
            self.nodes_displacement[idx][:, [i for i in range(10)] + [i for i in range(12, 18)]],
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
    """
    Dataset variant where all bars share the same cross-sectional area and Young’s modulus.

    This subclass simplifies the data by keeping only the first column from the
    `bars_area` and `bars_young` matrices. This effectively treats the truss as
    having uniform axial stiffness across all bars.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file containing the truss data.
    """

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]
