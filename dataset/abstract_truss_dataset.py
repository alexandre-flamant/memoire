from collections.abc import Iterable
from typing import List

import h5py
import numpy as np
import torch

from .abstract_hdf5_dataset import AbstractHDF5Dataset


class AbstractTrussDataset(AbstractHDF5Dataset):
    """
    Abstract dataset for truss structure data stored in an HDF5 file.

    This class loads various truss-related attributes such as geometry,
    material properties, and mechanical response from an HDF5 file. It
    serves as a base class for datasets used in structural simulations
    or machine learning models involving trusses.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file containing the dataset.
    dtype : torch.dtype, optional
        Data type to which tensors should be cast when returned.
        Default is `torch.float32`.

    Attributes
    ----------
    dtype : torch.dtype
        The target data type for PyTorch tensors.
    truss_height : np.ndarray
        Array of truss heights.
    truss_length : np.ndarray
        Array of truss total lengths.
    nodes_coordinate : np.ndarray
        Node coordinates.
    nodes_displacement : np.ndarray
        Displacement vectors at each node.
    nodes_load : np.ndarray
        Load vectors applied to nodes.
    bars_area : np.ndarray
        Cross-sectional area of each bar.
    bars_young : np.ndarray
        Young's modulus for each bar.
    bars_force : np.ndarray
        Internal force in each bar.
    bars_length_init : np.ndarray
        Initial (undeformed) length of each bar.
    bars_elongation : np.ndarray
        Elongation of each bar.
    bars_strain : np.ndarray
        Strain in each bar.
    stiffness_matrix : np.ndarray
        Global stiffness matrix of the truss.
    """

    def __init__(self, filepath: str, dtype=torch.float32):
        super().__init__(filepath)
        self.dtype = dtype
        with h5py.File(filepath, 'r') as f:
            self.truss_height       = f['truss_height'][:].astype(np.float64)
            self.truss_length       = f['truss_length'][:].astype(np.float64)
            self.nodes_coordinate   = np.vstack(f['nodes_coordinate'][:],   dtype=np.float64)
            self.nodes_displacement = np.vstack(f['nodes_displacement'][:], dtype=np.float64)
            self.nodes_load         = np.vstack(f['nodes_load'][:],         dtype=np.float64)
            self.bars_area          = np.vstack(f['bars_area'][:],          dtype=np.float64)
            self.bars_young         = np.vstack(f['bars_young'][:],         dtype=np.float64)
            self.bars_force         = np.vstack(f['bars_force'][:],         dtype=np.float64)
            self.bars_length_init   = np.vstack(f['bars_length_init'][:],   dtype=np.float64)
            self.bars_elongation    = np.vstack(f['bars_elongation'][:],    dtype=np.float64)
            self.bars_strain        = np.vstack(f['bars_strain'][:],        dtype=np.float64)
            self.stiffness_matrix   = np.vstack(f['stiffness_matrix'][:],   dtype=np.float64)

    def __len__(self):
        """
        Returns the number of truss samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.truss_height)
