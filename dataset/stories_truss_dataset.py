from typing import List, Callable

import numpy as np
import torch

from .abstract_hdf5_dataset import AbstractHDF5Dataset
import h5py


class SeismicTwoStoriesTrussDataset(AbstractHDF5Dataset):
    """
    Dataset class for seismic truss structures with two-story configurations.

    This class loads and structures truss simulation data from HDF5, and supports
    optional noise injection on various inputs such as geometry, loads, strain,
    and displacements. The data is prepared in a format suitable for machine learning
    models or surrogate modeling of truss dynamics under seismic conditions.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file containing the truss simulation data.
    noise_length : Callable[[tuple], np.ndarray], optional
        Function that returns additive noise for truss height, width, and bar lengths.
        Takes a shape tuple as input and returns an array of the same shape.
        Defaults to zero noise.
    noise_loads : Callable[[tuple], np.ndarray], optional
        Function that returns multiplicative noise for scalar loads.
        Takes a shape tuple as input and returns an array of the same shape.
        Defaults to ones (no noise).
    noise_strain : Callable[[tuple], np.ndarray], optional
        Function that returns multiplicative noise for bar strain, elongation, and force.
        Takes a shape tuple as input and returns an array of the same shape.
        Defaults to ones.
    noise_displacement : Callable[[tuple], np.ndarray], optional
        Function that returns additive noise for node displacements.
        Takes a shape tuple as input and returns an array of the same shape.
        Defaults to zero noise.
    dtype : torch.dtype, optional
        Data type to cast tensors. Default is `torch.float32`.

    Attributes
    ----------
    truss_height : np.ndarray
        Vertical dimensions of the truss structure.
    truss_width : np.ndarray
        Horizontal span of the truss.
    nodes_coordinate : np.ndarray
        2D coordinates of each node.
    nodes_displacement : np.ndarray
        Displacement vectors for each node.
    load : np.ndarray
        Scalar load applied to the top of the structure.
    bars_area : np.ndarray
        Cross-sectional area of each bar.
    bars_young : np.ndarray
        Young’s modulus for each bar.
    bars_force : np.ndarray
        Internal forces in each bar.
    bars_length_init : np.ndarray
        Initial lengths of bars before deformation.
    bars_elongation : np.ndarray
        Elongation of each bar after load.
    bars_strain : np.ndarray
        Strain values in each bar.
    stiffness_matrix : np.ndarray
        System stiffness matrix for structural analysis.
    """

    def __init__(self,
                 filepath: str,
                 noise_length: Callable[[tuple], np.ndarray] | None = None,
                 noise_loads: Callable[[tuple], np.ndarray] | None = None,
                 noise_strain: Callable[[tuple], np.ndarray] | None = None,
                 noise_displacement: Callable[[tuple], np.ndarray] | None = None,
                 dtype=torch.float32):

        super().__init__(filepath)

        # Noise configuration
        self.noise_length = noise_length or (lambda size: np.zeros(size))
        self.noise_loads = noise_loads or (lambda size: np.ones(size))
        self.noise_displacement = noise_displacement or (lambda size: np.zeros(size))
        self.noise_strain = noise_strain or (lambda size: np.ones(size))

        # Database extraction
        self.dtype = dtype
        with h5py.File(filepath, 'r') as f:
            self.truss_height = f['truss_height'][:].astype(np.float64)
            self.truss_width = f['truss_width'][:].astype(np.float64)
            self.nodes_coordinate = np.vstack(f['nodes_coordinate'][:], dtype=np.float64)
            self.nodes_displacement = np.vstack(f['nodes_displacement'][:], dtype=np.float64)
            self.load = f['load'][:].astype(np.float64)
            self.bars_area = np.vstack(f['bars_area'][:], dtype=np.float64)
            self.bars_young = np.vstack(f['bars_young'][:], dtype=np.float64)
            self.bars_force = np.vstack(f['bars_force'][:], dtype=np.float64)
            self.bars_length_init = np.vstack(f['bars_length_init'][:], dtype=np.float64)
            self.bars_elongation = np.vstack(f['bars_elongation'][:], dtype=np.float64)
            self.bars_strain = np.vstack(f['bars_strain'][:], dtype=np.float64)
            self.stiffness_matrix = np.vstack(f['stiffness_matrix'][:], dtype=np.float64)

        # Apply noise generators
        self.noise_length_fix = self.noise_length(self.truss_height.shape)
        self.noise_truss_width_fix = self.noise_length(self.truss_width.shape)
        self.noise_bars_length_init_fix = self.noise_length(self.bars_length_init.shape)
        self.noise_nodes_displacement_fix = self.noise_displacement(self.nodes_displacement.shape)
        self.noise_load_fix = self.noise_loads(self.load.shape)
        noise = self.noise_strain(self.bars_force.shape)
        self.noise_bars_force_fix = noise
        self.noise_bars_strain_fix = noise
        self.noise_bars_elongation_fix = noise

    def __getitems__(self, idx: List[int]):
        """
        Retrieve batched dataset items for the specified indices.

        Parameters
        ----------
        idx : List[int]
            Indices of samples to load.

        Returns
        -------
        List[List[torch.Tensor]]
            Each element is a list of:
            - data : torch.Tensor
                Features including height, width, bar length, displacements, load, strain.
            - target : torch.Tensor
                Target stiffness (area × Young’s modulus).
            - nodes : torch.Tensor
                Node coordinates reshaped to (n_nodes, 2).
            - displacements : torch.Tensor
                Full node displacements reshaped to (2 * n_nodes, 1).
            - load : torch.Tensor
                Structured load tensor shaped (2 * n_nodes, 1) with values mapped to specific nodes.
        """
        n_nodes = 6

        data = np.hstack([
            self.truss_height[idx].reshape((-1, 1)) + self.noise_length_fix[idx].reshape((-1, 1)),
            self.truss_width[idx].reshape((-1, 1)) + self.noise_truss_width_fix[idx].reshape((-1, 1)),
            self.bars_length_init[idx][:, [8]] + self.noise_bars_length_init_fix[idx][:, [8]],
            self.nodes_displacement[idx][:, [2, 3, 4, 5, 8, 9, 10, 11]] +
            self.noise_nodes_displacement_fix[idx][:, [2, 3, 4, 5, 8, 9, 10, 11]],
            self.load[idx].reshape((-1, 1)) * self.noise_load_fix[idx].reshape((-1, 1)),
            self.bars_strain[idx] * self.noise_bars_strain_fix[idx]
        ])

        data = torch.tensor(data, dtype=self.dtype)
        target = torch.tensor(self.bars_area[idx] * self.bars_young[idx], dtype=self.dtype)
        nodes = torch.tensor(self.nodes_coordinate[idx].reshape((-1, n_nodes, 2)), dtype=self.dtype)

        _load = self.load[idx]
        load = torch.zeros((len(idx), 2 * n_nodes, 1), dtype=self.dtype)
        for i, l in enumerate(_load):
            load[i, 8, :] = 0.5 * l
            load[i, 10, :] = l

        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)

        return [[data[i], target[i], nodes[i], displacements[i], load[i]] for i in range(len(idx))]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of truss samples.
        """
        return len(self.truss_height)


class SeismicTwoStoriesTrussDatasetSingleTarget(SeismicTwoStoriesTrussDataset):
    """
    Variant of `SeismicTwoStoriesTrussDataset` using only a single target value per sample.

    Reduces the learning target to the first bar's stiffness (area × Young’s modulus),
    useful for simple regression tasks or sensitivity analysis.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    noise_length : Callable[[tuple], np.ndarray], optional
        Noise function for geometry-related inputs.
    noise_loads : Callable[[tuple], np.ndarray], optional
        Noise function for node load inputs.
    noise_strain : Callable[[tuple], np.ndarray], optional
        Noise function for strain/force data.
    noise_displacement : Callable[[tuple], np.ndarray], optional
        Noise function for nodal displacement inputs.
    dtype : torch.dtype, optional
        Tensor data type. Default is `torch.float32`.
    """

    def __init__(self, filepath: str,
                 noise_length: Callable[[tuple], np.ndarray] | None = None,
                 noise_loads: Callable[[tuple], np.ndarray] | None = None,
                 noise_strain: Callable[[tuple], np.ndarray] | None = None,
                 noise_displacement: Callable[[tuple], np.ndarray] | None = None,
                 dtype=torch.float32):
        super().__init__(filepath, noise_length, noise_loads, noise_strain, noise_displacement, dtype)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]
