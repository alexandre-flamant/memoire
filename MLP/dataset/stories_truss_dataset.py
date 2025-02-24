from typing import List, Callable

import numpy as np
import torch

from .abstract_hdf5_dataset import AbstractHDF5Dataset
import h5py


class SeismicTwoStoriesTrussDataset(AbstractHDF5Dataset):
    def __init__(self,
                 filepath: str,
                 noise_length: Callable[[int], float] | None = None,
                 noise_loads: Callable[[int], float] | None = None,
                 noise_strain: Callable[[int], float] | None = None,
                 noise_displacement: Callable[[int], float] | None = None,
                 dtype=torch.float32):

        super().__init__(filepath)

        # Noise configuration
        self.noise_length = noise_length
        self.noise_strain = noise_strain
        self.noise_displacement = noise_displacement
        self.noise_loads = noise_loads
        if noise_length is None: self.noise_length = lambda size: np.zeros(size)
        if noise_loads is None: self.noise_loads = lambda size: np.ones(size)
        if noise_displacement is None: self.noise_displacement = lambda size: np.zeros(size)
        if noise_strain is None: self.noise_strain = lambda size: np.ones(size)

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
        n_nodes = 6

        data = np.hstack([
            self.truss_height[idx].reshape((-1, 1)) + self.noise_length_fix[idx].reshape((-1, 1)),
            self.truss_width[idx].reshape((-1, 1)) + self.noise_truss_width_fix[idx].reshape((-1, 1)),
            self.bars_length_init[idx] + self.noise_bars_length_init_fix[idx],
            self.nodes_displacement[idx][:, [2, 3, 4, 5, 8, 9, 10, 11]] \
            + self.noise_nodes_displacement_fix[idx][:, [2, 3, 4, 5, 8, 9, 10, 11]],
            self.load[idx].reshape((-1, 1)) + self.noise_load_fix[idx].reshape((-1, 1)),
            self.bars_strain[idx] + self.noise_bars_strain_fix[idx]
        ])

        # Data isolation
        data = torch.tensor(data, dtype=self.dtype)
        target = torch.tensor(self.bars_area[idx] * self.bars_young[idx], dtype=self.dtype)
        nodes = torch.tensor(self.nodes_coordinate[idx].reshape((-1, n_nodes, 2)), dtype=self.dtype)

        _load = self.load[idx]
        load = torch.zeros((len(idx), 2 * n_nodes, 1), dtype=self.dtype)
        for i, l in enumerate(_load):
            load[i, 8, :] = .5 * l
            load[i, 10, :] = l

        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)

        return [[data[i], target[i], nodes[i], displacements[i], load[i]] for i in range(len(idx))]

    def __len__(self):
        return self.truss_height.__len__()
