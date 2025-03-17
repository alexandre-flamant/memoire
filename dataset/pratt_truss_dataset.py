from typing import List, Callable

import numpy as np
import torch

from dataset import AbstractHDF5Dataset
import h5py


class FixedPrattTrussDataset(AbstractHDF5Dataset):
    def __init__(self,
                 filepath: str,
                 f_noise_length: Callable[[int], float] | None = None,
                 f_noise_loads: Callable[[int], float] | None = None,
                 f_noise_strain: Callable[[int], float] | None = None,
                 f_noise_displacement: Callable[[int], float] | None = None,
                 dtype=torch.float32):

        super().__init__(filepath)

        # Noise configuration
        self.f_noise_length = f_noise_length
        self.f_noise_strain = f_noise_strain
        self.f_noise_displacement = f_noise_displacement
        self.f_noise_loads = f_noise_loads

        if f_noise_length is None:       self.f_noise_length = lambda size: np.ones(size)
        if f_noise_loads is None:        self.f_noise_loads = lambda size: np.ones(size)
        if f_noise_displacement is None: self.f_noise_displacement = lambda size: np.ones(size)
        if f_noise_strain is None:       self.f_noise_strain = lambda size: np.ones(size)

        # Database extraction
        self.dtype = dtype
        with h5py.File(filepath, 'r') as f:
            self.height = f['height'][:].astype(np.float64)
            self.length = f['length'][:].astype(np.float64)
            self.n_panels = f['n_panels'][:].astype(np.int64)
            self.nodes_coordinate = np.vstack(f['nodes_coordinate'][:], dtype=np.float64)
            self.nodes_displacement = np.vstack(f['nodes_displacement'][:], dtype=np.float64)
            self.load = np.vstack(f['nodes_load'][:], dtype=np.float64)
            self.bars_area = np.vstack(f['bars_area'][:], dtype=np.float64)
            self.bars_young = np.vstack(f['bars_young'][:], dtype=np.float64)
            self.bars_force = np.vstack(f['bars_force'][:], dtype=np.float64)
            self.bars_length_init = np.vstack(f['bars_length_init'][:], dtype=np.float64)
            self.bars_elongation = np.vstack(f['bars_elongation'][:], dtype=np.float64)
            self.bars_strain = np.vstack(f['bars_strain'][:], dtype=np.float64)
            self.stiffness_matrix = np.vstack(f['stiffness_matrix'][:], dtype=np.float64)

        self.noise_length = self.f_noise_length(self.height.shape)
        self.noise_truss_width = self.f_noise_length(self.length.shape)
        self.noise_bars_length_init = self.f_noise_length(self.bars_length_init.shape)
        self.noise_nodes_displacement = self.f_noise_displacement(self.nodes_displacement.shape)
        self.noise_load = self.f_noise_loads(self.load.shape)
        noise = self.f_noise_strain(self.bars_force.shape)
        self.noise_bars_force = noise
        self.noise_bars_strain = noise
        self.noise_bars_elongation = noise

    def __getitems__(self, idx: List[int]):
        n_nodes = len(self.nodes_coordinate[0]) // 2

        data_1 = self.nodes_displacement[idx] * self.noise_nodes_displacement[idx]
        data_1 = data_1[:, [k for k in range(4 * self.n_panels[0]) if k not in (0, 1, 2 * self.n_panels[0] + 1)]]
        data_2 = self.load[idx] * self.noise_load[idx]
        data_2 = data_2[:, [i for i in range(3, self.n_panels[0] * 2, 2)]]
        data_3 = self.bars_strain[idx] * self.noise_bars_strain[idx]
        data = np.hstack([data_1, data_2, data_3])

        # Data isolation
        data = torch.tensor(data, dtype=self.dtype)
        target = torch.tensor(self.bars_area[idx] * self.bars_young[idx], dtype=self.dtype)
        nodes = torch.tensor(self.nodes_coordinate[idx].reshape((-1, n_nodes, 2)), dtype=self.dtype)
        load = torch.tensor(self.load[idx].reshape((-1, 2*n_nodes, 1)), dtype=self.dtype)
        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)

        return [[data[i], target[i], nodes[i], displacements[i], load[i]] for i in range(len(idx))]

    def __len__(self):
        return self.height.__len__()


class FixedPrattTrussDatasetSingleTarget(FixedPrattTrussDataset):
    def __init__(self,
                 filepath: str,
                 f_noise_length: Callable[[int], float] | None = None,
                 f_noise_loads: Callable[[int], float] | None = None,
                 f_noise_strain: Callable[[int], float] | None = None,
                 f_noise_displacement: Callable[[int], float] | None = None,
                 dtype=torch.float32):
        super().__init__(filepath=filepath,
                         f_noise_length=f_noise_length,
                         f_noise_loads=f_noise_loads,
                         f_noise_strain=f_noise_strain,
                         f_noise_displacement=f_noise_displacement,
                         dtype=dtype)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]
