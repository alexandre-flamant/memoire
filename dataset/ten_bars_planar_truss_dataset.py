from typing import List
from collections.abc import Iterable
import h5py
import numpy as np
import torch

from .hdf5_dataset import HDF5AbstractDataset


class TenBarsPlanarTrussDataset(HDF5AbstractDataset):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        with h5py.File(filepath, 'r') as f:
            self.length = f['length'][:].astype(np.float64)
            self.height = f['height'][:].astype(np.float64)
            self.nodes = np.vstack(f['nodes'][:], dtype=np.float64)
            self.young = np.vstack(f['young'][:], dtype=np.float64)
            self.area = np.vstack(f['area'][:], dtype=np.float64)
            self.load = np.vstack(f['load'][:], dtype=np.float64)
            self.nodes_displacement = np.vstack(f['nodes_displacement'][:], dtype=np.float64)
            self.forces = np.vstack(f['force'][:], dtype=np.float64)
            self.stiffness = np.vstack(f['stiffness'][:], dtype=np.float64)

    def __len__(self):
        return len(self.length)

    def __getitem__(self, idx: int | List[int]):
        if isinstance(idx, int):
            return self.__getitems__([idx])[0]
        if isinstance(idx, Iterable):
            return self.__getitems__(idx)
        if isinstance(idx, slice):
            return self.__getitem__(range(idx.start or 0, idx.stop, idx.step or 1))

    def __getitems__(self, idx: List[int]):
        data = np.hstack([
            self.length[idx].reshape((-1, 1)),
            self.height[idx].reshape((-1, 1)),
            self.nodes[idx],
            self.nodes_displacement[idx][:, 2:],
            self.load[idx][:, 9:10],
        ])

        data = torch.tensor(data, dtype=torch.float64)
        target = torch.tensor(self.area[idx] * self.young[idx], dtype=torch.float64)
        nodes = torch.tensor(self.nodes[idx].reshape((-1, 6, 2)), dtype=torch.float64)
        stiffness = torch.tensor(self.stiffness[idx].reshape((-1, 12, 12)), dtype=torch.float64)
        load = torch.tensor(self.load[idx].reshape((-1, 12, 1)), dtype=torch.float64)
        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 12, 1)), dtype=torch.float64)

        return [[data[i], target[i], nodes[i], stiffness[i], displacements[i], load[i]] for i in range(len(idx))]
