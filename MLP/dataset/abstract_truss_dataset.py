from collections.abc import Iterable
from typing import List

import h5py
import numpy as np
import torch

from .abstract_hdf5_dataset import AbstractHDF5Dataset


class AbstractTrussDataset(AbstractHDF5Dataset):
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
        return len(self.truss_height)
