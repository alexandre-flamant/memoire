from pathlib import Path
from typing import List, Iterable
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import numpy as np

class AbstractHDF5Dataset(Dataset, ABC):
    def __init__(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError(f"{filepath} file does not exists.")
        if not filepath.is_file():
            raise ValueError("filepath argument must point to a file.")
        if not filepath.suffix in ('.hdf5', '.h5', '.he5'):
            raise ValueError("filepath argument must point to a .hdf5 or .h5 file.")

        self.filepath = filepath

    def __getitem__(self, idx: int | List[int]):
        if np.issubdtype(type(idx), np.integer):
            return self.__getitems__([idx])[0]
        if isinstance(idx, Iterable):
            return self.__getitems__(idx)
        if isinstance(idx, slice):
            return self.__getitem__(range(idx.start or 0, idx.stop or self.truss_height, idx.step or 1))

    def __str__(self):
        return f"{self.__class__.__name__} loaded from {self.filepath}"

    @abstractmethod
    def __getitems__(self, idx: List[int]):
        raise NotImplemented()