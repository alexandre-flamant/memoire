from pathlib import Path
from typing import List, Iterable
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import numpy as np

class AbstractHDF5Dataset(Dataset, ABC):
    """
    Abstract base dataset class for HDF5-based data sources.

    This class ensures the file path is valid and delegates item access to
    subclasses through the `__getitems__` method.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file. Must be a file with a `.hdf5`, `.h5`, or `.he5` extension.

    Raises
    ------
    ValueError
        If the file does not exist, is not a file, or does not have a valid HDF5 extension.
    """

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
        """
        Retrieve one or more data items by index.

        Parameters
        ----------
        idx : int, list of int, or slice
            Index or indices specifying which data samples to retrieve.

        Returns
        -------
        data : object
            The data sample(s) corresponding to the provided index/indices.

        Raises
        ------
        TypeError
            If `idx` is not an integer, list of integers, or a slice.
        """
        if np.issubdtype(type(idx), np.integer):
            return self.__getitems__([idx])[0]
        if isinstance(idx, Iterable):
            return self.__getitems__(idx)
        if isinstance(idx, slice):
            return self.__getitem__(range(idx.start or 0, idx.stop or self.__len__(), idx.step or 1))

    def __str__(self):
        """
        String representation of the dataset.

        Returns
        -------
        str
            Description including the class name and file path.
        """
        return f"{self.__class__.__name__} loaded from {self.filepath}"

    @abstractmethod
    def __getitems__(self, idx: List[int]):
        """
        Abstract method to retrieve multiple items by index.

        This method must be implemented by subclasses to define how data is
        loaded from the underlying HDF5 file.

        Parameters
        ----------
        idx : list of int
            List of indices to retrieve.

        Returns
        -------
        data : object
            The data corresponding to the given indices.
        """
        raise NotImplemented()
