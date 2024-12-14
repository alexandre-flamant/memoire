from pathlib import Path
from torch.utils.data import Dataset

class HDF5AbstractDataset(Dataset):
    def __init__(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError(f"{filepath} file does not exists.")
        if not filepath.is_file():
            raise ValueError("filepath argument must point to a file.")
        if not filepath.suffix in ('.hdf5', '.h5', '.he5'):
            raise ValueError("filepath argument must point to a .hdf5 or .h5 file.")

        self.filepath = filepath

    def __str__(self):
        return f"{self.__class__.__name__} loaded from {self.filepath}"

