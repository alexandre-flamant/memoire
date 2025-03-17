import mlflow
from mlflow.data.dataset import Dataset
import hashlib
from typing import Dict, Any, Optional


class BasicDataset(Dataset):
    def __init__(
            self,
            name: str,
            source_file: str,
            size: int,
            kfold: Optional[int] = None,
            description: Optional[str] = None
    ):

        self._source_file = source_file
        self._size = size
        self._kfold = kfold
        self._description = description

        metadata_str = f"{name}|{source_file}|{size}"
        super().__init__(name=name, digest=source_file, source=source_file)

    def read_data(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_file": self._source_file,
            "size": self._size,
            "kfold": self._kfold,
            "description": self._description
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "source_type": "file",
            "source": self._source_file,
            "size": self._size,
            "kfold": self._kfold,
            "description": self._description
        }
