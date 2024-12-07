from abc import abstractmethod
from typing import Any

import numpy as np
from numpy import dtype
from openseespy import opensees as ops

from .abstract_structure import AbstractStructure


class AbstractPlanarTruss(AbstractStructure):
    @property
    def n_dof(self) -> int:
        return 2

    @property
    def n_dim(self) -> int:
        return 2

    def _get_r(self, a: float) -> np.ndarray[Any, dtype[np.float64]]:
        """Compute member rotation matrix"""
        c = np.cos(a)
        s = np.sin(a)
        return np.array(
                [[c, s, 0, 0],
                 [-s, c, 0, 0],
                 [0, 0, c, s],
                 [0, 0, -s, c]]
                )

    def compute_k_loc(self, idx: int) -> np.ndarray[Any, dtype[np.float64]]:
        return ops.basicStiffness(idx)*np.array(
                [[1, 0, -1, 0],
                 [0, 0, 0, 0],
                 [-1, 0, 1, 0],
                 [0, 0, 0, 0]]
                )

    def compute_k_global(self, k_loc: np.ndarray[Any, dtype[np.float64]], angle: float) -> np.ndarray[
        Any, dtype[np.float64]]:
        r = self._get_r(angle)
        return r.T@k_loc@r

    @abstractmethod
    def generate_structure(self, params: dict) -> None:
        pass
