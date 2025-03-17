"""
AbstractPlanarTruss Class

An abstract base class for planar truss structures in 2D space. This class
inherits from `AbstractStructure` and provides specific implementations
for planar trusses, including member rotation matrix computation, local
stiffness matrix generation, and transformation of stiffness matrices
to the global coordinate system.

This class defines properties and methods specific to 2D planar trusses
while leaving the `generate_structure` method abstract for subclasses
to implement.

Attributes
----------
n_dof : int
    The number of degrees of freedom per node (always 2 for planar trusses).
n_dim : int
    The number of spatial dimensions (always 2 for planar trusses).

Methods
-------
_compute_r(a: float) -> np.ndarray
    Compute the member rotation matrix for a given angle.
compute_k_loc(idx: int) -> np.ndarray
    Compute the local stiffness matrix for an element specified by its index.
compute_k_global(k_loc: np.ndarray, angle: float) -> np.ndarray
    Transform the local stiffness matrix to the global coordinate system.
generate_structure(params: dict)
    Abstract method to generate the structure based on parameters (to be implemented by subclasses).
"""

import numpy as np

from .abstract_structure import *


class AbstractPlanarTruss(AbstractStructure):
    """
    Abstract base class for planar truss structures in 2D.

    This class provides the implementation of methods specific to planar truss systems
    while leaving the structure generation process abstract.
    """

    @property
    def n_dof(self) -> int:
        """
        Get the number of degrees of freedom per node.

        Returns
        -------
        int
            The number of degrees of freedom (always 2 for planar trusses).
        """
        return 2

    @property
    def n_dim(self) -> int:
        """
        Get the number of spatial dimensions.

        Returns
        -------
        int
            The number of dimensions (always 2 for planar trusses).
        """
        return 2

    @staticmethod
    def _get_r(a: float) -> np.ndarray[Any, dtype[np.float64]]:
        """
        Compute the member rotation matrix for a given angle.

        Parameters
        ----------
        a : float
            The angle of rotation (in radians) between the local and global coordinate systems.

        Returns
        -------
        np.ndarray
            The 4x4 rotation matrix used for transforming stiffness matrices.
        """
        c = np.cos(a)
        s = np.sin(a)
        return np.array(
            [
                [c, s, 0, 0],
                [-s, c, 0, 0],
                [0, 0, c, s],
                [0, 0, -s, c]]
        )

    @staticmethod
    def _compute_k_loc(k: float):
        return k * np.array(
            [[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]]
        )

    def compute_k_loc(self, idx: int) -> np.ndarray[Any, dtype[np.float64]]:
        """
        Compute the local stiffness matrix for an element.

        Parameters
        ----------
        idx : int
            The index of the element for which the local stiffness matrix is computed.

        Returns
        -------
        np.ndarray
            The 4x4 local stiffness matrix for the specified element.
        """
        k = ops.basicStiffness(idx)
        return self._compute_k_loc(ops.basicStiffness(idx))

    @classmethod
    def compute_k_global(cls, k_loc: np.ndarray[Any, dtype[np.float64]], angle: float) -> np.ndarray[
        Any, dtype[np.float64]]:
        """
        Transform the local stiffness matrix to the global coordinate system.

        Parameters
        ----------
        k_loc : np.ndarray
            The local stiffness matrix in the local coordinate system.
        angle : float
            The angle of rotation (in radians) between the local and global coordinate systems.

        Returns
        -------
        np.ndarray
            The global stiffness matrix in the global coordinate system.
        """
        r = cls._get_r(angle)
        return r.T @ k_loc @ r

    @abstractmethod
    def generate_structure(self, params: dict) -> None:
        """
        Abstract method to generate the structure.

        This method must be implemented by subclasses to define how the truss
        structure is created, including node definitions, element connections,
        and material properties.

        Parameters
        ----------
        params : dict
            A dictionary containing parameters required to define the truss structure.
        """
        pass
