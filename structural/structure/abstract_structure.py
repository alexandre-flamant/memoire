from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from numpy import dtype
from openseespy import opensees as ops


class AbstractStructure(ABC):
    """
    Abstract class for creating and managing structural models in OpenSees.

    This abstract base class defines the necessary methods and properties
    for creating, analyzing, and extracting information from structural models.
    Any subclass must implement the methods for generating the structure
    and computing the local and global stiffness matrices.

    Properties
    ----------
    n_dof : int
        The number of degrees of freedom per node.
    n_dim : int
        The number of dimensions of the model 1, 2 or 3.
    n_nodes : int
        The total number of nodes in the model.
    n_elements : int
        The total number of elements in the model.
    nodes_coordinates : np.ndarray
        A 2D array of node coordinates (shape: [n_nodes, n_dim]).
    nodes_displacements : np.ndarray
        A 1D array of node displacements (shape: [n_dof * n_nodes,]).
    elements_connectivity : np.ndarray
        A 2D array describing the connectivity of elements (shape: [n_elements, 2]).
    elements_forces : np.ndarray
        A 1D array of element forces (shape: [n_elements,]).
    loads : np.ndarray
        A 1D array of loads applied to the nodes (shape: [n_dof * n_nodes,]).

    Methods
    -------
    generate_model(params: dict):
        Initializes the OpenSees model and calls the `generate_structure` method to build the model.
    generate_structure(params: dict):
        Abstract method to be implemented by subclasses to generate the structure based on parameters.
    _get_k_loc(idx: int) -> np.ndarray:
        Abstract method to be implemented by subclasses to compute the local stiffness matrix of an element.
    _get_k_global(k_loc: np.ndarray, angle: float) -> np.ndarray:
        Abstract method to be implemented by subclasses to compute the global stiffness matrix from the local stiffness matrix.
    K : np.ndarray
        Computes the global stiffness matrix for the entire structure, including boundary conditions.
    """

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Get the number of degrees of freedom per node."""
        pass

    @property
    @abstractmethod
    def n_dim(self) -> int:
        """Get the number of dimensions (2D or 3D)."""
        pass

    @property
    def supports(self) -> List[List[int | bool]]:
        """Get the supports locations and their fixed dof"""
        nodes = ops.getFixedNodes()
        supports = []
        for idx in range(self.n_nodes):
            if idx not in nodes:
                supports.append([idx, *[False for _ in range(self.n_dof)]])
                continue

            fixed = ops.getFixedDOFs(idx)
            fix = [i+1 in fixed for i in range(self.n_dof)]
            supports.append([idx, *fix])

        return supports

    @property
    def n_nodes(self) -> int:
        """Get the number of nodes in the model."""
        return len(ops.getNodeTags())

    @property
    def n_elements(self) -> int:
        """Get the number of elements in the model."""
        return len(ops.getEleTags())

    @property
    def nodes_coordinates(self) -> np.ndarray[Any, dtype[np.float64]]:
        """Get the coordinates of all nodes."""
        return np.array([ops.nodeCoord(idx) for idx in ops.getNodeTags()], dtype=np.float64)

    @property
    def nodes_displacements(self) -> np.ndarray[Any, dtype[np.float64]]:
        """Get the displacements of all nodes."""
        return np.array([ops.nodeDisp(idx) for idx in ops.getNodeTags()], dtype=np.float64)

    @property
    def elements_connectivity(self) -> np.ndarray[Any, dtype[np.float64]]:
        """Get the connectivity of all elements."""
        return np.array([ops.eleNodes(idx) for idx in ops.getEleTags()], dtype=int)

    @property
    def elements_forces(self) -> np.ndarray[Any, dtype[np.float64]]:
        """Get the forces in all elements."""
        return np.array([ops.basicForce(idx) for idx in ops.getEleTags()], dtype=np.float64)

    @property
    def loads(self) -> np.ndarray[Any, dtype[np.float64]]:
        """Get the loads applied to all nodes."""
        n_dof = self.n_dof
        n_nodes = self.n_nodes

        idx_nodes = ops.getNodeLoadTags()
        load_data = ops.getNodeLoadData()

        q = np.zeros((n_nodes, n_dof), dtype=np.float64)
        for i, idx in enumerate(idx_nodes):
            i *= n_dof
            q[idx, :] = load_data[i:i + n_dof]

        return q

    @abstractmethod
    def compute_k_loc(self, idx: int) -> np.ndarray[Any, dtype[np.float64]]:
        """
        Abstract method to get the local stiffness matrix of an element.

        Parameters
        ----------
        idx : int
            The index of the element for which the local stiffness matrix is to be calculated.

        Returns
        -------
        np.ndarray
            The local stiffness matrix of the element.
        """

        pass

    @abstractmethod
    def compute_k_global(self, k_loc: np.ndarray[Any, dtype[np.float64]], angle: float) -> np.ndarray[
        Any, dtype[np.float64]]:
        """
        Abstract method to get the global stiffness matrix of an element.

        This method transforms the local stiffness matrix into the global coordinate system.

        Parameters
        ----------
        k_loc : np.ndarray
            The local stiffness matrix of the element.
        angle : float
            The angle of rotation from the local coordinate system to the global coordinate system.

        Returns
        -------
        np.ndarray
            The global stiffness matrix of the element.
        """

        pass

    @property
    def stiffness_matrix(self) -> np.ndarray[Any, dtype[np.float64]]:
        """
        Compute the global stiffness matrix for the entire structure.

        This method assembles the global stiffness matrix by summing the contributions
        from each element's stiffness matrix and applying boundary conditions.

        Returns
        -------
        np.ndarray
            The global stiffness matrix for the entire structure.

        Example
        -------
        >>> structure = MyStructure()
        >>> global_stiffness_matrix = structure.stiffness_matrix
        """

        # Parameters
        n_nodes = self.n_nodes
        n_dof = self.n_dof
        nodes = self.nodes_coordinates
        elems = self.elements_connectivity

        elems_vec = np.array([nodes[e] - nodes[s] for s, e in elems])
        elems_angle = np.array([np.arctan2(*v[::-1]) - np.arctan2(0, 1) for v in elems_vec])

        # Stiffness matrix
        K = np.zeros((n_dof*n_nodes, n_dof*n_nodes))

        for idx in range(len(elems)):
            # Get element stiffness matrix
            s_i, e_i = elems[idx]*n_dof
            angle: float = elems_angle[idx]

            k_loc = self.compute_k_loc(idx)
            k_glob = self.compute_k_global(k_loc, angle)

            # Assemble global stiffness matrix
            K[s_i: s_i + n_dof, s_i: s_i + n_dof] += k_glob[0:n_dof, 0:n_dof]
            K[e_i: e_i + n_dof, e_i: e_i + n_dof] += k_glob[n_dof:2*n_dof, n_dof:2*n_dof]
            K[s_i: s_i + n_dof, e_i: e_i + n_dof] += k_glob[0:n_dof, n_dof:2*n_dof]
            K[e_i: e_i + n_dof, s_i: s_i + n_dof] += k_glob[n_dof:2*n_dof, 0:n_dof]

        # Boundary condition
        for idx in range(n_nodes):
            for i in ops.getFixedDOFs(idx):
                dof = n_dof*idx + i - 1  # OSP indices starts at 1

                K[dof, :] = 0.
                K[:, dof] = 0.
                K[dof, dof] = 1.

        return K

    def generate_model(self, params: Dict[str, float | int]):
        """
        Initializes the OpenSees model and generates the structure.

        This method wipes any previous model, sets up the model with the appropriate
        number of degrees of freedom and dimensions, and calls the `generate_structure`
        method to construct the structural elements based on the provided parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing parameters required to generate the structure.

        Example
        -------
        >>> structure = MyStructure()
        >>> structure.generate_model(params)
        """

        ops.wipe()
        ops.model('basic', '-ndm', self.n_dim, '-ndf', self.n_dof)
        self.generate_structure(params)

    @abstractmethod
    def generate_structure(self, params: Dict[str, float | int]) -> None:
        """
        Abstract method to generate the structure based on given parameters.

        This method must be implemented by subclasses to define how to generate
        the structure, including the definition of nodes, elements, and material
        properties based on the provided parameters.

        It assumes both nodes and elements are indexed starting from 0.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters needed to generate the structure.

        Example
        -------

        >>> class MyStructure(AbstractStructure):
        >>>     def generate_structure(self, params):
        >>>         # Implement structure generation logic here
        """
        pass
