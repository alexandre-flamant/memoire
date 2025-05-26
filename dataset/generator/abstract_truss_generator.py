import re
from typing import Dict

import numpy as np

from .abstract_generator import AbstractGenerator


class AbstractTrussGenerator(AbstractGenerator):
    """
    Abstract generator class for truss structures.

    This class provides a reusable implementation of the `construct_result` method,
    which builds a dictionary of computed results and structural parameters, suitable
    for saving to an HDF5 dataset or training a surrogate model.

    Subclasses must define `structure`, `analysis`, and `default_config`.

    Methods
    -------
    construct_result(params)
        Generate a result dictionary containing truss geometry, displacements,
        forces, and derived quantities such as strain and elongation.
    """

    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        """
        Construct and return a dictionary of simulation results and input parameters.

        Parameters
        ----------
        params : dict of str to float or int
            Parameter dictionary used to generate the structure. This should include:
            - "length" and "height" of the truss.
            - "A_i" and "E_i" for cross-sectional areas and Young's moduli of bars.
            - "P_x_i", "P_y_i" for nodal loads.

        Returns
        -------
        dict
            A dictionary containing:

            - 'truss_length' : float
                Total horizontal length of the truss.
            - 'truss_height' : float
                Vertical height of the truss.
            - 'nodes_coordinate' : np.ndarray
                Flattened 2D coordinates of all nodes.
            - 'nodes_displacement' : np.ndarray
                Flattened array of node displacements after analysis.
            - 'nodes_load' : np.ndarray
                Flattened array of external nodal loads applied to the system.
            - 'bars_area' : np.ndarray
                Cross-sectional areas of the bars (ordered).
            - 'bars_young' : np.ndarray
                Young’s modulus values of the bars (ordered).
            - 'bars_force' : np.ndarray
                Internal forces in each truss element.
            - 'bars_length_init' : np.ndarray
                Initial lengths of the bars before deformation.
            - 'bars_elongation' : np.ndarray
                Elongation of each bar (deformed - undeformed length).
            - 'bars_strain' : np.ndarray
                Strain in each bar (elongation / initial length).
            - 'stiffness_matrix' : np.ndarray
                Flattened global stiffness matrix (post-boundary conditions).
        """
        keys = params.keys()

        keys_a = sorted([s for s in keys if re.match("A_[0-9]*", s)])
        keys_e = sorted([s for s in keys if re.match("E_[0-9]*", s)])
        keys_p = sorted([s for s in keys if re.match("P_[x,y]_[0-9]*", s)])
        keys_p = tuple(zip(keys_p[:len(keys_p) // 2], keys_p[len(keys_p) // 2:]))

        bars_elongation = self.structure.initial_elements_length - self.structure.deformed_elements_length

        r = {
            'truss_length': params['length'],
            'truss_height': params['height'],
            'nodes_coordinate': self.structure.nodes_coordinates.reshape(-1),
            'nodes_displacement': self.structure.nodes_displacements.reshape(-1),
            'nodes_load': np.array([[params[k] for k in ks] for ks in keys_p]).reshape(-1),
            'bars_area': np.array([params[k] for k in keys_a]),
            'bars_young': np.array([params[k] for k in keys_e]),
            'bars_force': self.structure.elements_forces.reshape(-1),
            'bars_length_init': self.structure.initial_elements_length,
            'bars_elongation': bars_elongation,
            'bars_strain': bars_elongation / self.structure.initial_elements_length,
            'stiffness_matrix': self.structure.stiffness_matrix.reshape(-1),
        }

        return r
