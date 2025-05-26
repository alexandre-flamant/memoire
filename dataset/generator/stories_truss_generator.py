import numpy as np
import re
from typing import Dict

from dataset.structural.analysis import LinearAnalysis
from dataset.structural.structure import SeismicStoriesTruss
from .abstract_truss_generator import AbstractTrussGenerator


class SeismicTwoStoriesTrussGenerator(AbstractTrussGenerator):
    """
    Generator for a two-story seismic truss subjected to lateral loading.

    This class defines a 2D truss structure with two vertical stories and one horizontal span.
    A lateral seismic load is applied across the top nodes based on a total horizontal force `P`.

    The configuration includes geometric parameters and uniform bar properties. Results include
    displacements, forces, stiffness matrix, and strain data per element.

    Parameters
    ----------
    config : dict or str or None
        Configuration for sample generation. Can be a dictionary, YAML path, or None (uses defaults).

    Attributes
    ----------
    structure : SeismicStoriesTruss
        Structure model with top-floor horizontal load distribution logic.
    analysis : LinearAnalysis
        Structural solver that computes nodal displacements and element forces.
    default_config : dict
        Default sampling configuration for geometry, loading, and material parameters.
    """

    def __init__(self, config: Dict[str, int | float] | str | None = None):
        """
        Initialize the seismic two-story truss generator.

        Parameters
        ----------
        config : dict or str or None
            Sampling configuration. If None, the default config is used.
        """
        super().__init__(config)
        self.__structure = SeismicStoriesTruss()
        self.__analysis = LinearAnalysis()

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        """
        Default parameter configuration for the generator.

        Returns
        -------
        dict
            Dictionary specifying constant distributions for structural parameters.

        Notes
        -----
        - 10 bars are assumed (A_0 to A_9, E_0 to E_9).
        - The lateral load `P` is split across the top nodes during structure generation.
        """
        config = {
            '__area__': {'distribution': 'constant', 'value': 1.e-2},
            '__young__': {'distribution': 'constant', 'value': 70.e9},
            'n_stories': {'distribution': 'constant_int', 'value': 2},
            'n_spans': {'distribution': 'constant_int', 'value': 1},
            'width': {'distribution': 'constant', 'value': 6},
            'height': {'distribution': 'constant', 'value': 4},
            'P': {'distribution': 'constant', 'value': 1000e3},
        }

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(10)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(10)})

        return config

    @property
    def structure(self) -> SeismicStoriesTruss:
        """
        Structure object used for generating the truss model.

        Returns
        -------
        SeismicStoriesTruss
            The configured seismic truss structure.
        """
        return self.__structure

    @property
    def analysis(self) -> LinearAnalysis:
        """
        Structural analysis method used for the simulation.

        Returns
        -------
        LinearAnalysis
            Linear solver for displacements and internal forces.
        """
        return self.__analysis

    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        """
        Generate result dictionary for one sample of the seismic truss model.

        Parameters
        ----------
        params : dict
            Dictionary of sampled input parameters used for structure generation.

        Returns
        -------
        dict
            Dictionary containing simulation results and input metadata:

            - 'truss_width' : float
                Total width of the truss structure.
            - 'truss_height' : float
                Total height of the truss structure.
            - 'load' : float
                Total lateral seismic load applied at the top floor.
            - 'nodes_coordinate' : np.ndarray
                Flattened array of nodal positions.
            - 'nodes_displacement' : np.ndarray
                Flattened array of displacements from analysis.
            - 'bars_area' : np.ndarray
                Cross-sectional area of each bar.
            - 'bars_young' : np.ndarray
                Young’s modulus for each bar.
            - 'bars_force' : np.ndarray
                Internal force in each bar element.
            - 'bars_length_init' : np.ndarray
                Initial lengths of all elements.
            - 'bars_elongation' : np.ndarray
                Difference between initial and deformed lengths.
            - 'bars_strain' : np.ndarray
                Axial strain (elongation / initial length).
            - 'stiffness_matrix' : np.ndarray
                Flattened global stiffness matrix.
        """
        keys = params.keys()

        keys_a = sorted([s for s in keys if re.match(r"A_\d+", s)])
        keys_e = sorted([s for s in keys if re.match(r"E_\d+", s)])

        bars_elongation = self.structure.initial_elements_length - self.structure.deformed_elements_length

        r = {
            'truss_width': params['width'],
            'truss_height': params['height'],
            'load': params['P'],
            'nodes_coordinate': self.structure.nodes_coordinates.reshape(-1),
            'nodes_displacement': self.structure.nodes_displacements.reshape(-1),
            'bars_area': np.array([params[k] for k in keys_a]),
            'bars_young': np.array([params[k] for k in keys_e]),
            'bars_force': self.structure.elements_forces.reshape(-1),
            'bars_length_init': self.structure.initial_elements_length,
            'bars_elongation': bars_elongation,
            'bars_strain': bars_elongation / self.structure.initial_elements_length,
            'stiffness_matrix': self.structure.stiffness_matrix.reshape(-1),
        }

        return r
