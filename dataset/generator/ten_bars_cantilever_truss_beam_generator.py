from typing import Dict

from dataset.structural.analysis import LinearAnalysis
from dataset.structural.structure import TenBarsCantileverTruss
from .abstract_generator import ConfigDict
from .abstract_truss_generator import AbstractTrussGenerator


class TenBarsCantileverTrussGenerator(AbstractTrussGenerator):
    """
    Generator for the 10-bar cantilever truss benchmark structure.

    This generator constructs samples of a symmetric 2D cantilever truss with
    10 bars and 6 nodes. Fixed boundary conditions are applied at one end,
    with external loads and material properties defined by the configuration.

    Parameters
    ----------
    config : ConfigDict or str or None
        Configuration dictionary, YAML file path, or None to use defaults.

    Attributes
    ----------
    structure : TenBarsCantileverTruss
        Structural model used to create the truss geometry and topology.
    analysis : LinearAnalysis
        Analysis method used to compute nodal displacements and bar forces.
    default_config : dict
        Dictionary specifying default parameter distributions for sampling.
    """

    def __init__(self, config: ConfigDict | str | None = None):
        """
        Initialize the ten-bar cantilever truss generator.

        Parameters
        ----------
        config : ConfigDict or str or None
            Configuration for sample generation. If None, uses default settings.
        """
        super().__init__(config)
        self.__structure = TenBarsCantileverTruss()
        self.__analysis = LinearAnalysis()

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        """
        Default configuration for sampling truss parameters.

        Returns
        -------
        dict
            Dictionary with default parameter distributions for geometry,
            loading, and material properties.

        Notes
        -----
        - 10 bars: A_0 to A_9 and E_0 to E_9
        - 6 nodes: P_x_0 to P_y_5
        """
        config = {
            '__load__': {'distribution': 'constant', 'value': 0.},
            '__area__': {'distribution': 'constant', 'value': 1.e-3},
            '__young__': {'distribution': 'constant', 'value': 70.e9},
            'length': {'distribution': 'constant', 'value': 2},
            'height': {'distribution': 'constant', 'value': 2},
        }

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(10)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(10)})
        config.update({f"P_x_{i}": {'shared_with': '__load__'} for i in range(6)})
        config.update({f"P_y_{i}": {'shared_with': '__load__'} for i in range(6)})

        return config

    @property
    def structure(self) -> TenBarsCantileverTruss:
        """
        Structural model used for truss generation.

        Returns
        -------
        TenBarsCantileverTruss
            The initialized truss model.
        """
        return self.__structure

    @property
    def analysis(self) -> LinearAnalysis:
        """
        Analysis method used to compute structural response.

        Returns
        -------
        LinearAnalysis
            Linear static analysis solver.
        """
        return self.__analysis
