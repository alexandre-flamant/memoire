from typing import Dict

from dataset.structural.analysis import LinearAnalysis
from dataset.structural.structure import BiSupportedTrussBeam
from .abstract_generator import ConfigDict
from .abstract_truss_generator import AbstractTrussGenerator


class BiSupportedTrussBeamGenerator(AbstractTrussGenerator):
    """
    Generator class for bi-supported truss beam datasets.

    This generator defines default structural and loading parameters
    for a 2D truss beam with two supports and 21 bars. It uses constant
    distributions by default but can be overridden via configuration.

    Structure generation is delegated to `BiSupportedTrussBeam`, and the
    analysis is run using a linear static solver.

    Parameters
    ----------
    config : ConfigDict
        Configuration dictionary for parameter distributions and sample count.

    Attributes
    ----------
    structure : BiSupportedTrussBeam
        The structural model used for generation.
    analysis : LinearAnalysis
        The analysis method used to evaluate the structure.
    default_config : dict
        Dictionary of default parameter distributions (can be overridden).
    """

    def __init__(self, config: ConfigDict):
        """
        Initialize the bi-supported truss beam generator.

        Parameters
        ----------
        config : ConfigDict
            Configuration specifying sample size and parameter distributions.
        """
        super().__init__(config)
        self.__structure = BiSupportedTrussBeam()
        self.__analysis = LinearAnalysis()

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        """
        Default configuration of parameters for the generator.

        Returns
        -------
        dict
            A dictionary mapping each parameter name to a distribution spec.
            Shared keys are used to link similar parameters together.

        Notes
        -----
        - All loads default to 0.
        - All cross-sectional areas default to 1e-3.
        - All Young’s moduli default to 70e9.
        - The beam has:
            - 21 bars → A_0 to A_20, E_0 to E_20
            - 10 nodes → P_x_0 to P_x_9, P_y_0 to P_y_9
        """
        config = {
            '__load__': {'distribution': 'constant', 'value': 0.},
            '__area__': {'distribution': 'constant', 'value': 1.e-3},
            '__young__': {'distribution': 'constant', 'value': 70.e9},
            'length': {'distribution': 'constant', 'value': 8},
            'height': {'distribution': 'constant', 'value': 2},
        }

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(21)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(21)})
        config.update({f"P_x_{i}": {'shared_with': '__load__'} for i in range(10)})
        config.update({f"P_y_{i}": {'shared_with': '__load__'} for i in range(10)})

        return config

    @property
    def structure(self) -> BiSupportedTrussBeam:
        """
        Returns the truss structure used for generation.

        Returns
        -------
        BiSupportedTrussBeam
            The structure object used to generate the finite element model.
        """
        return self.__structure

    @property
    def analysis(self) -> LinearAnalysis:
        """
        Returns the structural analysis object.

        Returns
        -------
        LinearAnalysis
            The linear static analysis used to compute displacements and forces.
        """
        return self.__analysis
