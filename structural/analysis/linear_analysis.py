from .abstract_analysis import AbstractAnalysis
from openseespy import opensees as ops


class LinearAnalysis(AbstractAnalysis):
    """
    A class for performing linear static analysis using OpenSees.

    This class defines the setup and execution of a linear static analysis
    using OpenSees. It implements the `IAnalysis` interface and provides
    methods to configure and run the analysis.

    Methods
    -------
    setup():
        Configures the OpenSees environment for a linear static analysis.
    run_analysis():
        Sets up the analysis environment and runs the linear static analysis.
    """

    def setup(self):
        """
        Configures the OpenSees environment for a linear static analysis.

        This method sets up the analysis parameters, including the solution
        algorithm, constraint handling, equation numbering, system of equations,
        and analysis type.

        Notes
        -----
        The setup includes:
        - BandSPD system: Solves the equations using the banded symmetric positive-definite system.
        - RCM numbering: Uses the Reverse Cuthill-McKee numbering algorithm.
        - Plain constraints: Uses plain constraint handling.
        - LoadControl integrator: Controls the analysis steps with a step size of 1.0.
        - Linear algorithm: Solves the equations using a linear solution algorithm.
        - Static analysis: Performs a static type of analysis.

        Example
        -------
        >>> analysis = LinearAnalysis()
        >>> analysis.setup()
        """

        ops.system("BandSPD")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")

    def run_analysis(self):
        """
        Sets up and runs the OpenSees linear static analysis.

        This method calls the `setup` method to configure the analysis environment
        and then performs a single analysis step.

        Raises
        ------
        OpenSeesError
            If the analysis encounters an issue, an exception will be raised.

        Example
        -------
        >>> analysis = LinearAnalysis()
        >>> analysis.run_analysis()
        """

        self.setup()
        ops.analyze(1)
