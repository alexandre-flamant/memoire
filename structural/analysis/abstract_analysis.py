from abc import ABC, abstractmethod


class AbstractAnalysis(ABC):
    """
    An abstract base class (interface) for analysis classes.

    This interface defines the required structure for all analysis classes.
    Any subclass must implement the `setup` and `run_analysis` methods to define
    the behavior for setting up and running a specific type of analysis.

    Methods
    -------
    setup():
        Configures the analysis environment. This method must be implemented
        in subclasses to define how the analysis is prepared.
    run_analysis():
        Sets up and executes the analysis. This method must be implemented
        in subclasses to specify the steps for running the analysis.
    """

    @abstractmethod
    def setup(self):
        """
        Configures the analysis environment.

        This abstract method must be implemented by subclasses to define
        the steps required for preparing the analysis setup. The specific
        implementation will vary depending on the type of analysis.

        Raises
        ------
        NotImplementedError
            If this method is not implemented in a subclass.

        Example
        -------
        Subclasses should override this method:
        >>> class MyAnalysis(AbstractAnalysis):
        >>>     def setup(self):
        >>>         print("Setting up MyAnalysis.")
        """
        pass

    @abstractmethod
    def run_analysis(self):
        """
        Sets up and runs the analysis.

        This abstract method must be implemented by subclasses to define
        the procedure for executing the analysis after the setup. The specific
        implementation will vary depending on the type of analysis.

        Raises
        ------
        NotImplementedError
            If this method is not implemented in a subclass.

        Example
        -------
        Subclasses should override this method:
        >>> class MyAnalysis(AbstractAnalysis):
        >>>     def run_analysis(self):
        >>>         print("Running MyAnalysis.")
        """
        pass
