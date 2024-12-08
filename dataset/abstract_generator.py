import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable, Any, TypedDict, Dict

import numpy as np
import yaml

from structural.analysis import AbstractAnalysis
from structural.structure.abstract_structure import AbstractStructure


class ConfigDict(TypedDict, total=False):
    """
    A dictionary type for distribution configuration used in the generator.

    Attributes
    ----------
    n_sample : str or int
        Number of samples to generate.
    parameters : dict of str to dict of {str: str or int or float}
        Dictionary containing parameter names as keys and their configuration as values.
    """
    n_sample: str | int
    parameters: Dict[str, Dict[str, str | int | float]]


class AbstractGenerator(ABC, Iterable):
    """
    Abstract base class for generating structures and analyses based on parameter distributions.

    This class processes configurations provided in a YAML file to generate models and execute analyses.
    Subclasses must define the associated structure and analysis objects, as well as default parameter configurations.

    The YAML needs to be parsed in a dictionary before being used.

    YAML Configuration
    ------------------
    The configuration YAML file contains two main sections:
    - `n_samples` (int): The number of samples to generate.
    - `parameters`: A dictionary where:
        - Keys are parameter names.
        - Values define either:
          - `shared_with` (str): Links the parameter to another parameter's distribution.
          - A distribution description adhering to `get_distribution` specifications.

    Attributes
    ----------
    config : ConfigDict
        Dictionary containing sampling configurations and parameter descriptions.

    Example
    -------
    Example of a YAML configuration file:

    n_samples: 5000
    parameters:
        shared_uniform:
            distribution: uniform
            low: -1
            high: 1
        x:
            shared_with: shared_uniform
        y:
            shared_with: shared_uniform
        r:
            distribution: normal
            mean: 10
            std: 2

    In this example:
    - `x` and `y` share the same values, generated using a uniform distribution.
    - `r` is generated independently using a normal distribution with specified mean and standard deviation.
    """

    def __init__(self, config: ConfigDict | str):
        """
        Initialize the generator with the given parameters.

        Parameters
        ----------
        config : ConfigDict | str
            Path to a YAML configuration file.
            or
            Dictionary containing the number of samples and parameter definitions.
            If config is a string, it is assumed to be a YAML configuration file path.
        """
        if isinstance(config, str):
            with open(config, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config: ConfigDict = config

    @property
    @abstractmethod
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        """
        Default distributions configurations for the generator.

        Returns
        -------
        dict
            A dictionary containing parameter names and their default distributions.
        """
        pass

    @staticmethod
    def get_distribution(config: Dict[str, str]) -> Callable[[int], np.ndarray[[Any], np.dtype[np.float64]]]:
        """
        Create a distribution function based on parameter configuration.

        Parameters
        ----------
        config : dict
            Dictionary specifying the distribution type and its parameters.

        Returns
        -------
        callable
            A function that generates values based on the specified distribution.

        Raises
        ------
        ValueError
            If the specified distribution is not supported.

        Notes
        -----
        Supported distributions:
        - "normal": Requires `mean` and `std`.
        - "constant": Requires `value`.
        - "poisson": Requires `lambda`.
        - "uniform": Requires `low` and `high`.
        - "exponential": Requires `beta`.
        - Distributions ending in "_constant" will generate identical values
          for all samples using the base distribution.
        """

        match config["distribution"]:
            # Dynamically validate the _const type distributions
            case s if re.search(r"[a-z]*_constant", s):
                config = config.copy()
                config["distribution"] = s.split('_')[0]
                base_distribution = AbstractGenerator.get_distribution(config)
                return lambda size=1: np.full(shape=size, fill_value=base_distribution(1))
            case "normal":
                mean, std = float(config["mean"]), float(config["std"])
                return lambda size=1: np.random.normal(loc=mean, scale=std, size=size)
            case "constant":
                value = float(config["value"])
                return lambda size=1: np.full(shape=size, fill_value=value)
            case "poisson":
                l = float(config["lambda"])
                return lambda size=1: np.random.poisson(lam=l, size=size)
            case "uniform":
                low = float(config["low"])
                high = float(config["high"])
                return lambda size=1: np.random.uniform(low=low, high=high, size=size)
            case "exponential":
                beta = float(config["beta"])
                return lambda size=1: np.random.exponential(scale=beta, size=size)
            case _:
                raise ValueError(f"{config['distribution']} not supported.")

    def save(self):
        """
        Save the generated data or results.

        Notes
        -----
        This method should be implemented in a subclass to define specific
        save functionality (e.g., saving to a file or database).
        """
        pass

    @property
    @abstractmethod
    def structure(self) -> AbstractStructure:
        """
        Return the structure object associated with the generator.

        Returns
        -------
        AbstractStructure
            The structure used for model generation.
        """
        pass

    @property
    @abstractmethod
    def analysis(self) -> AbstractAnalysis:
        """
        Return the analysis object associated with the generator.

        Returns
        -------
        AbstractAnalysis
            The analysis object used for processing models.
        """
        pass

    @abstractmethod
    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        """
        Construct and return the results for a generated parameter set.

        Parameters
        ----------
        params : dict of str to float or int
            The parameter set used for generation.

        Returns
        -------
        dict
            Results based on the parameter set.
        """
        pass

    def __iter__(self):
        """
        Generate an iterator for the dataset.

        Yields
        ------
        dict
            A dictionary containing the results for each generated sample.

        Notes
        -----
        - Parameters from `self.config` override `self.default_config`.
        - Shared parameters specified by `shared_with` will have identical values.
        - For each sample, the structure is generated, analyzed, and results are constructed.
        """

        # Get the actual structural parameters names
        structural_params_keys = set(self.default_config.keys())

        # Number of sample
        n_sample = self.config["n_sample"]

        # Get the parameters descriptions and functions
        config = self.default_config.copy()
        config.update(self.config['parameters'])

        distribution = {param_name: self.get_distribution(distribution) for param_name, distribution in config.items()
                        if "distribution" in distribution}

        for _ in range(n_sample):
            # Generate a value for each distribution
            generated = {k: f(1)[0] for k, f in distribution.items()}

            # Parameters that will be passed to the structure generator
            generation_param: Dict[str, float] = {}

            for param_name in structural_params_keys:
                if 'shared_with' in config[param_name]:
                    value: float | int = generated[config[param_name]['shared_with']]
                else:
                    value: float | int = generated[param_name]

                if 'factor' in config[param_name]:
                    value *= config[param_name]['factor']

                generation_param[param_name] = value

            # Run the calculation
            self.structure.generate_model(generation_param)
            self.analysis.run_analysis()

            r = self.construct_result(generation_param)

            yield r
