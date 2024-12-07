import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import os
import polars as pl
import pprint

from abc import ABC, abstractmethod
from datetime import datetime
from polars import (Int8 as i8,
                    Float32 as f32,
                    Float64 as f64,
                    Boolean,
                    List)


from analysis import LinearAnalysis

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.2e}".rjust(11, ' ') if abs(x) > 1e-4 else "0.".rjust(11,
                                                                                                                  ' ') if x == 0 else "~0.".rjust(
    11, ' ')})


class StructuralDatasetGenerator(ABC):
    """
    DatasetGenerator is the parent class responsible for creating structural datasets
    based on specified parameters and distributions. It serves as a foundation for
    more specialized dataset generators.
    """

    def __init__(self, parameters=None):
        """
        Initializes the DatasetGenerator with provided parameters.

        :param parameters:
            A dictionary containing the configuration for the dataset generator. Defaults to {}.
            The specification of the parameters are the following:

                {
                    'name': str,
                    'parameters': {
                        ...,
                        'parameter_i': {
                            'default': { 'type': str, 'parameters': tuple },
                            ...
                            'parameter_1: { 'type': str, 'parameters': tuple },
                            ...
                        },
                        ...
                    },
                    'distributions': {
                        ...
                        'distribution_i': { 'type': str, 'parameters': tuple },
                        ...
                    }
                }

            The name field contains the name of the structural typology.
            The parameters field contains a collection of dictionaries stored with a key
            named after the parameter name. These dictionaries need to be encoded according
            to the following:

                { 'type': str, 'parameters': tuple }

            The 'type' field is the type of parameter distribution and "parameters' are
            the distribution parameters. Use the following:
                - CONSTANT(value)
                - UNIFORM(low, high)
                - EXPONENTIAL(lambda)
                - NORMAL(mean, standard deviation)
                - DISTRIBUTION(distribution name)

            The parameter component is either 'default' or a  string that starts with a
            digit followed by an optional series of codes. The following examples are valid:
                - 'default'
                - '12'
                - '2-y'
                - '1-x-y'

            The DISTRIBUTION type allows to share the exact same value between parameters.

        :raise
            ValueError: If the parameters are badly encoded.
        """
        super().__init__()
        if not hasattr(self, 'ndof'):
            raise NotImplementedError("Attribute ndof is not defined.")
        if parameters is None:
            parameters = {}
        self._parameters = {
            'name': None,
            'parameters': {},
            'distributions': {}
        }

        self._generators = {}
        self._type_schema = None

        # Check if the name is valid
        if 'name' in parameters:
            if not isinstance(parameters['name'], str):
                raise ValueError("Name parameter must be a string.")
            self._parameters['name'] = parameters['name']

        # Check if distribution follows the correct encoding
        if 'distributions' in parameters:
            for (distribution_name, distribution) in parameters['distributions'].items():
                self._check_distribution(distribution)
                self._parameters['distributions'][distribution_name] = parameters['distributions'][distribution_name]

        if 'parameters' in parameters:
            pass  # The parameters check is handled at the implementation level

    def __str__(self):
        return f"{self._parameters['name']} dataset generator"

    @abstractmethod
    def __iter__(self, size=None):
        pass

    @staticmethod
    def generate_group_parameters(generators, group, group_size, values, code_map=None):
        """
        Compute the generators for a parameter given.

        :param generators:
        :param group:
        :param group_size:
        :param values:
        :param code_map:
        :return:
        """

        # Make the default values for the parameter
        default_generator = generators[group]['default']['generator']

        # Handle the case if the default generator is a DISTRIBUTION
        if isinstance(default_generator, str):
            # Compute the values of the DISTRIBUTION and stores it in a dict for potential further use.
            # Allows to share the same values between parameters.
            if default_generator not in values:
                values[default_generator] = generators[group][default_generator]['generator'](group_size)
            parameters = values[default_generator]
        else:
            parameters = default_generator(group_size)

        # Will generate for each component in the parameter
        for generator_name in generators[group]:
            if generator_name == 'default': continue  # Already taken care of

            generator = generators[group][generator_name]['generator']
            targets = generators[group][generator_name]['targets']  # Components on which the generator acts

            # Compute the values and stores it in a dict for potential further use.
            # Allows to share the same values between parameters.
            if generator_name not in values:
                values[generator_name] = generator(1)[0]
            value = values[generator_name]

            # Some hard recursive update
            for target in targets:
                if target == 'default': continue  # Already taken care of

                # We get the component and the code
                target = target.split('-')
                direction = target[1:]
                target = int(target[0])

                # We get the index of the component dictionary recursively
                # ex: code=[x, y] and code_map = {x:{x:[1], y:[2]}, y:{x:[3], y:[4]}}
                # would give ids = 2
                idx = []
                for d in direction:
                    idx = code_map[d]
                # We fetch the sub-dictionaries recursively and update the key in the last one
                if (len(idx) == 0):
                    parameters[target] = value
                    continue
                parameter = parameters[target]
                for i in idx[:-1]:
                    parameter = parameters[i]
                parameter[idx[-1]] = value

        return parameters

    def get_K(self, ndof=2):
        '''Compute the global structural matrix'''
        # Parameters
        nodes = np.array([ops.nodeCoord(i) for i in ops.getNodeTags()], dtype=int)
        elems = np.array([ops.eleNodes(i) for i in ops.getEleTags()])

        elems_vec = np.array([nodes[e] - nodes[s] for s, e in elems])
        elems_angle = np.array([np.arctan2(*v[::-1]) - np.arctan2(0, 1) for v in elems_vec])

        # Stiffness matrix
        K = np.zeros((ndof * len(ops.getNodeTags()), ndof * len(ops.getNodeTags())))
        for idx in range(len(elems)):
            # Get element stiffness matrix
            s_i, e_i = elems[idx] * ndof
            angle = elems_angle[idx]

            k_loc = self._get_k_loc(idx)
            k_glob = self._get_k_global(k_loc, angle)

            # Assemble global stiffness matrix
            K[s_i: s_i + ndof, s_i: s_i + ndof] += k_glob[0:2, 0:2]
            K[e_i: e_i + ndof, e_i: e_i + ndof] += k_glob[2:4, 2:4]
            K[s_i: s_i + ndof, e_i: e_i + ndof] += k_glob[0:2, 2:4]
            K[e_i: e_i + ndof, s_i: s_i + ndof] += k_glob[2:4, 0:2]

        # Boundary condition
        for idx in range(len(nodes)):
            for i in ops.getFixedDOFs(idx):
                dof = ndof * idx + i - 1  # OSP indices starts at 1

                K[dof, :] = 0.
                K[:, dof] = 0.
                K[dof, dof] = 1.

        return K

    def save(self, dirname=None, size=100):
        if dirname is None:
            dirname = f"./dataset/{self.__class__.__name__}/{datetime.now().strftime("%Y%m%d_%H%M%S")}"

        os.makedirs(dirname, exist_ok=True)

        info_file = os.path.dirname(dirname + '/') + "/info.json"
        csv_file = os.path.dirname(dirname + '/') + "/data.csv"

        # Save model info
        info = {
            "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "model": self.__class__.__name__,
            "size": size,
            "model_arguments": self._parameters
        }
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=4)

        lf = pl.LazyFrame(self.__iter__(size), schema=self._type_schema, infer_schema_length=None, nan_to_null=True)
        lf.sink_csv(csv_file, batch_size=1000)

    @staticmethod
    def _check_distribution(distribution):
        """
        Validates the structural and parameters of a distribution.

        :param distribution (dict):
            A dictionary defining the distribution with keys 'type' and 'parameters'.

                { 'type': str, 'parameters': tuple }

            The 'type' field is the type of parameter distribution and "parameters' are the distribution
            parameters. Use the following:
                - CONSTANT(value)
                - UNIFORM(low, high)
                - EXPONENTIAL(lambda)
                - NORMAL(mean, standard deviation)
                - DISTRIBUTION(distribution name)

        :raise
            ValueError: If the distribution type is unsupported, or if the number of
                        parameters does not match the expected count for the distribution type.
        """
        # Valid distribution name and their number of parameters
        _valid_distributions = {
            'CONSTANT': 1,
            'DISTRIBUTION': 1,
            'EXPONENTIAL': 1,
            'NORMAL': 2,
            'UNIFORM': 2,
            'EXPONENTIAL_CONST': 1,
            'NORMAL_CONST': 2,
            'UNIFORM_CONST': 2,
            'RANDOM_CHOICE': max(1, len(distribution['parameters'])), # At least 1 parameter
        }

        # Check if distribution is correctly encoded
        if 'type' not in distribution or 'parameters' not in distribution:
            raise ValueError(
                f"Invalid distribution setup: {'{' + ", ".join([f"'{key}': ..." for key in distribution.keys()]) + '}'}. " + \
                "Must be {'type': ..., 'parameters': [...]}."
            )

        distribution_type = distribution['type']
        distribution_parameters = distribution['parameters']

        # Check if distribution type is supported
        if distribution_type not in _valid_distributions:
            raise ValueError(
                f"The distribution \"{distribution_type}\" is not supported.\n" + \
                "Choose from:\n" + \
                f"{"  -" + "\n  -".join(_valid_distributions.keys())}"
            )

        # Check if number of parameters is valid
        if len(distribution_parameters) != _valid_distributions[distribution_type]:
            raise ValueError(
                f"Distribution {distribution_type} takes {_valid_distributions[distribution_type]} parameters, " + \
                f"{len(distribution_parameters)} where given.")

    def _check_distributions_group(self, distributions, code_validator: (lambda code: True)):
        """
        Validates a group of distributions within a parameter group.

        :param distributions (dict):
            A dictionary containing the distributions of multiple parameters.
        :param code_validator (function, optional):
            A function to validate the code part of parameter names. It receives a code and return True if
            it is valid.

        Raises:
            ValueError: If parameter names are invalid or distributions are improperly defined.
        """
        for (parameter, distribution) in distributions.items():
            if not isinstance(parameter, str):
                raise ValueError(
                    f"'{parameter}' is not a valid parameter name." + \
                    f"\nNames in distribution groups must be either 'default' or \"$digit$-$code$\"." + \
                    f"Note that the -$code$ part is optionnal."
                )

            # Always valid parameter_name
            if parameter == "default":
                self._check_distribution(distribution)
                continue

            # Validate the parameter name
            parameter_list = parameter.split('-')
            idx = parameter_list[0]
            code = parameter_list[1:]

            if not idx.isdigit() and idx != 'default':
                raise ValueError(
                    f"'{parameter}' is not a valid parameter name. Parameter name must start with an index.")

            if not code_validator(code):
                raise ValueError(f"'{parameter}' is not a valid parameter name. The code does not fit the validator.")

            self._check_distribution(distribution)

    @staticmethod
    def _get_partial_generator(distribution):
        """
        Generate the partial functions that return the provided distribution.
        Each function only need one argument which is 'size'.

        The function for DISTRIBUTION simply returns a string with the name of
        the distribution.

        :param distribution: The distribution of the generator
        :return: The function

        :raise
            ValueError: If the distribution is not supported
        """
        distribution_type = distribution['type']
        distribution_parameters = distribution['parameters']
        match distribution_type:
            case 'CONSTANT':
                value, = distribution_parameters
                return lambda size: np.full(size, value)

            case 'UNIFORM':
                low, high = distribution_parameters
                return lambda size: np.random.uniform(low, high, size)

            case 'NORMAL':
                mean, std = distribution_parameters
                return lambda size: np.random.normal(mean, std, size)

            case 'EXPONENTIAL':
                l, = distribution_parameters
                return lambda size: np.random.exponential(l, size)

            case 'UNIFORM_CONST':
                low, high = distribution_parameters
                return lambda size: np.full(size, np.random.uniform(low, high, 1))

            case 'NORMAL_CONST':
                mean, std = distribution_parameters
                return lambda size: np.full(size, np.random.normal(mean, std, 1))

            case 'EXPONENTIAL_CONST':
                l, = distribution_parameters
                return lambda size: np.full(size, np.random.exponential(l, 1))

            case 'DISTRIBUTION':
                distribution_name, = distribution_parameters
                return distribution_name

            case 'RANDOM_CHOICE':
                values = distribution_parameters
                return lambda size: np.random.choice(values, size=size)

            case _:
                raise ValueError(f"{distribution_type} is not supported.")

    def _parse_group_generators(self, generators, parameter_name):
        """
        Parse the number generators of a given parameter name.

        :param generators: List of the generators already
        :param parameter_name: Parameter to which construct the generators.

        :raise
            ValueError: If the parameters are badly encoded.
        """
        parameters = self._parameters['parameters']

        for idx in parameters[parameter_name]:
            distribution_type = parameters[parameter_name][idx]['type']
            distribution_parameters = parameters[parameter_name][idx]['parameters']

            if distribution_type == 'DISTRIBUTION':
                distribution_parameters = distribution_parameters[0]
                # Has the distribution been defined
                if distribution_parameters not in self._parameters['distributions']:
                    raise ValueError(
                        f"\"{distribution_parameters}\" is not present in the defined distributions." + \
                        f"See:\n" + \
                        f"{"  -" + "\n  -".join(self._parameters['distributions'].keys())}"
                    )

                # The distribution is stored in the parameter related dict.
                # This dictionary contains a 'set' key that store which parameter component it generatess
                if distribution_parameters not in generators[parameter_name]:
                    generators[parameter_name][distribution_parameters] = {
                        'generator': self._get_partial_generator(
                            self._parameters['distributions'][distribution_parameters]),
                        'targets': set()
                    }

                generators[parameter_name][distribution_parameters]['targets'].add(idx)

            else:
                if idx == 'default': continue
                generators[parameter_name][f'{parameter_name}_generator_{idx}'] = {
                    'generator': self._get_partial_generator(parameters[parameter_name][idx]),
                    'targets': set()
                }
                generators[parameter_name][f'{parameter_name}_generator_{idx}']['targets'].add(idx)
        return generators


class PlanarTrussGenerator(StructuralDatasetGenerator):
    def __init__(self, parameters):
        self.ndof = 2
        super().__init__(parameters)

    @abstractmethod
    def __iter__(self, size=None): pass

    def _get_r(self, a):
        '''Compute member rotation matrix'''
        c = np.cos(a)
        s = np.sin(a)
        return np.array([[c, s, 0, 0],
                         [-s, c, 0, 0],
                         [0, 0, c, s],
                         [0, 0, -s, c]])

    def _get_k_loc(self, idx):
        '''Compute the local element matrix'''
        return ops.basicStiffness(idx) * np.array([[1, 0, -1, 0],
                                                   [0, 0, 0, 0],
                                                   [-1, 0, 1, 0],
                                                   [0, 0, 0, 0]])

    def _get_k_global(self, k_loc, angle):
        '''Compute the global element matrix'''
        r = self._get_r(angle)
        return r.T @ k_loc @ r


class LinearCantileverTrussGenerator(PlanarTrussGenerator, LinearAnalysis):
    """
    LinearTrussGenerator is a specialized DatasetGenerator for creating linear truss
    structures with a cross pattern. It defines specific parameters, distributions, and
    methods relevant to truss models, enabling the generation and analysis of multiple
    truss configurations.
    """

    def __init__(self, parameters=None):
        """
        Initializes the LinearTrussGenerator with default or provided parameters.

        :param parameters (dict, optional):
            A dictionary containing the configuration for the truss generator.
            If not provided, default parameters are used.

            Default structural:
            {
                'parameters': {
                    'supports': {
                        '1-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '1-y': {'type': 'CONSTANT', 'parameters': (1,)},
                        '2-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '2-y': {'type': 'CONSTANT', 'parameters': (1,)},
                    },
                    'loads': {
                        '4-y': {'type': 'CONSTANT', 'parameters': (-100e3,)},
                        '6-y': {'type': 'CONSTANT', 'parameters': (-100e3,)},
                    }
                }
            }

        :raises ValueError:
            If any provided parameter name is invalid or distributions are improperly defined.
        """
        super().__init__(parameters)


        # If no parameters are provided, use the default configuration
        if parameters is None:
            parameters = {
                'parameters': {
                    'supports': {
                        '0-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '0-y': {'type': 'CONSTANT', 'parameters': (1,)},
                        '1-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '1-y': {'type': 'CONSTANT', 'parameters': (1,)},
                    },
                    'loads': {
                        '3-y': {'type': 'CONSTANT', 'parameters': (-100.e3,)},
                        '5-y': {'type': 'CONSTANT', 'parameters': (-100.e3,)},
                    }
                }
            }

        # Initialize the parameters with default or provided values
        self._parameters = {
            'name': "Linear cantilever's cross pattern beam" if not self._parameters['name'] else self._parameters[
                'name'],
            'parameters': {  # UNIT
                'cell_number': {'default': {'type': 'CONSTANT', 'parameters': (2,)}},  # Number of cells
                'cell_length': {'default': {'type': 'CONSTANT', 'parameters': (4.,)}},  # Length of each cell (meters)
                'cell_height': {'default': {'type': 'CONSTANT', 'parameters': (4.,)}},  # Height of each cell (meters)
                'supports': {'default': {'type': 'CONSTANT', 'parameters': (0,)}},  # Support conditions
                'youngs': {'default': {'type': 'CONSTANT', 'parameters': (200.e9,)}},  # Young's modulus (N/m²)
                'areas': {'default': {'type': 'CONSTANT', 'parameters': (1.e-2,)}},  # Cross-sectional areas (m²)
                'loads': {'default': {'type': 'CONSTANT', 'parameters': (0.,)}}  # Applied loads (N)
            },
            'distributions': self._parameters['distributions']  # Inherit distributions from the parent class
        }

        # Validate and update parameters if provided

        if 'parameters' in parameters:
            for parameter_name in parameters['parameters'].keys():
                match parameter_name:
                    case 'areas' | 'loads' | 'youngs' | 'supports' | 'cell_number' | 'cell_length' | 'cell_height':
                        # Define specific validators based on parameter type
                        match parameter_name:
                            case 'areas' | 'youngs':
                                validator = lambda code: (len(code) == 0)  # No additional codes expected
                            case 'loads' | 'supports':
                                validator = lambda code: (len(code) == 1) and (
                                        code[0] in ('x', 'y'))  # Single direction
                            case 'cell_number' | 'cell_length' | 'cell_height':
                                validator = None  # No additional validation required

                        # Validate the distribution group for the parameter
                        self._check_distributions_group(parameters['parameters'][parameter_name], validator)
                        # Update the default parameters with user-provided configurations
                        self._parameters['parameters'][parameter_name].update(parameters['parameters'][parameter_name])
                    case _:
                        # Raise an error if an invalid parameter name is provided
                        raise ValueError(
                            f"\"{parameter_name}\" is not a valid parameter name. " +
                            "Choose from:\n" +
                            f"  -" + "\n  -".join(self._parameters['parameters'].keys())
                        )

        if self._parameters['parameters']['cell_number']['default']['type'] not in ['CONSTANT', 'RANDOM_CHOICE']:
            raise ValueError("Only CONSTANT and RANDOM_CHOICE are valid distributions for cell_number")

        # Type schema
        # Hypothesis: n_cells must be int so its either CONSTANT or RANDOM_CHOICE
        max_n_cells = np.max(self._parameters['parameters']['cell_number']['default']['parameters'])

        n_nodes = 2 * (max_n_cells + 1)
        n_elems = 5 * max_n_cells

        self._type_schema = {'n_cells': i8, 'cell_height': f32, 'cell_length': f32}
        self._type_schema.update({f"K_{i}": f64 for i in range((self.ndof * max_n_cells)**2)})
        self._type_schema.update({f"{d}_{i}": f32 for i in range(n_nodes) for d in ('x', 'y')})
        self._type_schema.update({f"fix_{d}_{i}": Boolean for i in range(n_nodes) for d in ('x', 'y')})
        self._type_schema.update({f"P_{d}_{i}": f64 for i in range(n_nodes) for d in ('x', 'y')})
        self._type_schema.update({f"u_{d}_{i}": f32 for i in range(n_nodes) for d in ('x', 'y')})

        self._type_schema.update({f"E_{i}": f64 for i in range(n_elems)})
        self._type_schema.update({f"A_{i}": f32 for i in range(n_elems)})
        self._type_schema.update({f"N_{i}": f32 for i in range(n_elems)})

        # Initialize generators for each parameter group
        parameters = self._parameters['parameters']
        group_distributions = self._parameters['distributions']
        self._generators = {
            param: {
                'default': {
                    'generator': self._get_partial_generator(parameters[param]['default']),
                    'targets': set()
                }
            }
            for param in parameters.keys()
        }

        # Parse and assign additional generators based on distributions
        for group in parameters.keys():
            self._generators = self._parse_group_generators(self._generators, group)

    def generate_parameters(self, generators):
        """
        Generates a complete set of parameters for the truss structural using the defined generators.

        :param generators (dict):
            Dictionary containing generator functions for each parameter group.

        :return (dict):
            A dictionary containing all generated parameters necessary to define the truss structural.
            Structure:
            {
                'cell_length': float,
                'cell_height': float,
                'cell_number': int,
                'materials': {
                    1: {'E': float},
                    2: {'E': float},
                    ...
                },
                'bars_materials': {
                    1: int,
                    2: int,
                    ...
                },
                'bars_areas': {
                    1: float,
                    2: float,
                    ...
                },
                'nodes_loads': {
                    1: {'x': float, 'y': float},
                    2: {'x': float, 'y': float},
                    ...
                },
                'supports': {
                    1: {'x': int, 'y': int},
                    2: {'x': int, 'y': int},
                    ...
                }
            }
        """
        # Dictionary to store generated values
        values = {}

        # Generate basic structural parameters
        cell_number = int(self.generate_group_parameters(generators, 'cell_number', 1, values)[0])
        cell_length = float(self.generate_group_parameters(generators, 'cell_length', 1, values)[0])
        cell_height = float(self.generate_group_parameters(generators, 'cell_height', 1, values)[0])

        self._max_n_cells = cell_number # Used to update the DB schema

        # Generate material and load parameters based on the number of cells
        areas = self.generate_group_parameters(generators, 'areas', 5 * cell_number, values, {'': []})
        youngs = self.generate_group_parameters(generators, 'youngs', 5 * cell_number, values, {'': []})
        loads = self.generate_group_parameters(generators, 'loads', (2 + 2 * cell_number, 2), values,
                                               {'x': [0], 'y': [1]})
        supports = self.generate_group_parameters(generators, 'supports', (2 + 2 * cell_number, 2), values,
                                                  {'x': [0], 'y': [1]})

        # Assemble all generated parameters into a structured dictionary
        return {
            'cell_length': cell_length,
            'cell_height': cell_height,
            'cell_number': cell_number,
            'materials': {i: {'E': youngs[i]} for i in range(5 * cell_number)},  # Material properties for each element
            'bars_materials': {i: i for i in range(5 * cell_number)},  # Mapping of bars to materials
            'bars_areas': {i: areas[i] for i in range(5 * cell_number)},  # Cross-sectional areas for each bar
            'nodes_loads': {i: {'x': loads[i][0], 'y': loads[i][1]} for i in range(2 + 2 * cell_number)},
            # Loads applied to nodes
            'supports': {i: {'x': supports[i][0], 'y': supports[i][1]} for i in range(2 + 2 * cell_number)},
            # Support conditions for nodes
        }

    @staticmethod
    def initialize_truss(parameters):
        """
        Initializes and defines the truss model in OpenSees based on the provided parameters.

        :param parameters (dict):
            A dictionary containing all necessary parameters to define the truss structural.
            Structure:
            {
                'cell_length': float,
                'cell_height': float,
                'cell_number': int,
                'materials': {
                    1: {'E': float},
                    2: {'E': float},
                    ...
                },
                'bars_materials': {
                    1: int,
                    2: int,
                    ...
                },
                'bars_areas': {
                    1: float,
                    2: float,
                    ...
                },
                'nodes_loads': {
                    1: {'x': float, 'y': float},
                    2: {'x': float, 'y': float},
                    ...
                },
                'supports': {
                    1: {'x': int, 'y': int},
                    2: {'x': int, 'y': int},
                    ...
                }
            }

        :note:
            The model is automatically cleared before defining a new one.
        """
        # Clear any existing model in OpenSees
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 2)

        ## Define materials in the model
        for tag, material in parameters['materials'].items():
            ops.uniaxialMaterial('Elastic', tag, material['E'])

        ## Geometry: Define nodes and elements based on cell configuration
        cell_number = parameters['cell_number']
        L = float(parameters['cell_length'])  # Cell length
        H = float(parameters['cell_height'])  # Cell height

        ### Create nodes for each cell
        for i in range(cell_number + 1):
            ops.node(2 * i, i * L, 0.5 * H)  # Upper node of the cell
            ops.node(2 * i + 1, i * L, -0.5 * H)  # Lower node of the cell

        ### Create truss elements (bars) between nodes
        bars_areas = parameters['bars_areas']
        bars_materials = parameters['bars_materials']

        #### Top horizontal bars
        for i in range(cell_number):
            tag = i
            area = bars_areas[tag]
            material = bars_materials[tag]
            ops.element("Truss", tag, *(2 * i, 2 + 2 * i), area, material)

        #### Bottom horizontal bars
        for i in range(cell_number):
            tag = cell_number + i
            area = bars_areas[tag]
            material = bars_materials[tag]
            ops.element("Truss", tag, *(1 + 2 * i, 3 + 2 * i), area, material)

        #### Vertical bars
        for i in range(cell_number):
            tag = 2 * cell_number + i
            area = bars_areas[tag]
            material = bars_materials[tag]
            ops.element("Truss", tag, *(2 * i + 2, 2 * i + 3), area, material)

        #### S-O to N-E diagonal bars
        for i in range(cell_number):
            tag = 3 * cell_number + 2 * i
            area = bars_areas[tag]
            material = bars_materials[tag]
            ops.element("Truss", tag, *(2 * i, 2 * i + 3), area, material)

        #### S-E to N-O diagonal bars
        for i in range(cell_number):
            tag = 3 * cell_number + 2 * i + 1
            area = bars_areas[tag]
            material = bars_materials[tag]
            ops.element("Truss", tag, *(2 * i + 1, 2 * (i + 1)), area, material)

        ## Define boundary conditions (supports)
        for (idx, conditions) in parameters['supports'].items():
            ops.fix(idx, bool(conditions['x']), bool(conditions['y']))

        ## Define loads
        ops.timeSeries('Constant', 1)  # Define a constant time series for loading
        ops.pattern("Plain", 1, 1)  # Define a plain load pattern
        for (idx, load) in parameters['nodes_loads'].items():
            ops.load(idx, float(load['x']), float(load['y']))  # Apply loads to nodes

    def __iter__(self, max_count=-1):
        """
        Creates an iterator that generates and analyzes truss structures using OpenSees.

        :param max_count (int, optional):
            The maximum number of structures to generate and analyze.
            If set to -1, the iterator is infinite. Defaults to -1.

        :yield (list):
            A list of basic force responses from each element in the truss structural.
        """
        # Initialize counter
        i = 0
        while i != max_count:
            i += 1
            # Generate a set of parameters for the truss structural
            parameters = self.generate_parameters(self._generators)
            # Initialize the truss model in OpenSees with the generated parameters
            self.initialize_truss(parameters)

            self.run_analysis()

            n_cell = parameters['cell_number']

            row = {'n_cells': parameters['cell_number'],
                   'cell_height': parameters['cell_height'],
                   'cell_length': parameters['cell_length']}
            row.update({f"x_{i}": ops.nodeCoord(i)[0] for i in range(2 * n_cell + 2)})

            row.update({f"y_{i}": ops.nodeCoord(i)[1] for i in range(2 * n_cell + 2)})
            row.update({f"fix_{d}_{i}": parameters['supports'][i][d]
                        for i in range(2 * n_cell + 2)
                        for d in ['x', 'y']})
            row.update({f"P_{d}_{i}": parameters['nodes_loads'][i][d]
                        for i in range(2 * n_cell + 2)
                        for d in ['x', 'y']})
            row.update({f"E_{i}": parameters['materials'][i]['E'] for i in range(5*n_cell)})
            row.update({f"A_{i}": parameters['bars_areas'][i] for i in range(5*n_cell)})
            row.update({f"u_x_{i}": ops.nodeDisp(i)[0] for i in range(2 * n_cell + 2)})
            row.update({f"u_y_{i}": ops.nodeDisp(i)[1] for i in range(2 * n_cell + 2)})
            row.update({f"N_{i}": ops.eleResponse(i, "basicForce")[0] for i in range(5 * n_cell)})
            row.update({f"K_{i}": K_i for i, K_i in enumerate(self.get_K(self.ndof).flatten())})

            # row =  [parameters['cell_number'], parameters['cell_height'], parameters['cell_length']]
            # row += [ops.nodeCoord(i)[d] for i in range(2 * n_cell + 2) for d in [0, 1]]  # Nodes locations
            # row += [parameters['supports'][i][d] for i in range(2 * n_cell + 2) for d in ['x', 'y']]  # Support
            # row += [parameters['nodes_loads'][i][d] for i in range(2 * n_cell + 2) for d in
            #         ['x', 'y']]  # Loads on nodes
            # row += [parameters['materials'][i]['E'] for i in range(5 * n_cell)]  # Young modulus
            # row += [parameters['bars_areas'][i] for i in range(5 * n_cell)]  # Areas
            # row += [u_i for tag in ops.getNodeTags() for u_i in ops.nodeDisp(tag)]  # Node displacement
            # row += [n_i for tag in ops.getEleTags() for n_i in ops.eleResponse(tag, "basicForce")]  # Elements forces

            yield row


class LinearTwoBarTruss(PlanarTrussGenerator, LinearAnalysis):
    """
    LinearTrussGenerator is a specialized DatasetGenerator for creating linear truss
    structures with a cross pattern. It defines specific parameters, distributions, and
    methods relevant to truss models, enabling the generation and analysis of multiple
    truss configurations.
    """

    def __init__(self, parameters=None):
        """
        Initializes the LinearTrussGenerator with default or provided parameters.

        :param parameters (dict, optional):
            A dictionary containing the configuration for the truss generator.
            If not provided, default parameters are used.

            Default structural:
            {
                'parameters': {
                    'supports': {
                        '1-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '1-y': {'type': 'CONSTANT', 'parameters': (1,)},
                        '2-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '2-y': {'type': 'CONSTANT', 'parameters': (1,)},
                    },
                    'loads': {
                        '4-y': {'type': 'CONSTANT', 'parameters': (-100e3,)},
                        '6-y': {'type': 'CONSTANT', 'parameters': (-100e3,)},
                    }
                }
            }

        :raises ValueError:
            If any provided parameter name is invalid or distributions are improperly defined.
        """
        super().__init__(parameters)

        # If no parameters are provided, use the default configuration
        if parameters is None:
            parameters = {
                'parameters': {
                    'supports': {
                        '0-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '0-y': {'type': 'CONSTANT', 'parameters': (1,)},
                        '1-x': {'type': 'CONSTANT', 'parameters': (1,)},
                        '1-y': {'type': 'CONSTANT', 'parameters': (1,)},
                    },
                    'loads': {
                        '2-y': {'type': 'CONSTANT', 'parameters': (-100.e3,)},
                    }
                }
            }

        # Type schema
        # Generalize to fit the actual parametric data !
        self._type_schema = {'height': f32, 'length': f32}
        self._type_schema.update({f'x_{i}': f32 for i in range(3)})
        self._type_schema.update({f'y_{i}': f32 for i in range(3)})
        self._type_schema.update({f'fix_x_{i}': Boolean for i in range(3)})
        self._type_schema.update({f'fix_y_{i}': Boolean for i in range(3)})
        self._type_schema.update({f'P_x_{i}': f64 for i in range(3)})
        self._type_schema.update({f'P_y_{i}': f64 for i in range(3)})
        self._type_schema.update({f'u_x_{i}': f32 for i in range(3)})
        self._type_schema.update({f'u_y_{i}': f32 for i in range(3)})
        self._type_schema.update({f'E_{i}': f64 for i in range(2)})
        self._type_schema.update({f'A_{i}': f32 for i in range(2)})
        self._type_schema.update({f'N_{i}': f64 for i in range(2)})
        self._type_schema.update({f'K_{i}': f64 for i in range((3*self.ndof)**2)})

        # Initialize the parameters with default or provided values
        self._parameters = {
            'name': "Linear cantilever's cross pattern beam" if not self._parameters['name'] else self._parameters[
                'name'],
            'parameters': {  # UNIT
                'length': {'default': {'type': 'CONSTANT', 'parameters': (4.,)}},  # Length of each cell (meters)
                'height': {'default': {'type': 'CONSTANT', 'parameters': (4.,)}},  # Height of each cell (meters)
                'supports': {'default': {'type': 'CONSTANT', 'parameters': (0,)}},  # Support conditions
                'youngs': {'default': {'type': 'CONSTANT', 'parameters': (200.e9,)}},  # Young's modulus (N/m²)
                'areas': {'default': {'type': 'CONSTANT', 'parameters': (1.e-2,)}},  # Cross-sectional areas (m²)
                'loads': {'default': {'type': 'CONSTANT', 'parameters': (0.,)}}  # Applied loads (N)
            },
            'distributions': self._parameters['distributions']  # Inherit distributions from the parent class
        }

        # Validate and update parameters if provided
        if 'parameters' in parameters:
            for parameter_name in parameters['parameters'].keys():
                match parameter_name:
                    case 'areas' | 'loads' | 'youngs' | 'supports' | 'length' | 'height':
                        # Define specific validators based on parameter type
                        match parameter_name:
                            case 'areas' | 'youngs':
                                validator = lambda code: (len(code) == 0)  # No additional codes expected
                            case 'loads' | 'supports':
                                validator = lambda code: (len(code) == 1) and (
                                        code[0] in ('x', 'y'))  # Single direction
                            case 'length' | 'height':
                                validator = None  # No additional validation required

                        # Validate the distribution group for the parameter
                        self._check_distributions_group(parameters['parameters'][parameter_name], validator)
                        # Update the default parameters with user-provided configurations
                        self._parameters['parameters'][parameter_name].update(parameters['parameters'][parameter_name])
                    case _:
                        # Raise an error if an invalid parameter name is provided
                        raise ValueError(
                            f"\"{parameter_name}\" is not a valid parameter name. " +
                            "Choose from:\n" +
                            f"  -" + "\n  -".join(self._parameters['parameters'].keys())
                        )

        # Initialize generators for each parameter group
        parameters = self._parameters['parameters']
        group_distributions = self._parameters['distributions']

        self._generators = {
            param: {
                'default': {
                    'generator': self._get_partial_generator(parameters[param]['default']),
                    'targets': set()
                }
            }
            for param in parameters.keys()
        }

        # Parse and assign additional generators based on distributions
        for group in parameters.keys():
            self._generators = self._parse_group_generators(self._generators, group)

    def generate_parameters(self, generators):
        """
        Generates a complete set of parameters for the truss structural using the defined generators.

        :param generators (dict):
            Dictionary containing generator functions for each parameter group.

        :return (dict):
            A dictionary containing all generated parameters necessary to define the truss structural.
            Structure:
            {
                'cell_length': float,
                'cell_height': float,
                'cell_number': int,
                'materials': {
                    1: {'E': float},
                    2: {'E': float},
                    ...
                },
                'bars_materials': {
                    1: int,
                    2: int,
                    ...
                },
                'bars_areas': {
                    1: float,
                    2: float,
                    ...
                },
                'nodes_loads': {
                    1: {'x': float, 'y': float},
                    2: {'x': float, 'y': float},
                    ...
                },
                'supports': {
                    1: {'x': int, 'y': int},
                    2: {'x': int, 'y': int},
                    ...
                }
            }
        """
        # Dictionary to store generated values
        values = {}

        # Generate basic structural parameters
        length = float(self.generate_group_parameters(generators, 'length', 1, values)[0])
        height = float(self.generate_group_parameters(generators, 'height', 1, values)[0])

        # Generate material and load parameters based on the number of cells
        areas = self.generate_group_parameters(generators, 'areas', 2, values, {'': []})
        youngs = self.generate_group_parameters(generators, 'youngs', 2, values, {'': []})
        loads = self.generate_group_parameters(generators, 'loads', (3, 2), values,
                                               {'x': [0], 'y': [1]})
        supports = self.generate_group_parameters(generators, 'supports', (3, 2), values,
                                                  {'x': [0], 'y': [1]})

        # Assemble all generated parameters into a structured dictionary
        return {
            'length': length,
            'height': height,
            'materials': {i: {'E': youngs[i]} for i in range(2)},  # Material properties for each element
            'bars_materials': {i: i for i in range(2)},  # Mapping of bars to materials
            'bars_areas': {i: areas[i] for i in range(2)},  # Cross-sectional areas for each bar
            'nodes_loads': {i: {'x': loads[i][0], 'y': loads[i][1]} for i in range(3)},
            # Loads applied to nodes
            'supports': {i: {'x': supports[i][0], 'y': supports[i][1]} for i in range(3)},
            # Support conditions for nodes
        }

    @staticmethod
    def initialize_truss(parameters):
        """
        Initializes and defines the truss model in OpenSees based on the provided parameters.

        :param parameters (dict):
            A dictionary containing all necessary parameters to define the truss structural.
            Structure:
            {
                'length': float,
                'height': float,
                'materials': {
                    1: {'E': float},
                    2: {'E': float}
                },
                'bars_materials': {
                    1: int,
                    2: int
                },
                'bars_areas': {
                    1: float,
                    2: float
                },
                'nodes_loads': {
                    1: {'x': float, 'y': float},
                    2: {'x': float, 'y': float},
                    3: {'x': float, 'y': float}
                },
                'supports': {
                    1: {'x': int, 'y': int},
                    2: {'x': int, 'y': int},
                    3: {'x': int, 'y': int}
                }
            }

        :note:
            The model is automatically cleared before defining a new one.
        """
        # Clear any existing model in OpenSees
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 2)

        ## Define materials in the model
        for tag, material in parameters['materials'].items():
            ops.uniaxialMaterial('Elastic', tag, material['E'])

        ## Geometry: Define nodes and elements based on cell configuration
        L = float(parameters['length'])  # Cell length
        H = float(parameters['height'])  # Cell height

        ### Create nodes
        ops.node(0, 0, H)
        ops.node(1, 0, 0)
        ops.node(2, L, H)

        ### Create truss elements (bars) between nodes
        bars_areas = parameters['bars_areas']
        bars_materials = parameters['bars_materials']

        # Bars
        ops.element('Truss', 0, *(0, 2), bars_areas[0], bars_materials[0])
        ops.element('Truss', 1, *(1, 2), bars_areas[1], bars_materials[1])

        ## Define boundary conditions (supports)
        ops.fix(0, True, True)
        ops.fix(1, True, True)

        ## Define loads
        ops.timeSeries('Constant', 1)  # Define a constant time series for loading
        ops.pattern("Plain", 1, 1)  # Define a plain load pattern

        for (idx, load) in parameters['nodes_loads'].items():
            ops.load(idx, float(load['x']), float(load['y']))  # Apply loads to nodes

    def __iter__(self, max_count=-1):
        """
        Creates an iterator that generates and analyzes truss structures using OpenSees.

        :param max_count (int, optional):
            The maximum number of structures to generate and analyze.
            If set to -1, the iterator is infinite. Defaults to -1.

        :yield (list):
            A list of basic force responses from each element in the truss structural.
        """
        # Initialize counter
        i = 0
        while i != max_count:
            i += 1
            # Generate a set of parameters for the truss structural
            parameters = self.generate_parameters(self._generators)
            # Initialize the truss model in OpenSees with the generated parameters
            self.initialize_truss(parameters)

            self.run_analysis()

            row = {'height': parameters['height'], 'length': parameters['length']}
            row.update({f'x_{i}': ops.nodeCoord(i)[0] for i in range(3)})
            row.update({f'y_{i}': ops.nodeCoord(i)[1] for i in range(3)})
            row.update({f'fix_x_{i}': parameters['supports'][i]['x'] for i in range(3)})
            row.update({f'fix_y_{i}': parameters['supports'][i]['y'] for i in range(3)})
            row.update({f'P_x_{i}': parameters['nodes_loads'][i]['x'] for i in range(3)})
            row.update({f'P_y_{i}': parameters['nodes_loads'][i]['y'] for i in range(3)})
            row.update({f'E_{i}': parameters['materials'][i]['E'] for i in range(2)})
            row.update({f'A_{i}': parameters['bars_areas'][i] for i in range(2)})
            row.update({f'u_x_{i}': ops.nodeDisp(i)[0] for i in range(3)})
            row.update({f'u_y_{i}': ops.nodeDisp(i)[1] for i in range(3)})
            row.update({f'N_{i}': ops.basicForce(i)[0] for i in range(2)})
            row.update({f'K_{i}': K_i for i, K_i in enumerate(self.get_K(self.ndof).flatten())})

            yield row


__all__ = [
    "StructuralDatasetGenerator",
    "PlanarTrussGenerator",
    "LinearCantileverTrussGenerator",
    "LinearTwoBarTruss"
]
