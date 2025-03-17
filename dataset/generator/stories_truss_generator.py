import numpy as np

from typing import Dict

from dataset.structural.analysis import LinearAnalysis
from dataset.structural.structure import SeismicStoriesTruss
from .abstract_truss_generator import AbstractTrussGenerator
import re


class SeismicTwoStoriesTrussGenerator(AbstractTrussGenerator):

    def __init__(self, config: Dict[str, int|float] | str | None = None):
        super().__init__(config)
        self.__structure = SeismicStoriesTruss()
        self.__analysis = LinearAnalysis()

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        config = {'__area__': {'distribution': 'constant', 'value': 1.e-2},
                  '__young__': {'distribution': 'constant', 'value': 70.e9},
                  'n_stories': {'distribution': 'constant_int', 'value': 2},
                  'n_spans': {'distribution': 'constant_int', 'value': 1},
                  'width': {'distribution': 'constant', 'value': 6},
                  'height': {'distribution': 'constant', 'value': 4},
                  'P': {'distribution': 'constant', 'value': 1000e3}}

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(10)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(10)})

        return config

    @property
    def structure(self) -> SeismicStoriesTruss:
        return self.__structure

    @property
    def analysis(self) -> LinearAnalysis:
        return self.__analysis

    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        keys = params.keys()

        keys_a = sorted([s for s in keys if re.match("A_[0-9]*", s)])
        keys_e = sorted([s for s in keys if re.match("E_[0-9]*", s)])

        bars_elongation = self.structure.initial_elements_length - self.structure.deformed_elements_length

        # Numpy data for HFS5 storage
        r = {
            'truss_width': params['width'],
            'truss_height': params['height'],
            'nodes_coordinate': self.structure.nodes_coordinates.reshape(-1),
            'nodes_displacement': self.structure.nodes_displacements.reshape(-1),
            'load': params['P'],
            'bars_area': np.array([params[k] for k in keys_a]),
            'bars_young': np.array([params[k] for k in keys_e]),
            'bars_force': self.structure.elements_forces.reshape(-1),
            'bars_length_init': self.structure.initial_elements_length,
            'bars_elongation': bars_elongation,
            'bars_strain': bars_elongation / self.structure.initial_elements_length,
            'stiffness_matrix': self.structure.stiffness_matrix.reshape(-1),
        }

        return r
