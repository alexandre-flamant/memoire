from typing import Dict

from structural.analysis import LinearAnalysis
from structural.structure import TenBarsPlanarTruss
from .abstract_generator import AbstractGenerator, ConfigDict

import re
import numpy as np


class TenBarsPlanarTrussGenerator(AbstractGenerator):

    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.__structure = TenBarsPlanarTruss()
        self.__analysis = LinearAnalysis()

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        config = {'__load__': {'distribution': 'constant', 'value': 0.},
                  '__area__': {'distribution': 'constant', 'value': 1.e-3},
                  '__young__': {'distribution': 'constant', 'value': 70.e9},
                  'length': {'distribution': 'constant', 'value': 2},
                  'height': {'distribution': 'constant', 'value': 2}, }

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(10)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(10)})
        config.update({f"P_x_{i}": {'shared_with': '__load__'} for i in range(6)})
        config.update({f"P_y_{i}": {'shared_with': '__load__'} for i in range(6)})

        return config

    @property
    def structure(self) -> TenBarsPlanarTruss:
        return self.__structure

    @property
    def analysis(self) -> LinearAnalysis:
        return self.__analysis

    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        keys = params.keys()

        keys_a = sorted([s for s in keys if re.match("A_[0-9]*", s)])
        keys_e = sorted([s for s in keys if re.match("E_[0-9]*", s)])
        keys_p = sorted([s for s in keys if re.match("P_[x,y]_[0-9]*", s)])
        keys_p = tuple(zip(keys_p[:len(keys_p) // 2], keys_p[len(keys_p) // 2:]))

        # Numpy data for HFS5 storage
        r = {
            'length': params['length'],
            'height': params['height'],
            'youngs': np.array([params[k] for k in keys_e]),
            'areas': np.array([params[k] for k in keys_a]),
            'loads': np.array([[params[k] for k in ks] for ks in keys_p]),
            'nodes': self.structure.nodes_coordinates,
            'nodes_displacements': self.structure.nodes_displacements,
            'forces': self.structure.elements_forces,
            'stiffness': self.structure.stiffness_matrix,
        }

        return r
