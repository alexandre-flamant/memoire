import numpy as np

from typing import Dict

from dataset.structural.analysis import LinearAnalysis
from dataset.structural.structure import PrattTruss
from dataset.generator.abstract_truss_generator import AbstractTrussGenerator
import re


class PrattTrussGenerator(AbstractTrussGenerator):

    def __init__(self, config: Dict[str, int | float] | str | None = None, analysis=None):
        super().__init__(config, analysis=analysis)
        self._structure = PrattTruss()

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        config = {'__area__': {'distribution': 'constant', 'value': 1.e-2},
                  '__young__': {'distribution': 'constant', 'value': 200.e9},
                  'n_panels': {'distribution': 'constant_int', 'value': 8},
                  'length': {'distribution': 'constant', 'value': 60.0},
                  'height': {'distribution': 'constant', 'value': 7.5},
                  'volumetric_weight': {'distribution': 'constant', 'value': 78.5e3}, }

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(29)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(29)})
        config.update({f"P_x_{i}": {'distribution': 'constant', 'value': 0.} for i in range(16)})
        config.update({f"P_y_{i}": {'distribution': 'constant', 'value': 0.} for i in range(16)})
        return config

    @property
    def structure(self) -> PrattTruss:
        return self._structure

    @property
    def analysis(self):
        return self._analysis

    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        keys = params.keys()

        keys_a = sorted([s for s in keys if re.match("A_[0-9]*", s)], key=lambda s: (s[:2], int(s[2:])))
        keys_e = sorted([s for s in keys if re.match("E_[0-9]*", s)], key=lambda s: (s[:2], int(s[2:])))

        keys_p = sorted([s for s in keys if re.match("P_[x,y]_[0-9]*", s)], key=lambda s: (s[:4], int(s[4:])))
        keys_p = tuple(zip(keys_p[:len(keys_p) // 2], keys_p[len(keys_p) // 2:]))

        bars_elongation = self.structure.initial_elements_length - self.structure.deformed_elements_length

        # Numpy data for HFS5 storage
        r = {
            'length': params['length'],
            'height': params['height'],
            'n_panels': params['n_panels'],
            'volumetric_weight': params['volumetric_weight'],
            'nodes_coordinate': self.structure.nodes_coordinates.reshape(-1),
            'nodes_displacement': self.structure.nodes_displacements.reshape(-1),
            #'nodes_load': np.array([[params[k] for k in ks] for ks in keys_p]).reshape(-1),
            'nodes_load': np.array(self.structure.loads).reshape(-1),
            'bars_area': np.array([params[k] for k in keys_a]),
            'bars_young': np.array([params[k] for k in keys_e]),
            'bars_force': self.structure.elements_forces.reshape(-1),
            'bars_length_init': self.structure.initial_elements_length,
            'bars_elongation': bars_elongation,
            'bars_strain': bars_elongation / self.structure.initial_elements_length,
            'stiffness_matrix': self.structure.stiffness_matrix.reshape(-1),
            'connectivity_matrix': self.structure.elements_connectivity.reshape(-1),
        }

        return r
