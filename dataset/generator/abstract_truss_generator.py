import re
from typing import Dict

import numpy as np

from .abstract_generator import AbstractGenerator


class AbstractTrussGenerator(AbstractGenerator):
    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        keys = params.keys()

        keys_a = sorted([s for s in keys if re.match("A_[0-9]*", s)])
        keys_e = sorted([s for s in keys if re.match("E_[0-9]*", s)])
        keys_p = sorted([s for s in keys if re.match("P_[x,y]_[0-9]*", s)])
        keys_p = tuple(zip(keys_p[:len(keys_p) // 2], keys_p[len(keys_p) // 2:]))

        bars_elongation = self.structure.initial_elements_length - self.structure.deformed_elements_length

        # Numpy data for HFS5 storage
        r = {
            'truss_length': params['length'],
            'truss_height': params['height'],
            'nodes_coordinate': self.structure.nodes_coordinates.reshape(-1),
            'nodes_displacement': self.structure.nodes_displacements.reshape(-1),
            'nodes_load': np.array([[params[k] for k in ks] for ks in keys_p]).reshape(-1),
            'bars_area': np.array([params[k] for k in keys_a]),
            'bars_young': np.array([params[k] for k in keys_e]),
            'bars_force': self.structure.elements_forces.reshape(-1),
            'bars_length_init': self.structure.initial_elements_length,
            'bars_elongation': bars_elongation,
            'bars_strain': bars_elongation / self.structure.initial_elements_length,
            'stiffness_matrix': self.structure.stiffness_matrix.reshape(-1),
        }

        return r
