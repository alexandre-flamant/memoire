from typing import Dict

from structural.analysis import LinearAnalysis
from structural.structure import TenBarsPlanarTruss
from .abstract_generator import AbstractGenerator, ConfigDict


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
        r = params.copy()
        coord = self.structure.nodes_coordinates
        k = self.structure.stiffness_matrix.reshape(-1)
        u = self.structure.nodes_displacements
        n = self.structure.elements_forces

        r.update({f'x_{i}': coord[i, 0] for i in range(6)})
        r.update({f'y_{i}': coord[i, 1] for i in range(6)})
        r.update({f'u_x_{i}': u[i, 0] for i in range(6)})
        r.update({f'u_y_{i}': u[i, 1] for i in range(6)})
        r.update({f"N_{i}": n[i] for i in range(10)})
        r.update({f"K_{i:03d}": k[i] for i in range(len(k))})

        return r
