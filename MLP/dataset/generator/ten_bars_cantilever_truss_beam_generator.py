from typing import Dict

from MLP.structural.analysis import LinearAnalysis
from MLP.structural.structure import TenBarsCantileverTruss
from .abstract_generator import ConfigDict
from .abstract_truss_generator import AbstractTrussGenerator


class TenBarsCantileverTrussGenerator(AbstractTrussGenerator):

    def __init__(self, config: ConfigDict | None = None):
        super().__init__(config)
        self.__structure = TenBarsCantileverTruss()
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
    def structure(self) -> TenBarsCantileverTruss:
        return self.__structure

    @property
    def analysis(self) -> LinearAnalysis:
        return self.__analysis
