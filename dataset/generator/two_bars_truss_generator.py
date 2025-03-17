from typing import Dict

from dataset.structural.analysis import LinearAnalysis
from dataset.structural.structure import TwoBarsTruss
from .abstract_generator import ConfigDict
from .abstract_truss_generator import AbstractTrussGenerator


class TwoBarsTrussGenerator(AbstractTrussGenerator):

    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.__structure = TwoBarsTruss()
        self.__analysis = LinearAnalysis()

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        config = {'__load__': {'distribution': 'constant', 'value': 0.},
                  '__area__': {'distribution': 'constant', 'value': 1.e-3},
                  '__young__': {'distribution': 'constant', 'value': 70.e9},
                  'length': {'distribution': 'constant', 'value': 2},
                  'height': {'distribution': 'constant', 'value': 2},}

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(2)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(2)})
        config.update({f"P_x_{i}": {'shared_with': '__load__'} for i in range(3)})
        config.update({f"P_y_{i}": {'shared_with': '__load__'} for i in range(3)})

        return config

    @property
    def structure(self) -> TwoBarsTruss:
        return self.__structure

    @property
    def analysis(self) -> LinearAnalysis:
        return self.__analysis
