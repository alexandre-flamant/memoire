from typing import TypedDict

from .abstract_planar_truss import *
from openseespy import opensees as ops


class ParamDict(TypedDict):
    length: float
    height: float

    A_0: float
    A_1: float
    A_2: float
    A_3: float
    A_4: float
    A_5: float
    A_6: float
    A_7: float
    A_8: float
    A_9: float

    E_0: float
    E_1: float
    E_2: float
    E_3: float
    E_4: float
    E_5: float
    E_6: float
    E_7: float
    E_8: float
    E_9: float

    P_x_0: float
    P_y_0: float
    P_x_1: float
    P_y_1: float
    P_x_2: float
    P_y_2: float
    P_x_3: float
    P_y_3: float
    P_x_4: float
    P_y_4: float
    P_x_5: float
    P_y_5: float


class TenBarsPlanarTruss(AbstractPlanarTruss):
    def generate_structure(self, params: ParamDict) -> None:
        n_nodes = 6
        n_bars = 10

        length = params["length"]
        height = params["height"]

        # Nodes
        ops.node(0, 0.0, -0.5*height)
        ops.node(1, length, -0.5*height)
        ops.node(2, 2.*length, -0.5*height)
        ops.node(3, 0.0, 0.5*height)
        ops.node(4, length, 0.5*height)
        ops.node(5, 2.*length, 0.5*height)

        # Support
        ops.fix(0, 1, 1)
        ops.fix(3, 1, 1)

        # Material
        for i in range(n_bars):
            ops.uniaxialMaterial("Elastic", i, params[f"E_{i}"])

        # Elements
        ops.element("Truss", 0, 0, 1, params["A_0"], 0)
        ops.element("Truss", 1, 1, 2, params["A_1"], 1)
        ops.element("Truss", 2, 3, 4, params["A_2"], 2)
        ops.element("Truss", 3, 4, 5, params["A_3"], 3)
        ops.element("Truss", 4, 1, 4, params["A_4"], 4)
        ops.element("Truss", 5, 2, 5, params["A_5"], 5)
        ops.element("Truss", 6, 0, 4, params["A_6"], 6)
        ops.element("Truss", 7, 3, 1, params["A_7"], 7)
        ops.element("Truss", 8, 1, 5, params["A_8"], 8)
        ops.element("Truss", 9, 4, 2, params["A_9"], 9)

        # Loads
        # Create TimeSeries
        ops.timeSeries("Linear", 1)
        # Create a plain load pattern
        ops.pattern("Plain", 1, 1)

        for i in range(n_nodes):
            p_x = params[f"P_x_{i}"]
            p_y = params[f"P_y_{i}"]
            if p_x != 0 or p_y != 0:
                ops.load(i, p_x, p_y)
