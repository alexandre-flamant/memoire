from typing import TypedDict

import torch

from .abstract_planar_truss import *


class TenBarsCantileverTruss(AbstractPlanarTruss):
    """
    Planar cantilever truss structure with 10 bars and 6 nodes.

    This model defines a symmetric cantilever truss configuration with fixed
    supports at the left end and various bar connections including vertical,
    horizontal, and diagonal members. Loads can be applied at any node via
    the parameter dictionary.

    The structure is commonly used for benchmark problems in structural optimization
    and stiffness inference tasks.

    Parameters
    ----------
    params : dict
        Dictionary containing geometric, material, and loading specifications.
        Required keys:

        - "length" : float
            Horizontal spacing between adjacent vertical columns.
        - "height" : float
            Vertical separation between the top and bottom nodes.
        - "E_{i}" : float
            Young’s modulus for bar i (0–9).
        - "A_{i}" : float
            Cross-sectional area for bar i (0–9).
        - "P_x_{i}" : float
            Horizontal point load applied at node i (0–5).
        - "P_y_{i}" : float
            Vertical point load applied at node i (0–5).
    """

    def generate_structure(self, params) -> None:
        """
        Generate the full cantilever truss model in OpenSees.

        This includes creation of nodes, assignment of fixed supports,
        material definitions, element connectivity, and application of point loads.

        Parameters
        ----------
        params : dict
            Dictionary of all geometry, material, and load parameters.
        """
        n_nodes = 6
        n_bars = 10

        length = params["length"]
        height = params["height"]

        # Nodes (top row: 0,1,2 ; bottom row: 3,4,5)
        ops.node(0, 0.0, 0.5 * height)
        ops.node(1, length, 0.5 * height)
        ops.node(2, 2.0 * length, 0.5 * height)
        ops.node(3, 0.0, -0.5 * height)
        ops.node(4, length, -0.5 * height)
        ops.node(5, 2.0 * length, -0.5 * height)

        # Supports (left-end top and bottom nodes)
        ops.fix(0, 1, 1)
        ops.fix(3, 1, 1)

        # Material definitions
        for i in range(n_bars):
            ops.uniaxialMaterial("Elastic", i, params[f"E_{i}"])

        # Truss elements (horizontal, vertical, diagonal)
        ops.element("Truss", 0, 0, 1, params["A_0"], 0)  # Top horizontal
        ops.element("Truss", 1, 1, 2, params["A_1"], 1)
        ops.element("Truss", 2, 3, 4, params["A_2"], 2)  # Bottom horizontal
        ops.element("Truss", 3, 4, 5, params["A_3"], 3)
        ops.element("Truss", 4, 1, 4, params["A_4"], 4)  # Vertical
        ops.element("Truss", 5, 2, 5, params["A_5"], 5)
        ops.element("Truss", 6, 0, 4, params["A_6"], 6)  # Diagonals
        ops.element("Truss", 7, 3, 1, params["A_7"], 7)
        ops.element("Truss", 8, 1, 5, params["A_8"], 8)
        ops.element("Truss", 9, 4, 2, params["A_9"], 9)

        # Loads
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        for i in range(n_nodes):
            p_x = params[f"P_x_{i}"]
            p_y = params[f"P_y_{i}"]
            if p_x != 0 or p_y != 0:
                ops.load(i, p_x, p_y)
