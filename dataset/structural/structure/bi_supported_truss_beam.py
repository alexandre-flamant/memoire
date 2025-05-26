from .abstract_planar_truss import *


class BiSupportedTrussBeam(AbstractPlanarTruss):
    """
    Planar truss beam model with bi-supported boundary conditions.

    This structure consists of a regular arrangement of vertical, horizontal,
    and diagonal truss elements forming a 2D beam-like structure. Supports are fixed
    at two bottom nodes, and loads can be applied at any node via the `params` dictionary.

    The geometry is parameterized by the number of cells, with each cell contributing
    a fixed set of bars. Each bar can have an individual cross-sectional area and Young’s modulus.

    Parameters
    ----------
    params : dict
        Dictionary containing geometric, material, and loading parameters. Required keys:

        - "length": float
            Total horizontal span of the truss.
        - "height": float
            Total vertical height of the truss.
        - "E_{i}": float
            Young's modulus for the i-th bar (indexed from 0 to n_bars - 1).
        - "A_{i}": float
            Cross-sectional area for the i-th bar (indexed from 0 to n_bars - 1).
        - "P_x_{i}": float
            Horizontal load applied at node i.
        - "P_y_{i}": float
            Vertical load applied at node i.

    Notes
    -----
    - The structure is built with 4 cells, resulting in:
        - 10 nodes (5 top, 5 bottom)
        - 21 elements (horizontal, vertical, and diagonal members)
    - Nodes are laid out in a regular grid, and element IDs are assigned deterministically.
    - Loads are applied only if the provided values are non-zero.
    """

    def generate_structure(self, params) -> None:
        n_cells = 4
        n_nodes = (n_cells + 1) * 2
        n_bars = 5 * n_cells + 1

        idx_start_top = 0
        idx_start_down = n_nodes // 2

        length = params["length"]
        height = params["height"]

        l_i = length / n_cells

        # Nodes
        for i in range(n_cells + 1):
            ops.node(i, i * l_i, height)
            ops.node(n_cells + 1 + i, i * l_i, 0.)

        # Support
        ops.fix(5, 1, 1)
        ops.fix(9, 1, 1)

        # Material
        for i in range(n_bars):
            ops.uniaxialMaterial("Elastic", i, params[f"E_{i}"])

        # Elements
        ops.element("Truss", 2 * n_cells, idx_start_top, idx_start_down, params[f"A_{2 * n_cells}"], 2 * n_cells)
        for i in range(n_cells):
            ops.element("Truss", i, i, i + 1, params[f"A_{i}"], i)  # H top
            ops.element("Truss", n_cells + i, idx_start_down + i, idx_start_down + i + 1,
                        params[f"A_{n_cells + i}"], n_cells + i)  # H down

            ops.element("Truss", 2 * n_cells + 1 + i, idx_start_top + 1 + i, idx_start_down + 1 + i,
                        params[f"A_{2 * n_cells + 1 + i}"], 2 * n_cells + 1 + i)  # V

            ops.element("Truss", 3 * n_cells + 1 + (2 * i), idx_start_top + i, idx_start_down + 1 + i,
                        params[f"A_{3 * n_cells + 1 + (2 * i)}"], 3 * n_cells + 1 + (2 * i))  # Diag 1
            ops.element("Truss", 3 * n_cells + 2 + (2 * i), idx_start_down + i, idx_start_top + 1 + i,
                        params[f"A_{3 * n_cells + 2 + (2 * i)}"], 3 * n_cells + 2 + (2 * i))  # Diag 2

        # Loads
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        for i in range(n_nodes):
            p_x = params[f"P_x_{i}"]
            p_y = params[f"P_y_{i}"]
            if p_x != 0 or p_y != 0:
                ops.load(i, p_x, p_y)
