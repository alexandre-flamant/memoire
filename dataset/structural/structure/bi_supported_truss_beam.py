from .abstract_planar_truss import *


class BiSupportedTrussBeam(AbstractPlanarTruss):
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
            ops.element("Truss", n_cells + i, idx_start_down + i, idx_start_down + i + 1, params[f"A_{n_cells + i}"],
                        n_cells + i)  # H down

            ops.element("Truss", 2 * n_cells + 1 + i, idx_start_top + 1 + i, idx_start_down + 1 + i,
                        params[f"A_{2 * n_cells + 1 + i}"], 2 * n_cells + 1 + i)  # V

            ops.element("Truss", 3 * n_cells + 1 + (2*i), idx_start_top + i, idx_start_down + 1 + i,
                        params[f"A_{3 * n_cells + 1 + (2*i)}"], 3 * n_cells + 1 + (2*i)) # Diag 1
            ops.element("Truss", 3 * n_cells + 2 + (2*i), idx_start_down + i, idx_start_top + 1 + i,
                        params[f"A_{3 * n_cells + 2 + (2*i)}"], 3 * n_cells + 2 + (2*i))  # Diag 2

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
