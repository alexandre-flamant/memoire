from .abstract_planar_truss import *


class StoriesTruss(AbstractPlanarTruss):
    def generate_structure(self, params: Dict[str, int | float]) -> None:
        n_stories = params['n_stories']
        n_spans = params['n_spans']
        height = params["height"]
        width = params["width"]

        n_nodes = (n_stories + 1) * (n_spans + 1)

        # Nodes
        for i in range(n_spans + 1):
            for j in range(n_stories + 1):
                idx = int(i * (n_stories + 1) + j)
                x = i * width
                y = j * height
                ops.node(idx, x, y)

        # Support
        for i in range(n_spans + 1):
            idx = int(i * (n_stories + 1))
            ops.fix(idx, 1, 1)

        idx_bar = 0
        # Vertical Elements
        for i in range(n_spans + 1):
            for j in range(n_stories):
                young = params[f"E_{idx_bar}"]
                area = params[f"A_{idx_bar}"]
                ops.uniaxialMaterial("Elastic", idx_bar, young)

                idx_start = int(i * (n_stories + 1) + j)
                idx_end = idx_start + 1

                ops.element("Truss", idx_bar, idx_start, idx_end, area, idx_bar)

                idx_bar += 1

        # Horizontal elements
        for i in range(n_spans):
            for j in range(1, n_stories + 1):
                young = params[f"E_{idx_bar}"]
                area = params[f"A_{idx_bar}"]
                ops.uniaxialMaterial("Elastic", idx_bar, young)

                idx_start = int(i * (n_stories + 1) + j)
                idx_end = int(idx_start + (n_stories + 1))

                ops.element("Truss", idx_bar, idx_start, idx_end, area, idx_bar)

                idx_bar += 1

        # Diagonals elements
        for i in range(n_spans):
            for j in range(n_stories):
                young = params[f"E_{idx_bar}"]
                area = params[f"A_{idx_bar}"]
                ops.uniaxialMaterial("Elastic", idx_bar, young)

                idx_start = int(i * (n_stories + 1) + j)
                idx_end = int(idx_start + (n_stories + 1) + 1)
                ops.element("Truss", idx_bar, idx_start, idx_end, area, idx_bar)
                idx_bar += 1

                young = params[f"E_{idx_bar}"]
                area = params[f"A_{idx_bar}"]
                ops.uniaxialMaterial("Elastic", idx_bar, young)

                idx_start = int(i * (n_stories + 1) + j + 1)
                idx_end = int(idx_start + (n_stories + 1) - 1)
                ops.element("Truss", idx_bar, idx_start, idx_end, area, idx_bar)
                idx_bar += 1

        # Loads
        # Create TimeSeries
        ops.timeSeries("Linear", 1)
        # Create a plain load pattern
        ops.pattern("Plain", 1, 1)

        self.generate_load(params, n_nodes)

    def generate_load(self, params, n_nodes):
        for i in range(n_nodes):
            p_x = params[f"P_x_{i}"]
            p_y = params[f"P_y_{i}"]
            if p_x != 0 or p_y != 0:
                ops.load(i, p_x, p_y)

class SeismicStoriesTruss(StoriesTruss):
    def generate_load(self, params, n_nodes):
        P = params['P']
        for i, j in enumerate(range((params['n_stories'] + 1) * params['n_spans'],
                                    (params['n_stories'] + 1) * (params['n_spans'] + 1))):
            ops.load(j, (P/ (params['n_stories'] + 1)) * i, 0.)