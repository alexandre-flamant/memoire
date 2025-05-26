from .abstract_planar_truss import *


class StoriesTruss(AbstractPlanarTruss):
    """
    Multi-story, multi-span planar truss structure generator.

    This class generates a building-like structure composed of vertical, horizontal,
    and diagonal truss elements. The truss spans horizontally across bays (spans)
    and vertically across floors (stories). Supports are added at the base nodes.

    Parameters
    ----------
    params : dict
        Dictionary of geometry, material, and loading specifications. Required keys:

        - "n_stories" : int
            Number of stories (vertical levels).
        - "n_spans" : int
            Number of horizontal spans.
        - "height" : float
            Height of each story.
        - "width" : float
            Width of each span.
        - "E_{i}" : float
            Young’s modulus for bar i.
        - "A_{i}" : float
            Cross-sectional area for bar i.
        - "P_x_{i}" : float
            Horizontal point load at node i.
        - "P_y_{i}" : float
            Vertical point load at node i.
    """

    def generate_structure(self, params: Dict[str, int | float]) -> None:
        """
        Generates the full multi-story planar truss model.

        This includes:
        - Node layout for all floors and bays.
        - Vertical, horizontal, and diagonal truss elements.
        - Fixed supports at ground level nodes.
        - Application of point loads at each node.

        Parameters
        ----------
        params : dict
            Model parameters (see class docstring).
        """
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

        # Supports (bottom nodes)
        for i in range(n_spans + 1):
            idx = int(i * (n_stories + 1))
            ops.fix(idx, 1, 1)

        idx_bar = 0

        # Vertical elements
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

        # Diagonal elements (X-shaped bracing)
        for i in range(n_spans):
            for j in range(n_stories):
                # Lower-left to upper-right
                young = params[f"E_{idx_bar}"]
                area = params[f"A_{idx_bar}"]
                ops.uniaxialMaterial("Elastic", idx_bar, young)

                idx_start = int(i * (n_stories + 1) + j)
                idx_end = int(idx_start + (n_stories + 1) + 1)

                ops.element("Truss", idx_bar, idx_start, idx_end, area, idx_bar)
                idx_bar += 1

                # Upper-left to lower-right
                young = params[f"E_{idx_bar}"]
                area = params[f"A_{idx_bar}"]
                ops.uniaxialMaterial("Elastic", idx_bar, young)

                idx_start = int(i * (n_stories + 1) + j + 1)
                idx_end = int(idx_start + (n_stories + 1) - 1)

                ops.element("Truss", idx_bar, idx_start, idx_end, area, idx_bar)
                idx_bar += 1

        # Loads
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        self.generate_load(params, n_nodes)

    def generate_load(self, params: dict, n_nodes: int):
        """
        Apply point loads to nodes from parameter dictionary.

        Parameters
        ----------
        params : dict
            Model parameters containing P_x_i and P_y_i values.
        n_nodes : int
            Total number of nodes.
        """
        for i in range(n_nodes):
            p_x = params[f"P_x_{i}"]
            p_y = params[f"P_y_{i}"]
            if p_x != 0 or p_y != 0:
                ops.load(i, p_x, p_y)


class SeismicStoriesTruss(StoriesTruss):
    """
    Subclass of `StoriesTruss` applying a horizontal seismic load to top-level nodes.

    Instead of reading per-node forces from the params dictionary, this model
    applies a distributed seismic load along the top floor.

    Parameters
    ----------
    params : dict
        Must include:
        - "P" : float
            Total horizontal load to be distributed across top nodes.
        - "n_stories" : int
        - "n_spans" : int
    """

    def generate_load(self, params: dict, n_nodes: int):
        """
        Applies a linearly distributed horizontal seismic load to top nodes.

        Parameters
        ----------
        params : dict
            Dictionary containing the seismic load ("P") and structure size.
        n_nodes : int
            Total number of nodes (unused here, inferred from params).
        """
        P = params['P']
        n_stories = params['n_stories']
        n_spans = params['n_spans']

        start = (n_stories + 1) * n_spans
        end = (n_stories + 1) * (n_spans + 1)

        for i, node_id in enumerate(range(start, end)):
            ops.load(node_id, (P / (n_stories + 1)) * i, 0.)
