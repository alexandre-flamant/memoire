from .abstract_planar_truss import *


class TwoBarsTruss(AbstractPlanarTruss):
    """
    Simple 2D truss structure with two bars forming a triangle.

    This class generates a minimal planar truss composed of two bars forming
    a triangular geometry. Supports are fixed at the base nodes, and a load
    is applied at the top (middle) node. Ideal for simple static analysis
    or training/testing small surrogate models.

    Parameters
    ----------
    params : dict
        Dictionary containing geometry, material, and loading parameters. Required keys:

        - "length" : float
            Horizontal span between the base nodes.
        - "height" : float
            Vertical height from the base to the central top node.
        - "E_0" : float
            Young’s modulus of the left bar.
        - "E_1" : float
            Young’s modulus of the right bar.
        - "A_0" : float
            Cross-sectional area of the left bar.
        - "A_1" : float
            Cross-sectional area of the right bar.
        - "P_x_1" : float
            Horizontal load applied at the central node (node 1).
        - "P_y_1" : float
            Vertical load applied at the central node (node 1).
    """

    def generate_structure(self, params) -> None:
        """
        Build the two-bar truss model in OpenSees.

        This includes node creation, bar definition, support conditions,
        and application of a point load at the central node.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.
        """
        length = params["length"]
        height = params["height"]

        # Nodes: [0] left support, [1] top, [2] right support
        ops.node(0, 0., 0.)
        ops.node(1, 0.5 * length, height)
        ops.node(2, length, 0.)

        # Support at both ends
        ops.fix(0, 1, 1)
        ops.fix(2, 1, 1)

        # Material assignment
        ops.uniaxialMaterial("Elastic", 0, params["E_0"])
        ops.uniaxialMaterial("Elastic", 1, params["E_1"])

        # Elements
        ops.element("Truss", 0, 0, 1, params["A_0"], 0)  # Left bar
        ops.element("Truss", 1, 1, 2, params["A_1"], 1)  # Right bar

        # Load at central node
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(1, params["P_x_1"], params["P_y_1"])
