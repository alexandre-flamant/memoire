from .abstract_planar_truss import *


class TwoBarsTruss(AbstractPlanarTruss):
    def generate_structure(self, params) -> None:
        length = params["length"]
        height = params["height"]

        ops.node(0, 0., 0.)
        ops.node(1, 0.5 * length, height)
        ops.node(2, length, 0.)

        # Support
        ops.fix(0, 1, 1)
        ops.fix(2, 1, 1)

        ops.uniaxialMaterial("Elastic", 0, params["E_0"])
        ops.uniaxialMaterial("Elastic", 1, params["E_1"])

        # Elements
        ops.element("Truss", 0, 0, 1, params["A_0"], 0)
        ops.element("Truss", 1, 1, 2, params["A_1"], 1)

        # Loads
        # Create TimeSeries
        ops.timeSeries("Linear", 1)
        # Create a plain load pattern
        ops.pattern("Plain", 1, 1)
        ops.load(1, params["P_x_1"], params["P_y_1"])
