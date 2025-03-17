from dataset.structural.structure.abstract_planar_truss import *


class PrattTruss(AbstractPlanarTruss):
    def generate_structure(self, params: Dict[str, int | float]) -> None:
        self.params = params.copy()
        length = float(params['length'])
        height = float(params["height"])
        n_panels = int(params["n_panels"])
        volumetric_weight = float(params["volumetric_weight"])
        panel_width = length / n_panels
        if n_panels % 2: raise Exception("n_panels must be pair")

        n_nodes = 2 * n_panels

        # Nodes
        for idx in range(n_panels + 1):
            x = idx * panel_width
            ops.node(idx, x, 0.)
        for idx in range(n_panels + 1, 2 * n_panels):
            x = length - panel_width * (idx - n_panels)
            ops.node(idx, x, height)

        idx_bar = 0
        # Horizontal Elements
        for idx in range(n_panels):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx
            idx_end = idx_start + 1

            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        for idx in range(n_panels + 1, 2 * n_panels - 1):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx
            idx_end = idx_start + 1

            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        # Vertical Elements
        for idx in range(1, n_panels):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx
            idx_end = 2 * n_panels - idx_start

            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        # Diagonal Elements

        area, young = self._get_bar_characteristics(idx_bar)
        ops.uniaxialMaterial('Elastic', idx_bar, young)
        ops.element('Truss', idx_bar, 0, 2 * n_panels - 1, area, idx_bar)
        self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
        idx_bar += 1

        for idx in range(1, n_panels // 2):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx + 1
            idx_end = 2 * n_panels + 1 - idx_start

            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        for idx in range(n_panels // 2, n_panels - 1):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx
            idx_end = 2 * n_panels - 1 - idx_start

            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        area, young = self._get_bar_characteristics(idx_bar)
        ops.uniaxialMaterial('Elastic', idx_bar, young)
        ops.element('Truss', idx_bar, n_panels, n_panels + 1, area, idx_bar)
        self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
        idx_bar += 1

        # Support
        ops.fix(0, 1, 1)
        ops.fix(n_panels, 0, 1)
        self.params[f"P_x_0"] = 0.
        self.params[f"P_y_0"] = 0.
        self.params[f"P_y_{n_panels}"] = 0.

        # Loads
        # Create TimeSeries
        ops.timeSeries("Linear", 1)
        # Create a plain load pattern
        ops.pattern("Plain", 1, 1)

        for i in range(n_nodes):
            p_x = self.params[f"P_x_{i}"]
            p_y = self.params[f"P_y_{i}"]
            if p_x != 0 or p_y != 0:
                ops.load(i, p_x, p_y)

    def _get_bar_characteristics(self, idx):
        young = self.params[f"E_{idx}"]
        area = self.params[f"A_{idx}"]

        return area, young

    def _set_bar_load(self, idx_start, idx_end, area, density):
        c_start = self.nodes_coordinates[idx_start, :]
        c_end = self.nodes_coordinates[idx_end, :]
        length = np.linalg.norm(c_start - c_end)
        volume = length * area
        q = volume * density

        self.params[f"P_y_{idx_start}"] -= .5 * q
        self.params[f"P_y_{idx_end}"] -= .5 * q
