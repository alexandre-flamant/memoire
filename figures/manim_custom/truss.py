from manim import *
from .supports import *
import numpy as np


class Truss(VMobject):
    """
    A Manim-compatible class for visualizing 2D truss structures.

    This class enables the creation of animated truss diagrams, complete with nodes, members, supports,
    applied loads, and optional displacements and internal force visualizations.

    Parameters
    ----------
    nodes : list[tuple[float, float, float]]
        Coordinates of the truss nodes in 2D (z = 0 recommended for rendering).
    connectivity_matrix : list[tuple[int, int]]
        List of index pairs defining the member connections between nodes.
    supports : dict[int, tuple[bool, bool]]
        Dictionary mapping node index to support conditions. Each value is a tuple (fix_x, fix_y).
    loads : dict[int, tuple[float, float]]
        Dictionary mapping node index to load vector (Fx, Fy).
    displacements : np.ndarray, optional
        Nodal displacements as an array of shape (n_nodes, 2). Defaults to zero if not provided.

    Style Parameters
    ----------------
    node_style : dict
        Appearance of nodes (e.g., {'color': YELLOW, 'radius': 0.1}).
    member_style : dict
        Appearance of members. Must include 'line_class' (e.g., Line, DashedLine), and optionally color and stroke_width.
    support_style : dict
        Appearance of supports (e.g., {'color': BLUE, 'stroke_width': 4, 'height': 1.5}).
    load_style : dict
        Appearance of load arrows (e.g., {'color': RED, 'stroke_width': 6, 'scale': 2}).
    deformed_style : dict
        Appearance of deformed members. Must include 'line_class'.
    deformed_node_style : dict
        Appearance of displaced node markers.

    Label Parameters
    ----------------
    display_node_labels : bool
        Whether to display node indices.
    node_labels : list[str], optional
        Custom labels for nodes. Defaults to automatic indexing.
    node_label_style : dict
        Style of node labels (e.g., font size, color, prefix, suffix).
    node_label_offsets : list[Manim Vector]
        List of offset vectors for node labels.

    display_member_labels : bool
        Whether to display member indices.
    member_labels : list[str], optional
        Custom labels for members.
    member_label_style : dict
        Style of member labels.
    member_label_offsets : list[Manim Vector]
        List of offset vectors for member labels.

    display_load_labels : bool
        Whether to display labels for loads.
    load_labels : list[str], optional
        Custom labels for loads.
    load_label_style : dict
        Style of load labels (can be TeX or Text).
    load_label_offsets : list[Manim Vector]
        List of offset vectors for load labels.

    Methods
    -------
    toggle_normal_forces(forces, compression_color=RED, traction_color=BLUE, max_width=25)
        Toggles between geometry and force-colored member visualization.
    deform_structure(scale=1, animate=False)
        Applies nodal displacements to deform the structure.
    overlap_deformation(scale=1, animate=False)
        Overlays the deformed shape on top of the undeformed structure.

    Example
    -------
    >>> from manim import Scene
    >>> import numpy as np
    >>> nodes = [(0, 0, 0), (3, 0, 0), (6, 0, 0), (3, 2, 0)]
    >>> connectivity = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    >>> supports = {0: (True, True), 2: (False, True)}
    >>> loads = {1: (0, -500)}
    >>> displacements = np.zeros((4, 2))  # Replace with FEA output

    >>> truss = Truss(
    >>>     nodes=nodes,
    >>>     connectivity_matrix=connectivity,
    >>>     supports=supports,
    >>>     loads=loads,
    >>>     displacements=displacements,
    >>>     node_style={'color': YELLOW, 'radius': 0.12},
    >>>     member_style={'line_class': Line, 'color': DARK_BLUE, 'stroke_width': 4},
    >>>     support_style={'color': GREY, 'height': 1, 'stroke_width': 3},
    >>>     load_style={'color': RED, 'scale': 1.5, 'stroke_width': 6},
    >>>     deformed_style={'line_class': DashedLine, 'color': ORANGE, 'stroke_width': 3},
    >>>     display_node_labels=True,
    >>>     node_label_style={'color': BLACK, 'prefix': 'N', 'font_size': 24},
    >>>     member_label_style={'color': DARK_BLUE, 'prefix': 'M', 'font_size': 22}
    >>> )
    >>> scene.add(truss)
    >>> scene.wait()
    >>> scene.play(*truss.deform_structure(scale=100, animate=True))
    """

    def __init__(
            self,
            nodes,
            connectivity_matrix,
            supports,
            loads,
            displacements=None,
            node_style=None,
            member_style=None,
            support_style=None,
            display_loads=True,
            load_style=None,
            tip_style=None,
            display_load_labels=False,
            load_labels=None,
            load_label_style=None,
            load_label_offsets=None,
            deformed_style=None,
            deformed_node_style=None,
            display_node_labels=False,
            node_labels=None,
            node_label_style=None,
            node_label_offsets=None,
            display_member_labels=False,
            member_labels=None,
            member_label_style=None,
            member_label_offsets=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self._nodes = nodes
        self._connectivity = connectivity_matrix
        self._supports = supports
        self._loads = loads
        self._displacements = np.zeros_like(nodes) if displacements is None else displacements
        self.display_loads = display_loads

        # Default styles
        self._node_style = {"color": BLACK, "radius": 0.1}
        self._node_style.update(node_style or {})
        self._display_node_label = display_node_labels
        self._node_labels = node_labels
        self._node_label_style = {'text_class': Text, 'color': self._node_style['color'], 'font_size': 30,
                                  'prefix': '', 'suffix': ''}
        self._node_label_style.update(node_label_style or {})
        self._node_label_class = self._node_label_style.pop("text_class")
        self._node_label_prefix = self._node_label_style.pop("prefix")
        self._node_label_suffix = self._node_label_style.pop("suffix")
        self._nodes_label_offsets = (node_label_offsets
                                     if node_label_offsets
                                     else [.2 * UP for _ in self._nodes])

        self._member_style = {"line_class": Line, "stroke_width": DEFAULT_STROKE_WIDTH, "color": BLACK}
        self._member_style.update(member_style or {})
        self._member_line_class = self._member_style.pop("line_class")
        self._display_member_label = display_member_labels
        self._member_labels = member_labels
        self._member_label_style = {'text_class': Text, 'color': self._member_style['color'], 'font_size': 30,
                                    'prefix': '(', 'suffix': ')'}
        self._member_label_style.update(member_label_style or {})
        self._member_label_class = self._member_label_style.pop("text_class")
        self._member_label_prefix = self._member_label_style.pop("prefix")
        self._member_label_suffix = self._member_label_style.pop("suffix")
        self._member_label_offsets = (member_label_offsets
                                      if member_label_offsets
                                      else [.2 * UR for _ in self._connectivity])

        self._support_style = {"height": .5, "stroke_width": DEFAULT_STROKE_WIDTH, "color": BLACK}
        self._support_style.update(support_style or {})

        self._load_style = {"color": RED, "stroke_width": DEFAULT_STROKE_WIDTH, "scale": 1.5}
        self._load_style.update(load_style or {})
        self._tip_style = {'tip_length': .5, 'tip_width': .4}
        self._tip_style.update(tip_style or {})
        self._load_scale = self._load_style.pop('scale') / max([np.linalg.norm(f) for f in loads.values()], default=1)
        self._display_load_label = display_load_labels
        self._loads_labels = load_labels

        self._load_label_style = {'text_class': Tex, 'color': self._load_style['color'], 'font_size': 35,
                                  'prefix': '\\textbf{$F_{', 'suffix': '}$}'}
        self._load_label_style.update(load_label_style or {})
        self._load_label_class = self._load_label_style.pop("text_class")
        self._load_label_prefix = self._load_label_style.pop("prefix")
        self._load_label_suffix = self._load_label_style.pop("suffix")
        self._load_label_offsets = (load_label_offsets
                                    if load_label_offsets
                                    else [.4 * RIGHT for _ in range(len(self._loads))])

        self._deformed_style = {"line_class": DashedLine, "stroke_width": DEFAULT_STROKE_WIDTH, "color": PURPLE}
        self._deformed_style.update(deformed_style or {})
        self._deformed_line_class = self._deformed_style.pop("line_class")

        self._deformed_node_style = {"color": PURPLE, "radius": 0.05}
        self._deformed_node_style.update(deformed_node_style or {})

        # Visual groups
        self.nodes = VGroup()
        self.members = VGroup()
        self.supports = VGroup()
        self.loads = VGroup()
        self.displaced_nodes = VGroup()
        self.displaced_members = VGroup()
        self.nodes_labels = VGroup()
        self.members_labels = VGroup()
        self.loads_labels = VGroup()

        self.__add_nodes(self._nodes, self.nodes)
        self.__add_members(self.nodes, self.members)
        self.__add_supports(self.nodes, self.supports)
        if self.display_loads:
            self.__add_loads(self.nodes, self.loads)
        self.__force_toggled = False

        if self._display_node_label:
            self.__add_labels(self._node_label_class, self._node_labels, self.nodes, self._nodes_label_offsets,
                              self.nodes_labels, self._node_label_style,
                              prefix=self._node_label_prefix, suffix=self._node_label_suffix)
        if self._display_member_label:
            self.__add_labels(self._member_label_class, self._member_labels, self.members, self._member_label_offsets,
                              self.members_labels, self._member_label_style,
                              prefix=self._member_label_prefix, suffix=self._member_label_suffix)
        if self._display_load_label:
            self.__add_labels(self._load_label_class, self._loads_labels, self.loads, self._load_label_offsets,
                              self.loads_labels, self._load_label_style,
                              prefix=self._load_label_prefix, suffix=self._load_label_suffix)

        self.add(self.members, self.supports, self.nodes, self.loads, self.nodes_labels, self.members_labels,
                 self.loads_labels)

    def __add_nodes(self, coords, group):
        for pos in coords:
            group.add(Dot(pos, **self._node_style))

    def __add_members(self, nodes, group):
        for i, j in self._connectivity:
            start, end = nodes[i], nodes[j]
            line = self._member_line_class(start=start, end=end, **self._member_style)
            line.add_updater(lambda l, s=start, e=end: l.put_start_and_end_on(s.get_center(), e.get_center()))
            group.add(line)

    def __add_supports(self, nodes, group):
        for idx, (fix_x, fix_y) in self._supports.items():
            node = nodes[idx]
            pos = node.get_center()

            support_kwargs = self._support_style.copy()

            if fix_x and fix_y:
                support = SimpleSupport(pos, **support_kwargs)
            elif fix_x:
                support = RollingSupport(pos, direction=RIGHT, **support_kwargs)
            elif fix_y:
                support = RollingSupport(pos, direction=UP, **support_kwargs)

            support.add_updater(lambda s, n=node: s.shift(n.get_center() - s.get_ref_point()))
            group.add(support)

    def __add_loads(self, nodes, group):
        for idx, (fx, fy) in self._loads.items():
            node = nodes[idx]
            vect = self._load_scale * (fx * RIGHT + fy * UP)
            arrow = (Line(start=node.get_center(), end=node.get_center() + vect, buff=0, **self._load_style)
                     .add_tip(**self._tip_style))
            arrow.add_updater(lambda a, n=node: a.shift(n.get_center() - a.get_start()))
            group.add(arrow)

    def __add_labels(self, text_class, labels, elements, offsets, group, style, prefix='', suffix=''):
        labels = ([text_class(f"{prefix}{i+1}{suffix}", **style) for i in range(len(elements))] if not labels
                  else [text_class(f"{prefix}{label}{suffix}", **style) for label in labels])

        for label, element, offset in zip(labels, elements, offsets):
            label.move_to(element.get_center() + offset)
            label.add_updater(lambda l, e=element, o=offset:
                              l.move_to(e.get_center() + o))
            group.add(label)

    def toggle_normal_forces(self, forces=None, compression_color=RED, traction_color=BLUE_D, max_width=25):
        """
        Toggle between displaying standard member appearance and visualizing internal axial forces.

        When activated, each member's color and thickness is adjusted based on the provided internal force.
        Compression forces are typically shown in one color (e.g., red), tension forces in another (e.g., blue),
        and member thickness is scaled proportionally to the force magnitude.

        Parameters
        ----------
        forces : list[float], optional
            List of axial forces (same order as `connectivity_matrix`). Positive = tension, Negative = compression.
            Required only on the first toggle. On subsequent calls, it restores the default styling.
        compression_color : Manim color, default=RED
            Color used to represent members in compression.
        traction_color : Manim color, default=BLUE
            Color used to represent members in tension.
        max_width : float, default=25
            Maximum stroke width for the most forceful member. Others are scaled proportionally.

        Raises
        ------
        ValueError
            If `forces` is None when enabling force visualization for the first time.

        Notes
        -----
        Internally, this method toggles a flag, so repeated calls alternate between
        standard geometry and force-based styling.
        """

        if self.__force_toggled:
            for line in self.members:
                line.set_stroke(color=self._member_style['color'],
                                width=self._member_style['stroke_width'])
        else:
            if forces is None:
                raise ValueError("forces is None, you need to provide values.")
            max_force = np.max(np.abs(forces))
            for line, force in zip(self.members, forces):
                line.set_stroke(
                    color=compression_color if force < 0
                    else traction_color if force > 0
                    else self._member_style['color'],
                    width=max(max_width * np.abs(force) / max_force, DEFAULT_STROKE_WIDTH/2)
                )

        self.__force_toggled = not self.__force_toggled

    def deform_structure(self, scale=1, animate=False):
        """
        Apply nodal displacements to deform the structure.

        This method updates the position of each node using the displacement vector stored in `self._displacements`,
        optionally animating the deformation. It modifies the actual geometry of the truss.

        Parameters
        ----------
        scale : float, default=1
            Scaling factor applied to the displacements to enhance visibility.
        animate : bool, default=False
            If True, returns a list of Manim animations to be used with `self.play(...)`.
            If False, applies the deformation instantly.

        Returns
        -------
        list[Animation] or None
            List of animations if `animate=True`, otherwise None.

        Example
        -------
        >>> animations = truss.deform_structure(scale=100, animate=True)
        >>> self.play(*animations)
        """
        anim_group = []
        for i, (ux, uy) in enumerate(self._displacements):
            original = self._nodes[i]
            shift = self.nodes[i].get_center() - original + scale * (ux * RIGHT + uy * UP)
            if animate:
                anim_group.append(self.nodes[i].animate.move_to(original + shift))
            else:
                self.nodes[i].move_to(original + shift)
        return anim_group if animate else None

    def overlap_deformation(self, scale=1, animate=False, u=None):
        """
        Overlay a transparent deformed truss over the current undeformed structure.

        This is useful for visually comparing the original and displaced shapes simultaneously.
        Deformed elements are rendered with distinct style and do not alter the original nodes.

        Parameters
        ----------
        scale : float, default=1
            Scale factor for the displacements.
        animate : bool, default=False
            If True, returns a list of Manim animations for use in a scene.
            If False, immediately shifts the deformed elements into place.

        Returns
        -------
        list[Animation] or None
            List of animations if `animate=True`, otherwise None.

        Notes
        -----
        The deformed structure is stored in `self.displaced_nodes` and `self.displaced_members`.
        It is added on top of the existing scene and updated dynamically.
        """
        original_positions = [n.get_center() for n in self.nodes]
        nodes = VGroup()
        members = VGroup()

        for pos in original_positions:
            nodes.add(Dot(pos, **self._deformed_node_style))

        for i, j in self._connectivity:
            start, end = nodes[i], nodes[j]
            line = self._deformed_line_class(start=start, end=end, **self._deformed_style)
            line.add_updater(lambda l, s=start, e=end: l.put_start_and_end_on(s.get_center(), e.get_center()))
            members.add(line)

        animations = []
        if u is None : u = self._displacements
        for node, disp in zip(nodes, u):
            vec = scale * np.array([disp[0], disp[1], 0.0])
            if animate:
                animations.append(node.animate.shift(vec))
            else:
                node.shift(vec)

        self.displaced_nodes = nodes
        self.displaced_members = members
        self.add(members, nodes)
        self.update()

        return animations if animate else (members, nodes)


__all__ = ['Truss']
