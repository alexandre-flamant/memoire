from matplotlib import pyplot as plt
from matplotlib.path import Path

from .utils import *

support_markers = {(1,): (Path([(2, 0), (-1, -3 ** .5), (-1, 3 ** .5), (2, 0)], closed=True), 120),
                   (2,): (Path([(0, 2), (-3 ** .5, -1), (3 ** .5, -1), (0, 2)], closed=True), 120),
                   (1, 2): (Path([(1.5, 1.5), (1.5, -1.5), (-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)], closed=True), 80)}


def display_structure(ax=None, initial=True, deformed=True, def_scale=50, load_scale=1.):
    nodes = None
    if initial:
        nodes = get_node_coordinates()
    if deformed:
        if nodes is not None:
            nodes = np.vstack((nodes, get_deformed_node_coordinates(def_scale)))
        else:
            nodes = get_deformed_node_coordinates(def_scale)

    elem_connectivity = get_elem_connectivity()

    min_x, min_y, max_x, max_y = _get_bounding_box(nodes)
    dx = max_x - min_x
    dy = max_y - min_y
    d = max(dx, dy)

    if ax is None:
        _, ax = _create_figure(nodes)

    if initial:
        nodes = get_node_coordinates()
        ax.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=20, zorder=1)
        for s, e in elem_connectivity:
            ax.plot([nodes[s][0], nodes[e][0]], [nodes[s][1], nodes[e][1]], c='black', zorder=0)

        loads = get_loads()
        loads /= np.max(np.abs(loads))
        loads *= 0.1 * d * load_scale

        for i, q in enumerate(loads):
            if q[0] == 0 and q[1] == 0:
                continue
            ax.arrow(
                nodes[i][0], nodes[i][1], q[0], q[1], width=0.01, length_includes_head=True, zorder=2, color='red',
                head_starts_at_zero=False, head_width=0.3
            )

        for node in ops.getFixedNodes():
            supp = tuple(ops.getFixedDOFs(node))
            marker, s = support_markers[supp]
            ax.scatter(*nodes[node], marker=marker, color="green", s=s, zorder=4)

    if deformed:
        nodes = get_deformed_node_coordinates(def_scale)
        for s, e in elem_connectivity:
            ax.plot([nodes[s][0], nodes[e][0]], [nodes[s][1], nodes[e][1]], c='magenta', zorder=5, linewidth=0.5)


def _create_figure(nodes):
    min_x, min_y, max_x, max_y = _get_bounding_box(nodes)
    dx = max_x - min_x
    dy = max_y - min_y
    d = max(dx, dy)
    min_x -= 0.15 * d
    min_y -= 0.15 * d
    max_x += 0.15 * d
    max_y += 0.15 * d

    # Create the plot
    fig, ax = plt.subplots()

    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Return the figure and axis for further customization if needed
    return fig, ax


def _get_bounding_box(nodes):
    min_x = min(nodes[:, 0])
    min_y = min(nodes[:, 1])
    max_x = max(nodes[:, 0])
    max_y = max(nodes[:, 1])
    return min_x, min_y, max_x, max_y
