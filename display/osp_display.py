from .utils import *
import numpy as np
from openseespy import opensees as ops
from matplotlib import pyplot as plt


def display_structure(initial=True, deformed=True, def_scale=50):  # ax):
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

    fig, ax = _create_figure(nodes)

    if initial:
        nodes = get_node_coordinates()
        ax.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=20, zorder=1)
        for s, e in elem_connectivity:
            ax.plot([nodes[s][0], nodes[e][0]], [nodes[s][1], nodes[e][1]], c='black', zorder=0)

        loads = get_loads()
        loads /= np.max(loads)
        loads *= 0.1*d

        for i, q in enumerate(loads):
            if q[0] == 0 and q[1] == 0:
                continue
            ax.arrow(
                    nodes[i][0], nodes[i][1], q[0], q[1], width=0.01, length_includes_head=True, zorder=2, color='red',
                    head_starts_at_zero=False, head_width=0.1
                    )

    if deformed:
        nodes = get_deformed_node_coordinates(def_scale)
        for s, e in elem_connectivity:
            ax.plot([nodes[s][0], nodes[e][0]], [nodes[s][1], nodes[e][1]], c='magenta', zorder=4, linewidth=0.5)


def _create_figure(nodes):
    min_x, min_y, max_x, max_y = _get_bounding_box(nodes)
    dx = max_x - min_x
    dy = max_y - min_y
    d = max(dx, dy)
    min_x -= 0.15*d
    min_y -= 0.15*d
    max_x += 0.15*d
    max_y += 0.15*d

    # Create the plot
    fig, ax = plt.subplots()

    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    ax.invert_yaxis()

    # Return the figure and axis for further customization if needed
    return fig, ax


def _get_bounding_box(nodes):
    min_x = min(nodes[:, 0])
    min_y = min(nodes[:, 1])
    max_x = max(nodes[:, 0])
    max_y = max(nodes[:, 1])
    return min_x, min_y, max_x, max_y
