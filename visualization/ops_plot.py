import numpy as np

from matplotlib import pyplot as plt
from openseespy import opensees as ops


def plot(ax=None,
         support=True,
         node_label=True, elem_label=True, dimension=True,
         line_color='black', line_style='--', line_width=1,
         deformed=True, deformed_line_color='red', deformed_line_style='-', deformed_line_width=1,
         scale_factor=50):
    # Create axis if not given
    if ax is None:
        _, ax = plt.subplots()
        ax.set_aspect('equal', 'box')

    # Get undisplaced nodes coordinates
    node_coords = {node_tag: ops.nodeCoord(node_tag) for node_tag in ops.getNodeTags()}

    # Get displaced node coordinates
    disp_coords = {node_tag:
        [
            ops.nodeCoord(node_tag)[0] + ops.nodeDisp(node_tag, 1) * scale_factor,
            ops.nodeCoord(node_tag)[1] + ops.nodeDisp(node_tag, 2) * scale_factor
        ]
        for node_tag in ops.getNodeTags()
    }

    # Get connectivity matrix
    connectivity = {elem_tag: ops.eleNodes(elem_tag) for elem_tag in ops.getEleTags()}

    # Plot underformed shape
    for tag, nodes in connectivity.items():
        x = [node_coords[node][0] for node in nodes]
        y = [node_coords[node][1] for node in nodes]
        ax.plot(x, y, linestyle=line_style, color=line_color, linewidth=line_width)

        if elem_label:
            annotate_element(ax, x, y, tag, color=line_color)

    # Plot nodes
    r = np.max([*ax.get_xlim(), *ax.get_ylim()]) * 0.005
    for tag, node_coord in node_coords.items():
        if node_label:
            annotate_node(ax, node_coord, tag, offset=r, color=line_color)
        plot_node(ax, node_coord, r, color=line_color)

    # Plot deformed shape
    if deformed:
        for tag, nodes in connectivity.items():
            x = [disp_coords[node][0] for node in nodes]
            y = [disp_coords[node][1] for node in nodes]
            ax.plot(x, y, linestyle=deformed_line_style, color=deformed_line_color, linewidth=deformed_line_width)

        # Plot nodes
        for node_coord in disp_coords.values():
            plot_node(ax, node_coord, r, color=deformed_line_color)

    return ax


def plot_node(ax, coordinates, r, color='black'):
    node = plt.Circle(coordinates, radius=r, color=color)
    ax.add_patch(node)


def annotate_node(ax, coordinates, tag, offset=0., color='black'):
    coordinates = [xi + offset for xi in coordinates]
    ax.annotate(tag, coordinates, color=color, fontweight="semibold")


def annotate_element(ax, x, y, tag, offset=0., color='black'):
    coordinates = [x[1] + .3 * (x[0] - x[1]), y[1] + .3 * (y[0] - y[1])]
    coordinates = [xi + offset for xi in coordinates]
    ax.annotate(tag, coordinates, color=color)
