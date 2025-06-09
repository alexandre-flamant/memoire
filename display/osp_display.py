from matplotlib import pyplot as plt
from matplotlib.path import Path

from .utils import *

support_markers = {(1,): (Path([(2, 0), (-1, -3 ** .5), (-1, 3 ** .5), (2, 0)], closed=True), 120),
                   (2,): (Path([(0, 2), (-3 ** .5, -1), (3 ** .5, -1), (0, 2)], closed=True), 120),
                   (1, 2): (Path([(1.5, 1.5), (1.5, -1.5), (-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)], closed=True), 80)}


def display_structure(ax=None, initial=True, deformed=True, def_scale=50, load_scale=1.,
                      show_node_ids=True, show_bar_ids=True, show_element_forces=True):
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

        # Compute element forces if requested
        elem_forces = []
        max_force = 0
        if show_element_forces:
            for i in range(len(elem_connectivity)):
                axial_force = ops.eleResponse(i, 'axialForce')[0]
                elem_forces.append(axial_force)
                max_force = max(max_force, abs(axial_force))

        for i, (s, e) in enumerate(elem_connectivity):
            x_coords = [nodes[s][0], nodes[e][0]]
            y_coords = [nodes[s][1], nodes[e][1]]

            # Plot force overlay if requested
            if show_element_forces:
                force = elem_forces[i]
                color = 'black' if force==0 else 'blue' if force > 0 else 'red'
                thickness = 0.5 + 3.0 * abs(force) / (max_force + 1e-6)
                ax.plot(x_coords, y_coords, color=color, linewidth=thickness, alpha=0.6, zorder=3)
            else:
                # Plot base element
                ax.plot(x_coords, y_coords, c='black', zorder=0)

            # Plot bar (element) ID at midpoint
            if show_bar_ids:
                mid_x = (x_coords[0] + x_coords[1]) / 2
                mid_y = (y_coords[0] + y_coords[1]) / 2
                ax.text(mid_x, mid_y, f'{i}', color='black', fontsize=8, ha='center', va='center', zorder=6)

        # Plot node IDs
        if show_node_ids:
            for i, (x, y) in enumerate(nodes):
                ax.text(x + 0.01 * d, y + 0.01 * d, f'{i}', color='blue', fontsize=8, zorder=6)

        # Plot loads
        loads = get_loads()
        if np.max(np.abs(loads)) > 0:
            loads /= np.max(np.abs(loads))
        loads *= 0.1 * d * load_scale

        for i, q in enumerate(loads):
            if q[0] == 0 and q[1] == 0:
                continue
            ax.arrow(
                nodes[i][0], nodes[i][1], q[0], q[1], width=0.01, length_includes_head=True, zorder=2, color='red',
                head_starts_at_zero=False, head_width=0.3
            )

        # Plot supports
        for node in ops.getFixedNodes():
            supp = tuple(ops.getFixedDOFs(node))
            marker, s = support_markers.get(supp, (None, None))
            if marker is not None:
                ax.scatter(*nodes[node], marker=marker, color="green", s=s, zorder=4)

    # Plot deformed shape
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
