import numpy as np
from openseespy import opensees as ops


def get_elem_connectivity():
    return np.array([ops.eleNodes(idx) for idx in ops.getEleTags()], dtype=int)


def get_loads():
    """Get the loads applied to all nodes."""
    n_dof = len(ops.nodeDOFs(0))
    n_nodes = len(ops.getNodeTags())

    idx_nodes = ops.getNodeLoadTags()
    load_data = ops.getNodeLoadData()

    q = np.zeros((n_nodes, n_dof), dtype=np.float64)
    for i, idx in enumerate(idx_nodes):
        i *= n_dof
        q[idx, :] = load_data[i:i + n_dof]

    return q


def get_node_coordinates():
    return np.array([ops.nodeCoord(idx) for idx in ops.getNodeTags()], dtype=np.float64)


def get_node_displacements():
    return np.array([ops.nodeDisp(idx) for idx in ops.getNodeTags()], dtype=np.float64)


def get_deformed_node_coordinates(scale=1):
    return get_node_coordinates() + scale * get_node_displacements()
