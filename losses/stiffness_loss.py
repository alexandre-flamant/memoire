import torch
from torch import FloatTensor, IntTensor
from torch.nn import Module
from torch.nn.functional import mse_loss


class StiffnessToLoadLoss(Module):
    def __init__(self) -> None:
        super(StiffnessToLoadLoss, self).__init__()

    def forward(self, k: torch.Tensor, u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the losses based on the Direct Stiffness Method.

        Parameters
        ----------
        k : torch.Tensor
            A 2D tensor representing the stiffness matrix of shape (n, n) with dtype float32.
        u : torch.Tensor
            A 2D tensor representing the displacement vector of shape (n, 1) with dtype float32.
        q : torch.Tensor
            A 2D tensor representing the force vector of shape (n, 1) with dtype float32.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed Mean Squared Error (MSE) losses between
            the predicted and actual force vectors.

        Notes
        -----
        The losses is computed as the MSE between the predicted force vector (obtained by
        multiplying the stiffness matrix K with the force vector Q) and the true force vector Q.
        """

        # Perform matrix multiplication
        q_pred = torch.matmul(k, u)
        # Calculate and return the MSE losses
        return mse_loss(q_pred, q)


class StiffnessToDisplacementLoss(Module):
    def __init__(self) -> None:
        super(StiffnessToDisplacementLoss, self).__init__()

    def forward(self, k: torch.Tensor, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the losses based on the Direct Stiffness Method.

        Parameters
        ----------
        k : torch.Tensor
            A 2D tensor representing the stiffness matrix of shape (n, n) with dtype float32.
        u : torch.Tensor
            A 2D tensor representing the displacement vector of shape (n, 1) with dtype float32.
        q : torch.Tensor
            A 2D tensor representing the force vector of shape (n, 1) with dtype float32.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed Mean Squared Error (MSE) losses between
            the predicted and actual force vectors.

        Notes
        -----
        The losses is computed as the MSE between the predicted force vector (obtained by
        multiplying the stiffness matrix K with the force vector Q) and the true force vector Q.
        """

        # Perform matrix multiplication

        # u_pred = torch.linalg.solve(k, q) # Not implemented for MPS

        # This is both numerically unstable and super slow
        k_inv = torch.inverse(k)
        u_pred = torch.matmul(k_inv, q)

        # Calculate and return the MSE losses
        return mse_loss(u_pred, u)


def construct_k_from_ea(EA: FloatTensor,
                        nodes: FloatTensor,
                        elems: IntTensor,
                        supports: IntTensor,
                        n_dof: int | None = None,
                        device = None):
    return construct_k_from_params(torch.ones_like(EA, dtype=EA.dtype), EA, nodes, elems, supports, n_dof, device)


def construct_k_from_params(young: FloatTensor,
                            area: FloatTensor,
                            nodes: FloatTensor,
                            elems: IntTensor,
                            supports: IntTensor,
                            n_dof: int | None = None,
                            device = None):
    """
    Perform a batch construction of the global stiffness matrix.

    Parameters
    ----------
    young : torch.FloatTensor
        A tensor of shape (batch_size, n_elems) containing the young modulus for each element in the batch.
    area : torch.FloatTensor
        A tensor of shape (batch_size, n_elems) containing the area for each element in the batch.
    nodes : torch.FloatTensor
        A tensor of shape (batch_size, n_nodes, n_dim) containing the coordinates of the nodes in the batch.
    elems : torch.IntTensor
        A tensor of shape (n_elems, 2) containing the connectivity of elements (pairs of nodes).
        It is assumed consistent along the batch.
    supports : torch.IntTensor
        A tensor of shape (n_supports,) containing the degrees of freedom (DOFs) that are fixed or supported.
        It is assumed consistent along the batch.
    n_dim : int, optional
        The number of spatial dimensions (default is 2).
    n_dof : int, optional
        The number of degrees of freedom per node (default is 3).

    Returns
    -------
    torch.FloatTensor
        A tensor of shape (batch_size, n_nodes * n_dof, n_nodes * n_dof) representing the global stiffness matrix for the batch.

    Notes
    -----
    The function performs the following steps:
        1)  Extract important constants such as Young's modulus, area, and node coordinates from the batched input.
        2)  Compute the angles and lengths of the structural elements for each element in the batch.
            This is done by converting the elements into vectors and calculating their length and direction.
        3)  Compute the axial stiffness of the elements using the formula:
            E * A / L, where E is Young's modulus, A is the area, and L is the length of the element.
        4)  Construct the global stiffness matrix by iterating over all elements in the batch,
            computing the local stiffness matrix, rotating it using the element's angle,
            and adding the contribution to the global stiffness matrix.
        5)  Apply support conditions by setting the appropriate rows and columns to zero and
            setting the diagonal to one for the supported degrees of freedom in the batch.
    """

    """
    Ensure that the data is batched
    """
    if len(young.shape) != 2: raise ValueError("E is expected to be a tensor of shape (batch_size, n_elems)")
    if len(area.shape) != 2: raise ValueError("A is expected to be a tensor of shape (batch_size, n_elems)")
    if len(nodes.shape) != 3: raise ValueError("nodes is expected to be a tensor of shape (batch_size, n_nodes, n_dof)")
    if len(elems.shape) != 2: raise ValueError("elems is expected to be a tensor of shape (n_elems, 2)")
    if len(supports.shape) != 1: raise ValueError("supports is expected to be a tensor of shape (n_supports,)")

    """
    Preprocess the data for stiffness matrix assembly
        1)  Extract important constants
        2)  Compute angles and lengths of the structural elements
            This is done through conversion of elements to vector.
            To maintain batch operation, we unsqueeze and expand the nodes.
        3)  Compute the axial stiffness of the structural elements
    """
    batch_size, n_nodes, n_dim = nodes.shape
    n_elems: int = elems.shape[0]

    if n_dof is None: n_dof = n_dim  # Truss hypothesis
    if device is None:
        device = "cpu"

    _start_nodes = elems.repeat(batch_size, 1, 1)[:, :, 0]
    _start_nodes = _start_nodes.unsqueeze(-1).expand(-1, -1, n_dim)  # (batch_size, n_elems, n_dim)
    _end_nodes = elems.repeat(batch_size, 1, 1)[:, :, 1]
    _end_nodes = _end_nodes.unsqueeze(-1).expand(-1, -1, n_dim)  # (batch_size, n_elems, n_dim

    elems_v = nodes.gather(1, _end_nodes) - nodes.gather(1, _start_nodes)  # (batch_size, n_elems, n_dim)

    lengths = torch.norm(elems_v, dim=2)  # (batch_size, n_elems)
    angles = torch.atan2(elems_v[:, :, 1], elems_v[:, :, 0])  # (batch_size, n_elems)

    eal = young * area / lengths  # (batch_size, n_elems)

    """
    Assembly of the stiffness matrix
        1)  Allocate size of the total matrix of shape (n_nodes * n_dof, n_nodes * n_dof)
        2)  For each element:
            a)  Compute the local stiffness matrix of the element
            b)  Compute the rotation matrix of the element
            c)  Computer the global stiffness matrix of the element
            d)  Add the elements stiffness to the global stiffness matrix
        3) Apply the support:
            a)  Set row and column of the corresponding degrees of freedom to 0.
            b)  Set the diagonal to 1.
    """
    K = torch.zeros((batch_size, n_nodes * n_dof, n_nodes * n_dof),
                    dtype=torch.float32).to(device)  # (batch_size, n_nodes * n_dof, n_nodes * n_dof)

    _k_loc = torch.tensor([[1, 0, -1, 0],
                           [0, 0, 0, 0],
                           [-1, 0, 1, 0],
                           [0, 0, 0, 0]],
                          dtype=torch.float32).to(device)  # (batch_size, 4, 4)

    for idx in range(n_elems):
        idx_start: int = elems[idx, 0] * n_dof
        idx_end: int = elems[idx, 1] * n_dof
        angle = angles[:, idx]  # (batch_size,)

        c = torch.cos(angle)  # (batch_size,)
        s = torch.sin(angle)  # (batch_size,)

        r = torch.stack(
            [torch.stack([c, s, torch.zeros_like(c), torch.zeros_like(c)], dim=-1),
             torch.stack([-s, c, torch.zeros_like(c), torch.zeros_like(c)], dim=-1),
             torch.stack([torch.zeros_like(c), torch.zeros_like(c), c, s], dim=-1),
             torch.stack([torch.zeros_like(c), torch.zeros_like(c), -s, c], dim=-1),
             ], dim=1
        )  # (batch_size, 4, 4)

        r_t = r.transpose(1, 2)  # (batch_size, 4, 4)

        k_loc = eal[:, idx].view(-1, 1, 1) * _k_loc  # (batch_size, 4, 4)
        k_glob = torch.matmul(torch.matmul(r_t, k_loc), r)  # (batch_size, 4, 4)

        K[:, idx_start:idx_start + n_dof, idx_start:idx_start + n_dof] += k_glob[:, 0:n_dof, 0:n_dof]
        K[:, idx_end:idx_end + n_dof, idx_end:idx_end + n_dof] += k_glob[:, n_dof:2 * n_dof, n_dof:2 * n_dof]
        K[:, idx_start:idx_start + n_dof, idx_end:idx_end + n_dof] += k_glob[:, 0:n_dof, n_dof:2 * n_dof]
        K[:, idx_end:idx_end + n_dof, idx_start:idx_start + n_dof] += k_glob[:, n_dof:2 * n_dof, 0:n_dof]

    K[:, supports, :] = 0.
    K[:, :, supports] = 0.
    K[:, supports, supports] = 1

    return K


__all__ = ['StiffnessToDisplacementLoss', 'StiffnessToLoadLoss', 'construct_k_from_ea']
