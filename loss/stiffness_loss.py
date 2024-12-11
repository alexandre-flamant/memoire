import torch
from torch.nn import Module
from torch.nn.functional import mse_loss

from structural.structure import AbstractStructure


class DirectStiffnessLoss(Module):
    def __init__(self, structure_cls: AbstractStructure) -> None:
        super(DirectStiffnessLoss, self).__init__()

    def forward(self, k: torch.Tensor, u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss based on the Direct Stiffness Method.

        Parameters
        ----------
        k : torch.Tensor
            A 2D tensor representing the stiffness matrix of shape (n, n) with dtype float64.
        u : torch.Tensor
            A 2D tensor representing the displacement vector of shape (n, 1) with dtype float64.
        q : torch.Tensor
            A 2D tensor representing the force vector of shape (n, 1) with dtype float64.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed Mean Squared Error (MSE) loss between
            the predicted and actual force vectors.

        Notes
        -----
        The loss is computed as the MSE between the predicted force vector (obtained by
        multiplying the stiffness matrix K with the force vector Q) and the true force vector Q.
        """

        # Perform matrix multiplication
        q_pred = torch.matmul(k, q)
        # Calculate and return the MSE loss
        return mse_loss(q_pred, q)

__all__ = ['DirectStiffnessLoss']
