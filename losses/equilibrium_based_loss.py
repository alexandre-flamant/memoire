import torch
from torch import nn
from typing import List, Dict

class NodeEquilibriumLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, EA:torch.Tensor, e:torch.Tensor, q:torch.Tensor,
                r:torch.Tensor, m_incidence:List[Dict[int, torch.Tensor]]) -> torch.Tensor:
        n_sample = len(EA)
        residual: torch.Tensor = torch.zeros((n_sample, q.shape[1], 2), device=EA.device)
        for node_idx, node_incidence in enumerate(m_incidence):
            axial_forces: torch.Tensor = EA * e
            for idx, vect in node_incidence.items():
                residual[:, node_idx] += vect.repeat((n_sample, 1)) * axial_forces[:, idx].unsqueeze(1)
            residual[:, node_idx] -= q[:, node_idx]
            residual[:, node_idx] -= r[:, node_idx]

        return residual.square().mean()

__all__ = ['NodeEquilibriumLoss']