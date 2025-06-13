import torch
from torch import nn

class EnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, EA, strain, u, q, L) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.compute_internal_energy(EA, L, strain),
                                            self.compute_external_work(u, q))

    def compute_internal_energy(self, EA: torch.Tensor, L, strain) -> torch.Tensor:
        return (EA * L * (strain) ** 2).sum(dim=1) # Equivalent to batched Sum(EA_i/L_i * dL_i^2)

    def compute_external_work(self, u, q) -> torch.Tensor:
        return (u * q).sum(dim=(1,2)) # Equivalent to batched Sum(q_i @ u_i)

__all__ = ["EnergyLoss"]