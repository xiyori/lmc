# Taken from https://worksheets.codalab.org/worksheets/0x641008eb0b1b4768b865b58eddbe419c
# Sinkhorn Re-basin paper


import torch

from torch import nn, Tensor
from scipy.optimize import linear_sum_assignment
from typing import Sequence

from ..modules import IndexedModel, Sinkhorn
from ..permutation_specs import PermutationSpec, apply_permutation


class SinkhornRebasinModel(nn.Module):
    def __init__(self, permutation_spec: PermutationSpec,
                 perm_sizes: Sequence[int],
                 n_iter: int = 20, tau: float = 1.0,
                 init: str = "identity"):
        super().__init__()
        self.permutation_spec = permutation_spec
        self.n_iter = n_iter
        self.tau = tau
        self.permutations = nn.ParameterList(
            [nn.Parameter(torch.eye(size) +
                          (torch.randn(size, size) * 0.1
                           if init == "random"
                           else 0)) for size in perm_sizes]
        )

    def forward(self, model: IndexedModel, out: IndexedModel) -> IndexedModel:
        if self.training:
            permutations = [
                Sinkhorn.apply(
                    -p,
                    torch.ones(p.shape[0]).to(p.device),
                    torch.ones(p.shape[1]).to(p.device),
                    self.n_iter,
                    self.tau
                ) for p in self.permutations]
        else:
            permutations = self.estimate_permutations()
        apply_permutation(model, self.permutation_spec, permutations, out)
        return out

    def estimate_permutations(self) -> Sequence[Tensor]:
        return [
            torch.tensor(
                linear_sum_assignment(p.cpu().detach().numpy(), maximize=True)[1],
                dtype=torch.long, device=p.device
            ) for p in self.permutations]
