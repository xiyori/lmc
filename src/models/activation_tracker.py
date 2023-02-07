import torch

from torch import nn
from torch import Tensor


class Tracker(nn.Module):
    def __init__(self, key: str, operator: nn.Module,
                 non_linearity: nn.Module = nn.Identity(),
                 mode: str = "pre_activation", enabled: bool = True):
        super().__init__()
        self.key = key
        self.operator = operator
        self.non_linearity = non_linearity
        self.tracking = False
        self.mode = mode
        self.enabled = enabled
        self.reset_statistics()

    def reset_statistics(self):
        self.activations = []
        self._mean: Tensor = 0
        self._mean_squared: Tensor = 0
        self.num_runs: int = 0

    def forward(self, x: Tensor) -> Tensor:
        x = self.operator(x)
        if self.mode == "post_activation":
            x = self.non_linearity(x)
        if self.tracking and self.enabled:
            # Move batch dimension to end and reshape BxCx... -> NxC
            activation = torch.movedim(x.detach().cpu(), 1, -1).reshape(-1, x.shape[1])
            self.activations.append(activation)
            self._mean = self._mean + activation.mean(dim=0)
            self._mean_squared = self._mean_squared + (activation ** 2).mean(dim=0)
            self.num_runs += 1
        if self.mode == "pre_activation":
            return self.non_linearity(x)
        return x

    def apply_permutation(self, permutation: Tensor):
        permutation = permutation.cpu()
        for i in range(len(self.activations)):
            self.activations[i] = torch.index_select(self.activations[i], 1, permutation)
        self._mean = torch.index_select(self._mean, 0, permutation)
        self._mean_squared = torch.index_select(self._mean_squared, 0, permutation)

    @property
    def mean(self) -> Tensor:
        return self._mean / (self.num_runs
                             if self.num_runs > 0 else 1)

    @property
    def mean_squared(self) -> Tensor:
        return self._mean_squared / (self.num_runs
                                     if self.num_runs > 0 else 1)

    @property
    def var(self) -> Tensor:
        return self.num_runs / (self.num_runs - 1) * (self.mean_squared - self.mean ** 2)

    @property
    def std(self) -> Tensor:
        return torch.sqrt(self.var)
