import torch

from torch import nn
from torch import Tensor


class Tracker(nn.Module):
    def __init__(self, key: str, operator: nn.Module,
                 non_linearity: nn.Module = nn.Identity(),
                 disabled: bool = False):
        super().__init__()
        self.key = key
        self.operator = operator
        self.non_linearity = non_linearity
        self.tracking = False
        self.disabled = disabled
        self.reset_statistics()

    def reset_statistics(self):
        self.activations = []
        self._mean: Tensor = 0
        self._mean_squared: Tensor = 0
        self.num_runs = 0

    def forward(self, x: Tensor) -> Tensor:
        x = self.non_linearity(self.operator(x))
        if self.tracking and not self.disabled:
            # Move batch dimension to end and reshape BxCx... -> NxC
            activation = torch.movedim(x.detach().cpu(), 1, -1).reshape(-1, x.shape[1])
            self.activations.append(activation)
            self._mean = self._mean + activation.mean(dim=0)
            self._mean_squared = self._mean_squared + (activation ** 2).mean(dim=0)
            self.num_runs += 1
        return x

    @property
    def mean(self) -> Tensor:
        return self._mean / (self.num_runs
                             if self.num_runs > 0 else 1)

    @property
    def mean_squared(self) -> Tensor:
        return self._mean_squared / (self.num_runs
                                     if self.num_runs > 0 else 1)
