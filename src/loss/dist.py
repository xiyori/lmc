import torch.nn as nn

from ..modules import IndexedModel
from ..permutation_specs import PermutationSpec


class DistLoss(nn.Module):
    def __init__(self, permutation_spec: PermutationSpec):
        super().__init__()
        self.permutation_spec = permutation_spec

    def forward(self, model0: IndexedModel, model1: IndexedModel):
        loss = 0
        num_params = 0
        for key in self.permutation_spec.axes2perm:
            if not model0[key.split(".")[0]].match_weights:
                continue
            num_params += 1
            loss += (model0[key] - model1[key]).abs().mean()
        loss /= num_params
        return loss
