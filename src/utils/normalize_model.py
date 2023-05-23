import torch
from torch import nn
from copy import deepcopy

from ..modules import IndexedModel


def simple_norm(x: torch.Tensor, *args, **kwargs):
    return torch.sum(x ** 2, *args, **kwargs) ** (1 / 2)


def normalize_model(model: IndexedModel) -> nn.Module:
    normalized_model = deepcopy(model)
    for module in normalized_model.modules.values():
        operator_name = module.operator.__class__.__name__.lower()
        if "conv2d" in operator_name:
            dims_to_norm = list(range(1, module.operator.weight.dim()))
            norm = simple_norm(module.operator.weight.detach(), dim=dims_to_norm, keepdim=True)

            weight = module.operator.weight.detach()
            module.operator.weight.data = weight / norm
            try:
                bias = module.operator.bias.detach()
                module.operator.bias.data = bias / norm.squeeze()
            except AttributeError:
                pass
    return normalized_model
