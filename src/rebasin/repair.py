import torch

from ..modules import IndexedModel
from ..utils import simple_norm, covariance


def repair_closed(model0: IndexedModel, model1: IndexedModel,
                  interpolated: IndexedModel, alpha: float = 0.5):
    for module0, module1, module_i in zip(model0.modules.values(),
                                          model1.modules.values(),
                                          interpolated.modules.values()):
        cov = covariance(module0, module1)
        rescale = ((1 - alpha) * module0.std + alpha * module1.std) / \
                  ((1 - alpha) ** 2 * module0.var + alpha ** 2 * module1.var +
                   2 * alpha * (1 - alpha) * torch.diag(cov)) ** (1 / 2)
        rescale = rescale.to(module_i.operator.weight.device)

        weight = module_i.operator.weight.detach()
        module_i.operator.weight.data = weight * rescale.view(-1, *([1] * (weight.dim() - 1)))
        try:
            bias = module_i.operator.bias.detach()
            module_i.operator.bias.data = bias * rescale
        except AttributeError:
            pass


def repair_trainmode(interpolated: IndexedModel):
    for module in interpolated.modules.values():
        if "norm" in module.operator.__class__.__name__.lower():
            module.operator.running_mean = None
            module.operator.running_var = None


def repair_sphere(model0: IndexedModel, model1: IndexedModel,
                  interpolated: IndexedModel, alpha: float = 0.5):
    repair_trainmode(interpolated)
    for module0, module1, module_i in zip(model0.modules.values(),
                                          model1.modules.values(),
                                          interpolated.modules.values()):
        # if "norm" in module0.operator.__class__.__name__.lower():
        #     dims_to_norm = list(range(1, module0.operator.running_mean.dim()))
        #     norm0 = simple_norm(module0.operator.running_mean.detach(), dim=dims_to_norm, keepdim=True)
        #     norm1 = simple_norm(module1.operator.running_mean.detach(), dim=dims_to_norm, keepdim=True)
        #     norm = (1 - alpha) * norm0 + alpha * norm1
        #     rescale = norm / simple_norm(module_i.operator.running_mean.detach(), dim=dims_to_norm, keepdim=True)
        #
        #     running_mean = module_i.operator.running_mean.detach()
        #     module_i.operator.running_mean.data = running_mean * rescale
        #     running_var = module_i.operator.running_var.detach()
        #     module_i.operator.running_var.data = running_var * rescale.squeeze()

        dims_to_norm = list(range(1, module0.operator.weight.dim()))
        norm0 = simple_norm(module0.operator.weight.detach(), dim=dims_to_norm, keepdim=True)
        norm1 = simple_norm(module1.operator.weight.detach(), dim=dims_to_norm, keepdim=True)
        norm = (1 - alpha) * norm0 + alpha * norm1
        rescale = norm / simple_norm(module_i.operator.weight.detach(), dim=dims_to_norm, keepdim=True)

        weight = module_i.operator.weight.detach()
        module_i.operator.weight.data = weight * rescale
        try:
            bias = module_i.operator.bias.detach()
            module_i.operator.bias.data = bias * rescale.squeeze()
        except AttributeError:
            pass
