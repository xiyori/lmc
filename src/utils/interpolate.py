from torch import nn
from copy import deepcopy


def interpolate(model0: nn.Module, model1: nn.Module,
                alpha: float = 0.5, copy: bool = True) -> nn.Module:
    if copy:
        interpolated = deepcopy(model0)
        for w0, w1, w_i in zip(model0.state_dict().values(),
                               model1.state_dict().values(),
                               interpolated.state_dict().values()):
            w_i.copy_((1 - alpha) * w0 + alpha * w1)
        return interpolated

    for w0, w1 in zip(model0.parameters(),
                      model1.parameters()):
        w0.mul_(1 - alpha)
        w0.add_(alpha * w1)
    return model0
