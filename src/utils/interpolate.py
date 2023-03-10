from torch import nn


def interpolate(model0: nn.Module, model1: nn.Module, out: nn.Module, alpha: float = 0.5):
    weights0 = model0.state_dict()
    weights1 = model1.state_dict()
    new_weights = {key: (1 - alpha) * weights0[key] + alpha * weights1[key]
                   for key in out.state_dict().keys()}
    out.load_state_dict(new_weights)
