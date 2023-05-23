import torch

from ..modules import IndexedModel


def copy_with_weights(model: IndexedModel) -> IndexedModel:
    copy = model.copy()
    with torch.no_grad():
        for w, w_c in zip(model.state_dict().values(),
                          copy.state_dict().values()):
            w_c.copy_(w)
    return copy
