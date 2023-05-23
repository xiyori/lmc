# Taken from https://github.com/samuela/git-re-basin
# Git Re-Basin paper


import torch

from torch import nn, Tensor
from copy import deepcopy
from collections import defaultdict
from typing import NamedTuple, Dict, Sequence

from ..modules import IndexedModel


class PermutationSpec(NamedTuple):
    perm2axes: list
    axes2perm: dict


class AxisInfo(NamedTuple):
    key: str
    axis: int


def to_list(a: Dict):
    return [a[i] for i in range(len(a))]


def permutation_spec_from_axes_to_perm(axes2perm: dict) -> PermutationSpec:
    perm2axes = defaultdict(list)
    for key, axis_perms in axes2perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm2axes[perm].append(AxisInfo(key=key, axis=axis))
    return PermutationSpec(perm2axes=to_list(dict(perm2axes)), axes2perm=axes2perm)


def get_permuted_param(model: IndexedModel, permutation_spec: PermutationSpec,
                       permutations: Sequence, key: str, except_axis: int = None) -> Tensor:
    weight = model[key].detach()
    for axis, p in enumerate(permutation_spec.axes2perm[key]):
        if p is None or axis == except_axis:
            continue
        p = permutations[p]
        # if p.shape[0] < weight.shape[axis]:
        #     if weight.shape[axis] % p.shape[0] == 0:
        #         multiplier = weight.shape[axis] // p.shape[0]
        #         if p.dim() == 1:
        #             p = torch.tensor([
        #                 start_index + i
        #                 for start_index in p * multiplier
        #                 for i in range(multiplier)
        #             ]).to(p.device)
        #         else:
        #             p = torch.cat([
        #                 torch.cat([
        #                     (torch.diag(torch.arange(multiplier, dtype=p.dtype))
        #                      if elem else torch.zeros(multiplier, multiplier, dtype=p.dtype))
        #                     for elem in row], dim=1)
        #                 for row in p], dim=0).to(p.device)
        #     else:
        #         raise ValueError(
        #             "permutation of shape",
        #             p.shape,
        #             "cannot be applied to weight of shape",
        #             weight.shape
        #         )
        if p.dim() == 1:
            weight = torch.index_select(weight, axis, p)
        else:
            weight = torch.movedim(
                torch.movedim(weight, axis, -1) @ p.T,
                -1, axis
            )
    return weight


def apply_permutation(model: IndexedModel, permutation_spec: PermutationSpec,
                      permutations: Sequence, copy: bool = False) -> IndexedModel:
    if copy:
        out = deepcopy(model)
    else:
        out = model

    for key in permutation_spec.axes2perm:
        out[key].requires_grad = False
        out[key].copy_(get_permuted_param(model, permutation_spec, permutations, key))
    return out


def apply_permutation_stats(model: IndexedModel, permutation_spec: PermutationSpec,
                            permutations: Sequence):
    permuted = set()
    for key, axis_perms in permutation_spec.axes2perm.items():
        key = key.split(".")[0]
        if key in permuted:
            continue
        permuted.add(key)
        if axis_perms[0] is not None:
            model[key].apply_permutation(permutations[axis_perms[0]])
