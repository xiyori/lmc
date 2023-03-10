# Taken from https://github.com/samuela/git-re-basin
# Git Re-Basin paper


import torch

from torch import Tensor
from typing import NamedTuple, Dict, Sequence
from collections import defaultdict

from ..modules import TrackedModel


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


def get_permuted_param(model: TrackedModel, permutation_spec: PermutationSpec,
                       permutations: Sequence, key: str, except_axis: int = None) -> Tensor:
    weight = model[key].detach()
    for axis, p in enumerate(permutation_spec.axes2perm[key]):
        if p is None or axis == except_axis:
            continue
        if permutations[p].dim() == 1:
            weight = torch.index_select(weight, axis, permutations[p])
        else:
            weight = torch.movedim(
                torch.movedim(weight, axis, -1) @ permutations[p],
                -1, axis
            )
    return weight


def apply_permutation(model: TrackedModel, output, permutation_spec: PermutationSpec,
                      permutations: Sequence):
    for key in permutation_spec.axes2perm:
        output[key].copy_(get_permuted_param(model, permutation_spec, permutations, key))


def apply_permutation_stats(model: TrackedModel, permutation_spec: PermutationSpec,
                            permutations: Sequence):
    permuted = set()
    for key, axis_perms in permutation_spec.axes2perm.items():
        key = key.split(".")[0]
        if key in permuted:
            continue
        permuted.add(key)
        if axis_perms[0] is not None:
            model[key].apply_permutation(permutations[axis_perms[0]])
