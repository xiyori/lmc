from typing import NamedTuple, Dict, Sequence
from collections import defaultdict


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
