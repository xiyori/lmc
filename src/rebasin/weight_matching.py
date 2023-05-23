import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

from ..modules import IndexedModel
from ..utils import normalize_model, set_random_seed
from ..permutation_specs import PermutationSpec, get_permuted_param, \
    apply_permutation


def match_weights(model0: IndexedModel, model1: IndexedModel,
                  permutation_spec: PermutationSpec, normalize: bool = False,
                  max_iter: int = 100, eps: float = 1e-12,
                  seed: int = 42):
    if seed is not None:
        set_random_seed(seed)



    orig_model0 = model0
    if normalize:
        model0 = normalize_model(model0)
        model1 = normalize_model(model1)

    device = model0[permutation_spec.perm2axes[0][0].key].device
    perm_sizes = [model0[axes[0].key].shape[axes[0].axis]
                  for axes in permutation_spec.perm2axes]
    permutations = [torch.arange(size, dtype=torch.long, device=device)
                    for size in perm_sizes]
    for iteration in range(max_iter):
        progress = False
        for p in np.random.choice(len(permutations), len(permutations), replace=False):
            size = perm_sizes[p]
            A = torch.zeros((size, size), device=device)
            for key, axis in permutation_spec.perm2axes[p]:
                if not model0[key.split(".")[0]].match_weights:
                    continue

                w1 = model1[key].detach()
                w0 = get_permuted_param(model0, permutation_spec, permutations,
                                        key, except_axis=axis)
                w1 = torch.movedim(w1, axis, 0).reshape(size, -1)
                w0 = torch.movedim(w0, axis, 0).reshape(size, -1)
                A += w1 @ w0.T

            row_ind, col_ind = linear_sum_assignment(A.cpu().numpy(), maximize=True)
            assert (row_ind == np.arange(A.shape[0])).all()
            new_permutation = torch.tensor(col_ind, dtype=torch.long, device=device)

            oldL = torch.diag(A[permutations[p]]).sum()
            newL = torch.diag(A[new_permutation]).sum()
            progress = progress or torch.abs(newL - oldL) > eps

            permutations[p] = new_permutation

        if not progress:
            break

    apply_permutation(orig_model0, permutation_spec, permutations)
