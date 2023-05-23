import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

from ..modules import IndexedModel
from ..utils import covariance
from ..permutation_specs import PermutationSpec, \
    apply_permutation, apply_permutation_stats


def match_activations(model0: IndexedModel, model1: IndexedModel,
                      permutation_spec: PermutationSpec,
                      eps: float = 1e-4):
    device = model0[permutation_spec.perm2axes[0][0].key].device
    permutations = []
    for module0, module1 in zip(model0.modules.values(), model1.modules.values()):
        if not module0.track_activations:
            continue

        cov = covariance(module0, module1)
        corr = cov / (torch.outer(module1.std, module0.std) + eps)

        row_ind, col_ind = linear_sum_assignment(corr.numpy(), maximize=True)
        assert (row_ind == np.arange(corr.shape[0])).all()
        permutations.append(torch.tensor(col_ind, dtype=torch.long, device=device))

    #         plt.imshow(corr)
    #         plt.grid()
    #         plt.show()
    #         non_diag = corr[~np.diag(np.ones(corr.shape[0], dtype=bool))]
    #         print(non_diag.max(), non_diag.mean())
    #         assert (col_ind == np.arange(corr.shape[0])).all()

    apply_permutation(model0, permutation_spec, permutations)
    apply_permutation_stats(model0, permutation_spec, permutations)
