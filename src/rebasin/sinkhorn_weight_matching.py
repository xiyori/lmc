import torch

from ..modules import IndexedModel
from src.models import SinkhornRebasinModel
from src.loss import DistLoss
from ..utils import normalize_model
from ..permutation_specs import PermutationSpec, \
    apply_permutation


def match_weights_sinkhorn(model0: IndexedModel, model1: IndexedModel,
                           permutation_spec: PermutationSpec, normalize: bool = False,
                           max_iter: int = 50, sinkhorn_iter: int = 20,
                           sinkhorn_tau: float = 1.0):
    orig_model0 = model0
    if normalize:
        model0 = normalize_model(model0)
        model1 = normalize_model(model1)

    model0.requires_grad(False)
    model1.requires_grad(False)
    device = model0[permutation_spec.perm2axes[0][0].key].device
    perm_sizes = [model0[axes[0].key].shape[axes[0].axis]
                  for axes in permutation_spec.perm2axes]

    #     random_perms = [torch.tensor(np.random.choice(size, size, replace=False),
    #                                  dtype=torch.long, device=device)
    #                     for size in perm_sizes]
    #     model1 = apply_permutation(model0, permutation_spec, random_perms, preserve_grad=True)
    #     model1.requires_grad(False)

    rebasin_model = SinkhornRebasinModel(permutation_spec, perm_sizes,
                                         sinkhorn_iter, sinkhorn_tau)
    rebasin_model.to(device)
    criterion = DistLoss(permutation_spec)
    optimizer = torch.optim.AdamW(rebasin_model.parameters(), lr=0.1)

    for iteration in range(max_iter):
        # Training
        rebasin_model.train()  # This uses soft permutation matrices

        optimizer.zero_grad()

        permuted_model0 = rebasin_model(model0)
        train_loss = criterion(permuted_model0, model1)
        train_loss.backward()
        optimizer.step()  # Only updates the permutation matrices
        del permuted_model0

        # Validation
        rebasin_model.eval()  # This uses hard permutation matrices

        with torch.no_grad():
            permuted_model0 = rebasin_model(model0)
            valid_loss = criterion(permuted_model0, model1)
            del permuted_model0

        print(
            "Iteration {:02d}: training loss {:1.3f}, validation loss {:1.3f}".format(
                iteration, train_loss, valid_loss
            )
        )
        if valid_loss == 0:
            break

    permutations = rebasin_model.estimate_permutations()
    apply_permutation(orig_model0, permutation_spec, permutations)
