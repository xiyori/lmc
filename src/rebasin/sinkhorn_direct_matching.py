import torch
import numpy as np

from ..modules import IndexedModel
from src.models import SinkhornRebasinModel
from ..utils import interpolate, set_random_seed
from ..permutation_specs import PermutationSpec, \
    apply_permutation


def sinkhorn_direct(model0: IndexedModel, model1: IndexedModel,
                    permutation_spec: PermutationSpec,
                    mode: str = "random", repair: str = "none",
                    max_iter: int = 20, sinkhorn_iter: int = 20,
                    sinkhorn_tau: float = 1.0, seed: int = 42):
    if seed is not None:
        set_random_seed(seed, deterministic=False)

    model0.requires_grad(False)
    model1.requires_grad(False)
    model0.eval()
    model1.eval()
    device = model0[permutation_spec.perm2axes[0][0].key].device
    perm_sizes = [model0[axes[0].key].shape[axes[0].axis]
                  for axes in permutation_spec.perm2axes]

    rebasin_model = SinkhornRebasinModel(permutation_spec, perm_sizes,
                                         sinkhorn_iter, sinkhorn_tau,
                                         init="random")
    rebasin_model.to(device)
    optimizer = torch.optim.AdamW(rebasin_model.parameters(), lr=0.1)

    for iteration in range(max_iter):
        # Training
        rebasin_model.train()  # This uses soft permutation matrices
        train_loss = 0
        for x_batch, y_batch in trainloader:
            data = x_batch.to(device)
            target = y_batch.to(device)

            optimizer.zero_grad()
            permuted_model0 = rebasin_model(model0)
            if mode == "random":
                alpha = np.random.rand()
            else:
                alpha = 0.5
            interpolated = interpolate(permuted_model0, model1, alpha, copy=False)

            if repair == "closed":
                repair_closed(permuted_model0, model1, interpolated, alpha)
            elif repair == "trainmode":
                repair_trainmode(interpolated)
            elif repair == "sphere":
                repair_sphere(permuted_model0, model1, interpolated, alpha)

            output = interpolated(data)

            loss = criterion_train(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            del permuted_model0
        train_loss /= len(trainloader)

        # Validation
        rebasin_model.eval()  # This uses hard permutation matrices
        with torch.no_grad():
            permuted_model0 = rebasin_model(model0)
            interpolated = interpolate(permuted_model0, model1, alpha, copy=False)

            if repair == "closed":
                repair_closed(permuted_model0, model1, interpolated, alpha)
            elif repair == "trainmode":
                repair_trainmode(interpolated)
            elif repair == "sphere":
                repair_sphere(permuted_model0, model1, interpolated, alpha)
        valid_loss = test(interpolated, validloader)[0]
        del permuted_model0

        print(
            "Iteration {:02d}: training loss {:1.3f}, validation loss {:1.3f}".format(
                iteration, train_loss, valid_loss
            )
        )
        if valid_loss == 0:
            break

    permutations = rebasin_model.estimate_permutations()
    apply_permutation(model0, permutation_spec, permutations)
