import torch
from ..modules import Indexer


def covariance(module0: Indexer, module1: Indexer):
    outer = 0
    for a0, a1 in zip(module0.activations, module1.activations):
        outer = outer + (a1.T @ a0) / a0.shape[0]
    outer /= module0.num_runs
    cov = outer - torch.outer(module1.mean, module0.mean)
    return cov
