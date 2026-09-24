"""Gaussian geometry in float32/float64 on CPU or CUDA."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class GMM:
    weights: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor

    def __post_init__(self):
        k, d = self.means.shape
        if k == 0 or self.weights.shape != (k,) or self.covariances.shape != (k, d, d):
            raise ValueError("expected weights (K,), means (K,d), covariances (K,d,d)")
        tensors = (self.weights, self.means, self.covariances)
        if any(t.device != self.means.device or t.dtype != self.means.dtype for t in tensors):
            raise ValueError("all GMM tensors must share device and floating dtype")
        if self.means.dtype not in (torch.float32, torch.float64):
            raise ValueError("GMM tensors must be float32 or float64")
        if any(not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError("GMM contains nonfinite values")
        if bool((self.weights <= 0).any()) or not torch.isclose(self.weights.sum(), self.weights.new_tensor(1.0)):
            raise ValueError("GMM weights must be positive and sum to one")
        if not torch.allclose(self.covariances, self.covariances.transpose(-1, -2)):
            raise ValueError("covariances must be symmetric")
        if bool((torch.linalg.eigvalsh(self.covariances) <= 0).any()):
            raise ValueError("covariances must be positive definite")

    @classmethod
    def from_numpy(cls, weights, means, covariances, *, device="cpu", dtype=torch.float64):
        w = torch.as_tensor(np.asarray(weights), dtype=dtype, device=device)
        return cls(w / w.sum(), torch.as_tensor(np.asarray(means), dtype=dtype, device=device),
                   torch.as_tensor(np.asarray(covariances), dtype=dtype, device=device))

    @property
    def count(self):
        return len(self.weights)

    @property
    def dimension(self):
        return self.means.shape[1]


def covariance_roots(covariances):
    values, vectors = torch.linalg.eigh(covariances)
    root = (vectors * values.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    inverse_root = (vectors * values.rsqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return root, inverse_root


def symmetric_sqrt(matrix):
    matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)


def gaussian_pair_costs(source, target, rows, cols, *, batch_size=65536):
    """Evaluate only requested pairs; never allocate a K0 x K1 matrix."""
    roots, _ = covariance_roots(source.covariances)
    costs = source.weights.new_empty(rows.numel())
    for start in range(0, rows.numel(), batch_size):
        sl = slice(start, start + batch_size)
        i, j = rows[sl], cols[sl]
        middle = roots[i] @ target.covariances[j] @ roots[i]
        middle = (middle + middle.transpose(-1, -2)) * 0.5
        cross_trace = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum(-1)
        mean_cost = (source.means[i] - target.means[j]).square().sum(-1)
        trace = torch.diagonal(source.covariances[i] + target.covariances[j], dim1=-2, dim2=-1).sum(-1)
        costs[sl] = (mean_cost + trace - 2 * cross_trace).clamp_min(0)
    return costs
