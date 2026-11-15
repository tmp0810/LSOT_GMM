"""Batched 2-D Gaussian tuple costs for the MW2 multi-marginal LP."""
import math

import numpy as np
from scipy.sparse import csc_matrix
import torch


def tuple_constraints(shape, batch_size):
    """Build the unit-coefficient marginal matrix in flat tuple order."""
    count, marginals = math.prod(shape), len(shape)
    index_type = np.int32 if count * marginals < np.iinfo(np.int32).max else np.int64
    row = np.empty((count, marginals), dtype=index_type)
    offsets = np.cumsum((0,) + shape[:-1])
    strides = [math.prod(shape[j+1:]) for j in range(marginals)]
    for start in range(0, count, batch_size):
        stop = min(count, start+batch_size)
        flat = np.arange(start, stop, dtype=index_type)
        for j, size in enumerate(shape):
            row[start:stop, j] = (flat//strides[j]) % size + offsets[j]
    ptr = np.arange(0, (count+1)*marginals, marginals, dtype=index_type)
    return csc_matrix((np.ones(count*marginals), row.reshape(-1), ptr), shape=(sum(shape), count))


def det2(matrix):
    return matrix[..., 0, 0]*matrix[..., 1, 1] - matrix[..., 0, 1]*matrix[..., 1, 0]


def trace(matrix):
    return matrix.diagonal(dim1=-2, dim2=-1).sum(-1)


def symmetric(matrix):
    return .5*(matrix+matrix.transpose(-1, -2))


def sqrt2(matrix, identity):
    matrix = symmetric(matrix)
    det_root = det2(matrix).clamp_min(0).sqrt()
    scale = (trace(matrix)+2*det_root).clamp_min(0).sqrt().clamp_min(torch.finfo(matrix.dtype).tiny)
    return (matrix+det_root[..., None, None]*identity)/scale[..., None, None]


class BatchedGaussianTuples:
    """Keep one block of tuple geometry on device, retain costs on the host."""

    def __init__(self, means, covariances, weights, device, batch_size):
        if any(m.shape[-1] != 2 for m in means):
            raise ValueError('Batched MW2 tuple geometry supports only 2-D mixtures')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('tuple_batch_size must be positive')
        self.shape = tuple(len(m) for m in means)
        self.count = math.prod(self.shape)
        self.strides = tuple(math.prod(self.shape[j+1:]) for j in range(len(means)))
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.means = [torch.as_tensor(m, device=device, dtype=torch.float64) for m in means]
        self.covariances = [torch.as_tensor(c, device=device, dtype=torch.float64) for c in covariances]
        self.weights = torch.as_tensor(weights, device=device, dtype=torch.float64)
        self.identity = torch.eye(2, device=device, dtype=torch.float64)

    def _gather(self, flat):
        codes = [(flat//stride) % size for stride, size in zip(self.strides, self.shape)]
        means = torch.stack([m[idx] for m, idx in zip(self.means, codes)], dim=1)
        covs = torch.stack([c[idx] for c, idx in zip(self.covariances, codes)], dim=1)
        return means, covs

    def _fit(self, means, covs, iterations, tolerance=0., *, diagnostics=False):
        center = torch.einsum('j,tjd->td', self.weights, means)
        cov = self.identity.expand(len(means), 2, 2)
        change = None
        for step in range(1, iterations+1):
            previous = cov
            root = sqrt2(previous, self.identity)[:, None]
            cov = torch.einsum('j,tjab->tab', self.weights, sqrt2(root @ covs @ root, self.identity))
            if tolerance or (diagnostics and step == iterations):
                change = float((torch.linalg.vector_norm(cov-previous, dim=(-2, -1)) /
                                torch.linalg.vector_norm(previous, dim=(-2, -1)).clamp_min(1e-15)).max())
                if tolerance and change <= tolerance:
                    break
        return center, symmetric(cov), step, change

    @torch.inference_mode()
    def costs(self, iterations, tolerance):
        costs = np.empty(self.count, dtype=float)
        steps, changes = [], []
        for start in range(0, self.count, self.batch_size):
            stop = min(self.count, start+self.batch_size)
            means, covs = self._gather(torch.arange(start, stop, device=self.device))
            center, cov, step, change = self._fit(means, covs, iterations, tolerance, diagnostics=True)
            cross_trace = ((covs*cov[:, None].transpose(-1, -2)).sum((-2, -1))
                           + 2*(det2(covs)*det2(cov)[:, None]).clamp_min(0).sqrt()).clamp_min(0).sqrt()
            squared = ((means-center[:, None]).square().sum(-1)
                       + trace(covs)+trace(cov)[:, None]-2*cross_trace).clamp_min(0)
            costs[start:stop] = (squared @ self.weights).cpu().numpy()
            steps.append(step)
            changes.append(change)
        info = dict(covariance_iterations=max(steps), covariance_relative_change=max(changes),
                    covariance_tolerance_met=(max(changes) <= tolerance) if tolerance else None)
        return costs, steps, info

    @torch.inference_mode()
    def active_centers(self, indices, batch_steps):
        """Recompute LP support with the same iteration count used to score it."""
        centers = np.empty((len(indices), 2))
        covariances = np.empty((len(indices), 2, 2))
        # Different source blocks can have different iteration counts.
        # Group LP support by step count, so sparse support runs in large batches.
        steps = np.asarray(batch_steps)[indices//self.batch_size]
        for step in np.unique(steps):
            selected = np.flatnonzero(steps == step)
            for start in range(0, len(selected), self.batch_size):
                out = selected[start:start+self.batch_size]
                flat = torch.as_tensor(indices[out], device=self.device)
                means, covs = self._gather(flat)
                center, cov, _, _ = self._fit(means, covs, int(step))
                centers[out] = center.cpu().numpy()
                covariances[out] = cov.cpu().numpy()
        return centers, covariances
