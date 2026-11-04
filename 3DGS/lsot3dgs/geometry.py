"""Chunked full-covariance Gaussian W2 and fixed-plan Bures barycenters."""
from dataclasses import dataclass
import torch

EIGH_CHUNK = 4096


def sym(x):
    return (x + x.transpose(-1, -2)) * 0.5


def eig(x):
    values, vectors = [], []
    for block in x.split(EIGH_CHUNK):
        v, q = torch.linalg.eigh(sym(block))
        values.append(v)
        vectors.append(q)
    return torch.cat(values), torch.cat(vectors)


def repair_spd(x, *, absolute_floor=None):
    """Repair roundoff only by default; reject materially indefinite inputs."""
    v, q = eig(x)
    floor = 64 * torch.finfo(x.dtype).eps * v[:, -1].clamp_min(torch.finfo(x.dtype).tiny)
    if not bool(torch.isfinite(v).all()) or bool((v[:, -1] <= 0).any()):
        raise ValueError("Nonfinite or nonpositive covariance")
    if bool((v[:, 0] < -floor).any()):
        raise ValueError("Materially indefinite covariance")
    if absolute_floor is None:
        repaired = v <= 0
        corrected = torch.where(repaired, floor[:, None], v)
    else:
        repaired = v < absolute_floor
        corrected = v.clamp_min(absolute_floor)
    count = int(repaired.any(1).sum())
    if not count:
        return sym(x), 0
    return sym((q * corrected[:, None, :]) @ q.transpose(-1, -2)), count


def roots(x, inverse=False):
    v, q = eig(x)
    if bool((v <= 0).any()):
        raise ValueError("Expected SPD covariance for square root/inverse")
    r = (q * v.sqrt()[:, None, :]) @ q.transpose(-1, -2)
    if inverse:
        return r, (q * v.rsqrt()[:, None, :]) @ q.transpose(-1, -2)
    return r


def psd_sqrt(x):
    v, q = eig(x)
    # The product of SPD matrices may have small negative roundoff eigenvalues.
    return (q * v.clamp_min(0).sqrt()[:, None, :]) @ q.transpose(-1, -2)


@dataclass(frozen=True)
class GaussianSet:
    weights: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor

    def __post_init__(self):
        n = len(self.weights)
        if n == 0 or self.means.shape != (n, 3) or self.covariances.shape != (n, 3, 3):
            raise ValueError("Expected positive weights (N,), means (N,3), covariances (N,3,3)")
        items = (self.weights, self.means, self.covariances)
        if any(x.dtype != self.means.dtype or x.device != self.means.device for x in items):
            raise ValueError("Gaussian tensors must share dtype and device")
        if self.means.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64")
        if any(not bool(torch.isfinite(x).all()) for x in items) or bool((self.weights <= 0).any()):
            raise ValueError("Nonfinite parameters or nonpositive weights")
        if abs(float(self.weights.sum()) - 1) > (1e-10 if self.means.dtype == torch.float64 else 1e-5):
            raise ValueError("Weights must be normalized")
        if not torch.allclose(self.covariances, self.covariances.transpose(-1, -2)):
            raise ValueError("Covariances must be symmetric")
        for block in self.covariances.split(EIGH_CHUNK):
            if bool((torch.linalg.eigvalsh(block) <= 0).any()):
                raise ValueError("Covariances must be positive definite")

    @property
    def count(self):
        return len(self.weights)

    @property
    def dimension(self):
        return 3


class GaussianCost:
    def __init__(self, source, chunk=4096):
        self.source = source
        self.chunk = min(int(chunk), EIGH_CHUNK)
        if self.chunk < 1:
            raise ValueError("Cost chunk must be positive")
        self.source_roots = roots(source.covariances)
        self.source_trace = source.covariances.diagonal(dim1=-2, dim2=-1).sum(-1)

    def pairs(self, target, rows, cols):
        out = self.source.weights.new_empty(len(rows))
        target_trace = target.covariances.diagonal(dim1=-2, dim2=-1).sum(-1)
        for k in range(0, len(rows), self.chunk):
            i, j = rows[k:k+self.chunk], cols[k:k+self.chunk]
            middle = sym(self.source_roots[i] @ target.covariances[j] @ self.source_roots[i])
            cross = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum(-1)
            out[k:k+self.chunk] = ((self.source.means[i] - target.means[j]).square().sum(-1)
                                  + self.source_trace[i] + target_trace[j] - 2 * cross).clamp_min(0)
        return out

    def plan_cost(self, target, plan):
        total = self.source.weights.new_zeros(())
        for i, j, mass in plan.edges(self.chunk):
            total += torch.dot(mass, self.pairs(target, i, j))
        return float(total)


def bures_update(source, target, plan, *, steps=64, tolerance=5e-4, chunk=4096):
    """Conditional Gaussian barycenters; never use a mixture moment covariance."""
    if steps < 1 or tolerance <= 0:
        raise ValueError("Invalid Bures iteration budget")
    means = torch.zeros_like(target.means)
    for i, j, p in plan.edges(chunk):
        means.index_add_(0, j, p[:, None] * source.means[i])
    means /= target.weights[:, None]
    covariance = target.covariances.clone()
    residual = float("inf")
    repairs = 0
    for iteration in range(1, steps + 1):
        root, inv = roots(covariance, inverse=True)
        aggregate = torch.zeros_like(covariance)
        for i, j, p in plan.edges(chunk):
            middle = root[j] @ source.covariances[i] @ root[j]
            aggregate.index_add_(0, j, (p / target.weights[j])[:, None, None] * psd_sqrt(middle))
        updated = sym(inv @ aggregate @ aggregate @ inv)
        updated, count = repair_spd(updated)
        repairs += count
        denominator = torch.linalg.matrix_norm(covariance).clamp_min(torch.finfo(covariance.dtype).tiny)
        residual = float((torch.linalg.matrix_norm(updated - covariance) / denominator).max())
        covariance = updated
        if residual <= tolerance:
            break
    return GaussianSet(target.weights, means, covariance), {
        "inner_iterations": iteration, "inner_residual": residual,
        "inner_converged": residual <= tolerance, "covariance_repairs": repairs}
