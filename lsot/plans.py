"""Proportional lifting of monotone 1D plans; no ground-cost masking.

The fast path batches projections without ties. Exact ties use the original
fiber-wise proportional rule, including when a self-plan is not diagonal.
"""
from dataclasses import dataclass
import numpy as np
import torch

from param_proj.projections import project_gaussians
from .gaussians import gaussian_pair_costs


@dataclass(frozen=True)
class SparsePlan:
    rows: torch.Tensor
    cols: torch.Tensor
    mass: torch.Tensor
    shape: tuple

    @classmethod
    def from_entries(cls, rows, cols, mass, shape):
        sparse = torch.sparse_coo_tensor(torch.stack((rows, cols)), mass, shape).coalesce()
        keep = sparse.values() > 0
        indices = sparse.indices()[:, keep]
        return cls(indices[0], indices[1], sparse.values()[keep], tuple(shape))

    @classmethod
    def from_dense(cls, matrix):
        rows, cols = torch.where(matrix > 0)
        return cls.from_entries(rows, cols, matrix[rows, cols], matrix.shape)

    @property
    def nnz(self):
        return self.mass.numel()

    def dense(self):
        return torch.sparse_coo_tensor(torch.stack((self.rows, self.cols)), self.mass, self.shape).to_dense()

    def marginals(self):
        a = self.mass.new_zeros(self.shape[0]).scatter_add_(0, self.rows, self.mass)
        b = self.mass.new_zeros(self.shape[1]).scatter_add_(0, self.cols, self.mass)
        return a, b

    def marginal_error(self, source, target):
        a, b = self.marginals()
        return torch.maximum((a - source.weights).abs().sum(), (b - target.weights).abs().sum()).item()

    def cost(self, source, target):
        return torch.dot(self.mass, gaussian_pair_costs(source, target, self.rows, self.cols))


def _monotone_entries(a, b):
    """Batched cumulative-interval intersections; a=(L,K0), b=(L,K1)."""
    ca, cb = a.cumsum(-1), b.cumsum(-1)
    # Probability intervals end at exactly one despite floating point roundoff.
    ca = torch.cat((ca[:, :-1].clamp(max=1), torch.ones_like(ca[:, -1:])), dim=1)
    cb = torch.cat((cb[:, :-1].clamp(max=1), torch.ones_like(cb[:, -1:])), dim=1)
    edges = torch.cat((a.new_zeros((len(a), 1)), ca, cb), dim=1).sort(dim=1).values
    left, right = edges[:, :-1], edges[:, 1:]
    # right=True chooses the atom immediately to the right of each left edge.
    i = torch.searchsorted(ca.contiguous(), left.contiguous(), right=True).clamp(max=a.shape[1] - 1)
    j = torch.searchsorted(cb.contiguous(), left.contiguous(), right=True).clamp(max=b.shape[1] - 1)
    return i, j, right - left


def lift_projection(a_values, b_values, a, b):
    """One projection with exact fiber aggregation and proportional lifting."""
    av, ia = torch.sort(a_values, stable=True)
    bv, ib = torch.sort(b_values, stable=True)
    _, ag = torch.unique_consecutive(av, return_inverse=True)
    _, bg = torch.unique_consecutive(bv, return_inverse=True)
    aa, bb = a[ia], b[ib]
    am = a.new_zeros(int(ag[-1]) + 1).scatter_add_(0, ag, aa)
    bm = b.new_zeros(int(bg[-1]) + 1).scatter_add_(0, bg, bb)
    fi, fj, mass = _monotone_entries(am[None], bm[None])
    rows, cols, values = [], [], []
    for r, t, value in zip(fi[0], fj[0], mass[0]):
        if value <= 0:
            continue
        si, tj = torch.where(ag == r)[0], torch.where(bg == t)[0]
        block = value * (aa[si] / am[r])[:, None] * (bb[tj] / bm[t])[None, :]
        rows.append(ia[si].repeat_interleave(len(tj)))
        cols.append(ib[tj].repeat(len(si)))
        values.append(block.flatten())
    return SparsePlan.from_entries(torch.cat(rows), torch.cat(cols), torch.cat(values), (len(a), len(b)))


def average_projected_plans(x, y, a, b):
    """Average L lifts from locations (K0,L)/(K1,L), preserving exact ties."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.shape[1] == 0:
        raise ValueError("expected projected locations (K0,L) and (K1,L), L>0")
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("projected locations must be finite")
    count = x.shape[1]
    xs, ix = torch.sort(x.T, dim=1, stable=True)
    ys, iy = torch.sort(y.T, dim=1, stable=True)
    ties = (xs[:, 1:] == xs[:, :-1]).any(1) | (ys[:, 1:] == ys[:, :-1]).any(1)
    rows, cols, values = [], [], []
    good = ~ties
    if bool(good.any()):
        i, j, mass = _monotone_entries(a[ix[good]], b[iy[good]])
        keep = mass > 0
        rows.append(torch.gather(ix[good], 1, i)[keep])
        cols.append(torch.gather(iy[good], 1, j)[keep])
        values.append(mass[keep] / count)
    for ell in torch.where(ties)[0].tolist():
        plan = lift_projection(x[:, ell], y[:, ell], a, b)
        rows.append(plan.rows)
        cols.append(plan.cols)
        values.append(plan.mass / count)
    return SparsePlan.from_entries(torch.cat(rows), torch.cat(cols), torch.cat(values), (len(a), len(b)))


def average_lsot(source, target, bank, kind):
    """Return the averaged lift; evaluate its true Gaussian cost separately."""
    x = project_gaussians(source.means, source.covariances, bank, kind)
    y = project_gaussians(target.means, target.covariances, bank, kind)
    return average_projected_plans(x, y, source.weights, target.weights)


def solve_mw2(source, target):
    """Run the preserved gmmot.GW2 baseline (SciPy/POT on CPU).

    Its legacy name GW2 means MW2 here; the returned scalar is SQUARED cost.
    Transfers occur inside this function and belong to its measured runtime.
    """
    from gmmot import GW2
    arrays = [t.detach().cpu().numpy() for t in
              (source.weights, target.weights, source.means, target.means,
               source.covariances, target.covariances)]
    matrix, squared_cost = GW2(*arrays)
    matrix = torch.as_tensor(np.real_if_close(matrix), dtype=source.weights.dtype,
                             device=source.weights.device)
    return SparsePlan.from_dense(matrix), float(np.real_if_close(squared_cost))
