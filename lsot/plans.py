from dataclasses import dataclass
import numpy as np
import torch

from .projections import project_gaussians
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
    ca, cb = a.cumsum(-1), b.cumsum(-1)
    ca = torch.cat((ca[:, :-1].clamp(max=1), torch.ones_like(ca[:, -1:])), dim=1)
    cb = torch.cat((cb[:, :-1].clamp(max=1), torch.ones_like(cb[:, -1:])), dim=1)
    edges = torch.cat((a.new_zeros((len(a), 1)), ca, cb), dim=1).sort(dim=1).values
    left, right = edges[:, :-1], edges[:, 1:]
    i = torch.searchsorted(ca.contiguous(), left.contiguous(), right=True).clamp(max=a.shape[1] - 1)
    j = torch.searchsorted(cb.contiguous(), left.contiguous(), right=True).clamp(max=b.shape[1] - 1)
    return i, j, right - left


def lift_projection(a_values, b_values, a, b):
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


def _projected_plan_entries(x, y, a, b):
    xs, ix = torch.sort(x.T, dim=1, stable=True)
    ys, iy = torch.sort(y.T, dim=1, stable=True)
    ties = (xs[:, 1:] == xs[:, :-1]).any(1) | (ys[:, 1:] == ys[:, :-1]).any(1)
    directions, rows, cols, values = [], [], [], []
    good = ~ties
    if bool(good.any()):
        i, j, mass = _monotone_entries(a[ix[good]], b[iy[good]])
        keep = mass > 0
        directions.append(torch.where(good)[0][:, None].expand_as(mass)[keep])
        rows.append(torch.gather(ix[good], 1, i)[keep])
        cols.append(torch.gather(iy[good], 1, j)[keep])
        values.append(mass[keep])
    for ell in torch.where(ties)[0].tolist():
        plan = lift_projection(x[:, ell], y[:, ell], a, b)
        directions.append(torch.full_like(plan.rows, ell))
        rows.append(plan.rows)
        cols.append(plan.cols)
        values.append(plan.mass)
    return tuple(torch.cat(parts) for parts in (directions, rows, cols, values))


def average_projected_plans(x, y, a, b):
    _, rows, cols, mass = _projected_plan_entries(x, y, a, b)
    return SparsePlan.from_entries(rows, cols, mass / x.shape[1], (len(a), len(b)))


def average_lsot(source, target, bank, kind):
    x = project_gaussians(source.means, source.covariances, bank, kind)
    y = project_gaussians(target.means, target.covariances, bank, kind)
    return average_projected_plans(x, y, source.weights, target.weights)


@dataclass(frozen=True)
class MinimumLSOTResult:
    plan: SparsePlan
    cost_squared: torch.Tensor
    projection_index: int
    projection_costs: torch.Tensor


def minimum_projected_plan(x, y, source, target):
    directions, rows, cols, mass = _projected_plan_entries(x, y, source.weights, target.weights)
    pairs, inverse = torch.unique(rows * target.count + cols, return_inverse=True)
    pair_costs = gaussian_pair_costs(source, target, pairs // target.count, pairs % target.count)
    costs = mass.new_zeros(x.shape[1]).scatter_add_(0, directions, mass * pair_costs[inverse])
    selected = int(torch.argmin(costs).item())
    keep = directions == selected
    plan = SparsePlan.from_entries(rows[keep], cols[keep], mass[keep], (source.count, target.count))
    return MinimumLSOTResult(plan, costs[selected], selected, costs)


def minimum_lsot(source, target, bank, kind):
    x = project_gaussians(source.means, source.covariances, bank, kind)
    y = project_gaussians(target.means, target.covariances, bank, kind)
    return minimum_projected_plan(x, y, source, target)


@dataclass(frozen=True)
class TopKLSOTResult:
    plan: SparsePlan
    cost_squared: torch.Tensor
    projection_indices: torch.Tensor
    projection_costs: torch.Tensor


def topk_projected_plan(x, y, source, target, k):
    if type(k) is not int or k < 1 or k > x.shape[1]:
        raise ValueError("k must be an integer between 1 and the bank size")
    directions, rows, cols, mass = _projected_plan_entries(x, y, source.weights, target.weights)
    pairs, inverse = torch.unique(rows * target.count + cols, return_inverse=True)
    pair_costs = gaussian_pair_costs(source, target, pairs // target.count, pairs % target.count)
    costs = mass.new_zeros(x.shape[1]).scatter_add_(0, directions, mass * pair_costs[inverse])
    selected = torch.argsort(costs, stable=True)[:k]
    keep = torch.isin(directions, selected)
    plan = SparsePlan.from_entries(rows[keep], cols[keep], mass[keep] / k,
                                   (source.count, target.count))
    return TopKLSOTResult(plan, costs[selected].mean(), selected, costs)


def topk_lsot(source, target, bank, kind, k):
    x = project_gaussians(source.means, source.covariances, bank, kind)
    y = project_gaussians(target.means, target.covariances, bank, kind)
    return topk_projected_plan(x, y, source, target, k)


def solve_mw2(source, target):
    from gmmot import GW2
    arrays = [t.detach().cpu().numpy() for t in
              (source.weights, target.weights, source.means, target.means,
               source.covariances, target.covariances)]
    matrix, squared_cost = GW2(*arrays)
    matrix = torch.as_tensor(np.real_if_close(matrix), dtype=source.weights.dtype,
                             device=source.weights.device)
    return SparsePlan.from_dense(matrix), float(np.real_if_close(squared_cost))
