"""MW2 multi-marginal LP, retaining the notebook's N=10 Gaussian iteration.

Vectorized SPD square roots and sparse LP constraints implement the same
finite problem as create_cost_matrix_from_gmm / solveMMOT in gmmot.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from lsot.gaussians import GMM


def arrays(gmm):
    return tuple(t.detach().cpu().numpy() for t in
                 (gmm.weights, gmm.means, gmm.covariances))


def spd_sqrt(matrix):
    values, vectors = np.linalg.eigh(.5*(matrix + np.swapaxes(matrix, -1, -2)))
    if np.any(values < -1e-10):
        raise ValueError("Matrix square root requires a positive semidefinite input")
    return (vectors*np.sqrt(np.maximum(values, 0))[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def gaussian_barycenters(means, covariances, weights, *, iterations=10, roots=None):
    """Batched notebook GaussianBarycenterW2; axes are (tuple,input,d,...)."""
    means, covariances = np.asarray(means), np.asarray(covariances)
    weights = np.asarray(weights, dtype=float)
    if iterations < 1 or means.ndim != 3 or len(weights) != means.shape[1]:
        raise ValueError("Invalid Gaussian barycenter shapes or iteration count")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Barycenter weights must be nonnegative and sum to one")
    count, _, dimension = means.shape
    mean = np.einsum("j,tjd->td", weights, means)
    covariance = np.broadcast_to(np.eye(dimension), (count, dimension, dimension)).copy()
    for _ in range(iterations):
        root = spd_sqrt(covariance)
        middle = root[:, None] @ covariances @ root[:, None]
        covariance = np.einsum("j,tjab->tab", weights, spd_sqrt(middle))
    covariance = .5*(covariance + np.swapaxes(covariance, -1, -2))
    roots = spd_sqrt(covariances) if roots is None else roots
    middle = roots @ covariance[:, None] @ roots
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh(.5*(middle+np.swapaxes(middle,-1,-2))), 0)).sum(-1)
    trace = np.trace(covariances, axis1=-2, axis2=-1) + np.trace(covariance, axis1=-2, axis2=-1)[:, None]
    costs = ((means-mean[:, None])**2).sum(-1) + trace - 2*cross
    return mean, covariance, np.maximum(costs, 0) @ weights


@dataclass
class ReferenceResult:
    gmm: GMM
    tuples: np.ndarray
    masses: np.ndarray
    ground_costs: np.ndarray
    cost_squared: float
    marginal_error: float


def mw2_barycenter(inputs, weights, *, iterations=10):
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(inputs) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid input weights")
    active = np.flatnonzero(weights > 0)
    if len(active) == 1:
        source = inputs[active[0]]
        w, _, _ = arrays(source)
        return ReferenceResult(source, np.arange(source.count)[:, None], w, np.zeros(source.count), 0., 0.)
    selected = [inputs[j] for j in active]
    probabilities, means, covariances = zip(*(arrays(gmm) for gmm in selected))
    shape = tuple(len(p) for p in probabilities)
    tuples = np.indices(shape).reshape(len(shape), -1).T
    tuple_means = np.stack([means[j][tuples[:, j]] for j in range(len(shape))], axis=1)
    tuple_covariances = np.stack([covariances[j][tuples[:, j]] for j in range(len(shape))], axis=1)
    centers, covs, costs = gaussian_barycenters(tuple_means, tuple_covariances,
                                               weights[active], iterations=iterations)
    count = len(tuples)
    offsets = np.cumsum((0,) + shape[:-1])
    rows = np.concatenate([tuples[:, j]+offsets[j] for j in range(len(shape))])
    cols = np.tile(np.arange(count), len(shape))
    constraints = coo_matrix((np.ones(len(rows)), (rows, cols)),
                             shape=(sum(shape), count)).tocsr()
    rhs = np.concatenate(probabilities)
    result = linprog(costs, A_eq=constraints, b_eq=rhs, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"MW2 multi-marginal LP failed: {result.message}")
    masses = np.maximum(result.x, 0)
    error = float(np.max(np.abs(constraints @ masses-rhs)))
    if error > 1e-7:
        raise RuntimeError(f"MW2 multi-marginal marginal error: {error}")
    keep = masses > 0
    gmm = GMM.from_numpy(masses[keep], centers[keep], covs[keep])
    return ReferenceResult(gmm, tuples[keep], masses[keep], costs[keep],
                           float(masses @ costs), error)


def shared_initialization(inputs, weights, component_budget, *, iterations=10, seed=0):
    """A method-independent common-quantile initialization; never uses MW2 LP.

    Ordering uses a fixed spatial direction, rather than a LSOT family.
    Component Gaussian barycenters are evaluated only on positive intervals.
    """
    if component_budget < 1:
        raise ValueError("component_budget must be positive")
    weights = np.asarray(weights, dtype=float)
    if len(inputs) != len(weights) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid barycenter input weights")
    active = np.flatnonzero(weights > 0)
    selected = [inputs[j] for j in active]
    probabilities, means, covariances = zip(*(arrays(gmm) for gmm in selected))
    direction = np.arange(1, inputs[0].dimension+1, dtype=float)**.5
    order = [np.argsort(m @ direction, kind="stable") for m in means]
    cumulative = [np.r_[0., np.cumsum(p[o])[:-1], 1.] for p, o in zip(probabilities, order)]
    edges = np.unique(np.concatenate(cumulative))
    left, right = edges[:-1], edges[1:]
    keep = right-left > 1e-12
    left, right = left[keep], right[keep]
    masses = right-left
    midpoints = .5*(left+right)
    tuples = np.column_stack([o[np.searchsorted(c[1:], midpoints, side="right").clip(max=len(o)-1)]
                              for o, c in zip(order, cumulative)])
    tm = np.stack([means[j][tuples[:, j]] for j in range(len(selected))], axis=1)
    tc = np.stack([covariances[j][tuples[:, j]] for j in range(len(selected))], axis=1)
    centers, covs, _ = gaussian_barycenters(tm, tc, np.asarray(weights)[active], iterations=iterations)
    rng = np.random.default_rng(seed)
    # Moment-preserving merges allow an explicitly smaller component budget.
    while len(masses) > component_budget:
        small = int(np.argmin(masses))
        distances = ((centers-centers[small])**2).sum(-1)
        distances[small] = np.inf
        other = int(np.argmin(distances))
        total = masses[small]+masses[other]
        center = (masses[small]*centers[small]+masses[other]*centers[other])/total
        covariance = sum(masses[k]*(covs[k]+np.outer(centers[k]-center, centers[k]-center))
                         for k in (small, other))/total
        masses[other], centers[other], covs[other] = total, center, covariance
        masses, centers, covs = (np.delete(a, small, axis=0) for a in (masses, centers, covs))
    # Split components with opposite small perturbations to avoid duplicate atoms.
    while len(masses) < component_budget:
        index = int(np.argmax(masses))
        perturb = .01*np.linalg.cholesky(covs[index]) @ rng.normal(size=centers.shape[1])
        original = centers[index].copy()
        masses[index] *= .5
        centers[index] = original-perturb
        masses = np.r_[masses, masses[index]]
        centers = np.concatenate((centers, (original+perturb)[None]))
        covs = np.concatenate((covs, covs[index:index+1]))
    return GMM.from_numpy(masses, centers, covs)
