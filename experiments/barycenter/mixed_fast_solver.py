"""Lifted tree initialization and mixed Adam / affine-invariant min-LSOT-B."""

from dataclasses import dataclass
import itertools
import time

import numpy as np
import torch

from lsot.gaussians import GMM
from lsot.plans import SparsePlan
from lsot.projections import sample_projection_bank
from .fast_geometry import TensorGMM, pair_costs, project_components
from .fast_solver import FastObjective, join_banks, plan_entries, subset_bank
from .reference import gaussian_barycenters


def bank(L, seed):
    return sample_projection_bank(2, L, kind='B', seed=seed)


@dataclass
class TupleCandidate:
    inputs: tuple
    indices: np.ndarray
    mass: np.ndarray
    gmm: GMM
    cost: float


def from_tuples(inputs, indices, mass, weights=None):
    keep = mass > 1e-12
    indices, mass = indices[keep], mass[keep]
    mass = mass / mass.sum()
    for j, g in enumerate(inputs):
        error = np.max(np.abs(np.bincount(indices[:, j], mass, minlength=g.count)-g.weights.numpy()))
        if error > 1e-9:
            raise ValueError(f'Tuple marginal error {error}')
    omega = np.ones(len(inputs)) if weights is None else np.asarray(weights, dtype=float)
    m, c, cost = gaussian_barycenters(
        np.stack([g.means.numpy()[indices[:, j]] for j, g in enumerate(inputs)], 1),
        np.stack([g.covariances.numpy()[indices[:, j]] for j, g in enumerate(inputs)], 1),
        omega/omega.sum(), iterations=30, fast=True,
    )
    return TupleCandidate(tuple(inputs), indices, mass, GMM.from_numpy(mass, m, c), float(mass @ cost))


def singleton(g):
    return TupleCandidate((g,), np.arange(g.count)[:, None], g.weights.numpy(), g, 0.)


def best_plans(a, b, directions, count):
    """Best distinct lifted B plans; exact proportional-fiber handling for ties."""
    with torch.no_grad():
        x, y = project_components(a, directions, 'B'), project_components(b, directions, 'B')
        ds, rows, cols, masses = plan_entries(x, y, a.weights, b.weights)
        costs = pair_costs(a, b)
        scores = masses.new_zeros(directions.count).scatter_add_(0, ds, masses*costs[rows, cols])
        seen, result = set(), []
        for ell in scores.argsort().tolist():
            keep = (ds == ell) & (masses > 1e-12)
            r, c, w = rows[keep], cols[keep], masses[keep]
            order = (r*b.count+c).argsort()
            signature = ((r*b.count+c)[order].numpy().tobytes(), np.round(w[order].numpy(), 12).tobytes())
            if signature in seen:
                continue
            seen.add(signature)
            p = SparsePlan(r, c, w, (a.count, b.count))
            if p.marginal_error(a, b) > 1e-9:
                raise ValueError('Invalid lifted plan')
            result.append(p)
            if len(result) == count:
                break
    return result


def join(a, b, plan, weights=None):
    ix = np.column_stack([a.indices[plan.rows.numpy()], b.indices[plan.cols.numpy()]])
    return from_tuples(a.inputs+b.inputs, ix, plan.mass.numpy(), weights)


def balanced_search(inputs, *, L, banks, width, root_width, weights=None):
    """Rank feasible tuple plans built from lifted B projections."""
    omega = np.ones(len(inputs)) if weights is None else np.asarray(weights, dtype=float)
    best = None
    evaluated = 0
    if len(inputs) == 2:
        a, b = map(singleton, inputs)
        for seed in range(banks):
            for plan in best_plans(a.gmm, b.gmm, bank(L, 3100000 + seed), width):
                candidate = join(a, b, plan, omega)
                evaluated += 1
                if best is None or candidate.cost < best.cost:
                    best = candidate
        return best, evaluated
    if len(inputs) != 4:
        raise ValueError('Lifted tree needs two or four active inputs')
    for order in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
        a, b, c, d = [singleton(inputs[i]) for i in order]
        w = omega[list(order)]
        for seed in range(banks):
            directions = bank(L, 3100000 + seed)
            left = [join(a, b, p, w[:2]) for p in best_plans(a.gmm, b.gmm, directions, width)]
            right = [join(c, d, p, w[2:]) for p in best_plans(c.gmm, d.gmm, directions, width)]
            for ab, cd in itertools.product(left, right):
                for plan in best_plans(ab.gmm, cd.gmm, directions, root_width):
                    candidate = join(ab, cd, plan, w)
                    evaluated += 1
                    if best is None or candidate.cost < best.cost:
                        best = candidate
    return best, evaluated


def pad(g, budget, seed):
    a, m, c = [v.numpy().copy() for v in (g.weights, g.means, g.covariances)]
    keep = a > 1e-12
    a, m, c = a[keep], m[keep], c[keep]
    a /= a.sum()
    if len(a) > budget:
        raise ValueError('Tree exceeds component budget; no silent reduction')
    rng = np.random.default_rng(seed)
    while len(a) < budget:
        i = int(a.argmax())
        shift = .01*np.linalg.cholesky(c[i]) @ rng.normal(size=2)
        a[i] *= .5
        a = np.r_[a, a[i]]
        m = np.vstack([m, m[i]+shift]); m[i] -= shift
        c = np.concatenate([c, c[i:i+1]])
    return GMM.from_numpy(a, m, c)


def covariance_step(covariance, gradient, rate):
    """Affine-invariant exponential step with a spectral step-size safeguard."""
    s = covariance.detach()
    g = (gradient + gradient.transpose(-1, -2)) / 2
    eig,vec=torch.linalg.eigh(s)
    root=(vec*eig.sqrt()[:,None,:]) @ vec.transpose(-1,-2)
    h=root @ g @ root
    size=torch.linalg.eigvalsh(h).abs().amax(-1).clamp_min(1e-30)
    eta=torch.minimum(torch.full_like(size,rate),4./size)
    result=root @ torch.matrix_exp(-eta[:,None,None]*h) @ root
    return (result + result.transpose(-1, -2)) / 2


def solve_mixed(inputs, initial, cfg, weights=None):
    """Fit min-LSOT-B, selecting the iterate on an independent fixed bank."""
    started = time.perf_counter()
    means = torch.nn.Parameter(initial.means.clone())
    logits = torch.nn.Parameter(initial.weights.log().clone())
    covariance = initial.covariances.clone()
    optimizer = torch.optim.Adam([
        dict(params=[logits], lr=cfg['mass']),
        dict(params=[means], lr=cfg['mean']),
    ])
    L = cfg['L']
    directions = bank(L, 0)
    omega = [1/len(inputs)] * len(inputs) if weights is None else weights
    train = FastObjective(inputs, omega, directions, 'B', 'min')
    validation = FastObjective(inputs, omega, bank(512, 1000003), 'B', 'min')
    best, state, selected, history = float('inf'), None, 0, []

    def candidate():
        return TensorGMM(logits.softmax(0), means, covariance)

    def checkpoint(step):
        nonlocal best, state, selected
        with torch.no_grad():
            g = candidate()
            value = float(validation(g))
            if not np.isfinite(value) or torch.linalg.eigvalsh(covariance).min() <= 0:
                raise FloatingPointError('Invalid covariance or loss')
            if value < best:
                best, selected = value, step
                state = tuple(t.detach().clone() for t in (g.weights, g.means, g.covariances))
            history.append(dict(step=step, validation=value, best_validation=best, selected_step=selected))

    checkpoint(0)
    for step in range(1, cfg['steps'] + 1):
        if cfg['sampling'] == 'partial' and step > 1 and (step - 1) % 10 == 0:
            winners = torch.cat(train.last_winners).unique().tolist()
            keep = list(dict.fromkeys(winners + list(range(L))))[:L - L // 4]
            directions = join_banks(subset_bank(directions, torch.tensor(keep)),
                                   bank(L // 4, 10000019 + step))
            train.set_bank(directions)
        optimizer.zero_grad(set_to_none=True)
        covariance = covariance.detach().requires_grad_(True)
        g = candidate()
        loss = train(g)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(covariance.grad).all():
            raise FloatingPointError(f'Nonfinite gradient/loss at step {step}')
        with torch.no_grad():
            mass = g.weights.detach().clamp_min(1e-10)
            new_cov = covariance_step(covariance, covariance.grad / mass[:, None, None], cfg['rate'])
        optimizer.step()
        covariance = new_cov
        if step % 10 == 0 or step == cfg['steps']:
            checkpoint(step)
    return GMM(*state), history, dict(
        optimizer_ms=1000 * (time.perf_counter() - started), selected_step=selected,
        validation_objective=best, status='max_steps', error='',
    )
