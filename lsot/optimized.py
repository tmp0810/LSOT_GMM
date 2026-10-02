"""Stein-gradient search for a low-cost lifted Gaussian component plan.

Adapted from the optimization principle in Chapel--Tavenard--Vaiter,
Differentiable Generalized Sliced Wasserstein Plans (NeurIPS 2025),
https://github.com/rtavenar/dgswp/blob/main/dgswp/losses.py.

Their sampled-point matching is replaced here by weighted 1D OT, proportional
component lifting, and the Gaussian W2-squared ground cost. We optimize the
smoothed objective, while returning one actual lifted plan at the best
evaluated projection (not the smoothed/averaged plan).
"""
from dataclasses import dataclass

import torch

from distribution_proj import BusemannBank, Busemann1DBank
from param_proj import ProjectionBank

from .plans import SparsePlan, minimum_projected_plan
from .projections import project_gaussians, PROJECTION_KINDS


@dataclass(frozen=True)
class OptimizedLSOTResult:
    plan: SparsePlan
    cost_squared: torch.Tensor
    initial_projection_index: int
    initial_cost_squared: torch.Tensor
    initial_projection_costs: torch.Tensor
    best_iteration: int
    cost_history: torch.Tensor
    best_bank: object


def _initial_parameters(bank, index, kind):
    """Unconstrained coordinates whose decoded first value is the given ray."""
    if kind in {"Mix", "SMix"}:
        parameters = [bank.theta[index], bank.psi[index]]
        if kind == "Mix":
            parameters.append(bank.matrices[index])
    elif kind == "B":
        matrix = bank.tangent_matrices[index]
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        factor = (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.T
        parameters = [bank.mean_directions[index], factor]
    elif kind == "B1D":
        a = bank.mean_speeds[index].clamp(-1 + 1e-12, 1 - 1e-12)
        parameters = [bank.theta[index], torch.atanh(a).reshape(1)]
    else:
        raise ValueError(f"unknown projection kind: {kind}")
    return torch.cat([p.reshape(-1) for p in parameters])


def _decode(parameters, kind, dimension):
    """Decode (N,q) raw coordinates into N valid, original-family rays."""
    n = len(parameters)
    d = dimension
    tiny = torch.finfo(parameters.dtype).tiny

    def unit(t, dims):
        return t / torch.linalg.vector_norm(t, dim=dims, keepdim=True).clamp_min(tiny)

    theta = unit(parameters[:, :d], -1)
    if kind in {"Mix", "SMix"}:
        psi = unit(parameters[:, d:d + 2], -1)
        if kind == "Mix":
            matrix = parameters[:, d + 2:].reshape(n, d, d)
            matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
            matrix = unit(matrix, (-2, -1))
        else:
            matrix = parameters.new_zeros((n, d, d))  # unused by SMix
        return ProjectionBank(theta, psi, matrix)
    if kind == "B":
        factor = parameters[:, d:].reshape(n, d, d)
        tangent = factor @ factor.transpose(-1, -2)
        norm = (parameters[:, :d].square().sum(-1) + tangent.square().sum((-2, -1))).sqrt().clamp_min(tiny)
        return BusemannBank(parameters[:, :d] / norm[:, None], tangent / norm[:, None, None])
    if kind == "B1D":
        a = parameters[:, d].tanh()
        b = (1 - a.square()).clamp_min(0).sqrt()
        return Busemann1DBank(theta, a, b)
    raise ValueError(f"unknown projection kind: {kind}")


def _evaluate(source, target, raw, kind):
    bank = _decode(raw, kind, source.dimension)
    x = project_gaussians(source.means, source.covariances, bank, kind)
    y = project_gaussians(target.means, target.covariances, bank, kind)
    return minimum_projected_plan(x, y, source, target), bank


def optimized_minimum_lsot(source, target, bank, kind, *, steps=20, samples=8,
                           epsilon=0.05, learning_rate=0.05, seed=0,
                           max_gradient_norm=10.0):
    """Optimize a smoothed projection objective; keep the best true-cost plan.

    First choose the best of the *same* initial L bank used by minimum_lsot.
    At every step evaluate the central direction and N Gaussian perturbations;
    (h(u+eps*z)-h(u))*z/(N*eps) estimates the smoothed gradient with a
    control variate. Track the best original (unsmoothed) plan seen, so the
    result cannot be worse than finite-bank min-LSOT in ground cost.

    This is a heuristic search, not an exact global minimizer and not a
    differentiable computation graph through the 1D sorting/lifting operation.
    """
    if kind not in PROJECTION_KINDS:
        raise ValueError(f"kind must be one of {PROJECTION_KINDS}")
    if steps < 0 or samples < 1 or epsilon <= 0 or learning_rate <= 0 or max_gradient_norm <= 0:
        raise ValueError("invalid Stein optimizer settings")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    x = project_gaussians(source.means, source.covariances, bank, kind)
    y = project_gaussians(target.means, target.covariances, bank, kind)
    initial = minimum_projected_plan(x, y, source, target)
    u = _initial_parameters(bank, initial.projection_index, kind)
    best_plan, best_cost = initial.plan, initial.cost_squared
    best_bank = _decode(u[None], kind, source.dimension)
    best_iteration = 0
    history = [best_cost]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for iteration in range(1, steps + 1):
        noise = torch.randn(samples, u.numel(), generator=generator,
                            dtype=torch.float64).to(device=u.device, dtype=u.dtype)
        candidates = torch.cat([u[None], u[None] + epsilon * noise], dim=0)
        result, candidate_bank = _evaluate(source, target, candidates, kind)
        costs = result.projection_costs
        h0 = costs[0]
        gradient = ((costs[1:] - h0)[:, None] * noise).mean(0) / epsilon
        gradient = gradient * (max_gradient_norm /
                               gradient.norm().clamp_min(torch.finfo(u.dtype).tiny)).clamp(max=1)
        if bool(result.cost_squared < best_cost):
            best_cost, best_plan = result.cost_squared, result.plan
            # Return one selected projection rather than all earlier candidates.
            best_bank = type(candidate_bank)(*(tensor[result.projection_index:result.projection_index + 1]
                                               for tensor in vars(candidate_bank).values()))
            best_iteration = iteration
        u = u - learning_rate * gradient
        if not bool(torch.isfinite(u).all()):
            raise RuntimeError("Stein projection optimizer produced nonfinite parameters")
        history.append(best_cost)

    final_result, final_bank = _evaluate(source, target, u[None], kind)
    if bool(final_result.cost_squared < best_cost):
        best_cost, best_plan, best_bank = final_result.cost_squared, final_result.plan, final_bank
        best_iteration = steps
    history.append(best_cost)
    return OptimizedLSOTResult(best_plan, best_cost, initial.projection_index,
                               initial.cost_squared, initial.projection_costs,
                               best_iteration, torch.stack(history), best_bank)
