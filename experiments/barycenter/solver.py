"""Variational avg/min LSOT barycenters with fixed slices and L-BFGS.

Hard projected ordering is recomputed at every evaluation. Gradients are
piecewise derivatives of the actual lifted Gaussian cost, including the
dependence of interval masses on trainable component weights. This is a
local numerical solver for a nonsmooth/nonconvex sampled objective.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lsot.gaussians import GMM, gaussian_pair_costs
from lsot.plans import _projected_plan_entries
from lsot.projections import project_gaussians


def on_device(gmm, device):
    return GMM(*(tensor.detach().to(device=device, dtype=torch.float64).clone()
                 for tensor in (gmm.weights, gmm.means, gmm.covariances)))


def detached_gmm(gmm):
    return GMM(*(tensor.detach().clone()
                 for tensor in (gmm.weights, gmm.means, gmm.covariances)))


def pair_cost_matrix(source, target):
    rows = torch.arange(source.count, device=source.weights.device).repeat_interleave(target.count)
    cols = torch.arange(target.count, device=source.weights.device).repeat(source.count)
    return gaussian_pair_costs(source, target, rows, cols).reshape(source.count, target.count)


class ParameterizedGMM(nn.Module):
    def __init__(self, initial, *, coordinate_scale=1., learn_weights=True,
                 minimum_std=1e-6):
        super().__init__()
        if coordinate_scale <= 0 or minimum_std <= 0:
            raise ValueError("coordinate_scale and minimum_std must be positive")
        self.scale = coordinate_scale
        self.minimum_diag = minimum_std/coordinate_scale
        self.means = nn.Parameter(initial.means.detach().clone()/self.scale)
        cholesky = torch.linalg.cholesky(initial.covariances.detach())/self.scale
        diagonal = torch.diagonal(cholesky, dim1=-2, dim2=-1)-self.minimum_diag
        if bool((diagonal <= 0).any()):
            raise ValueError("Initial covariance is too small for minimum_std")
        # Stable inverse softplus, including large image-coordinate entries.
        raw_diagonal = diagonal + torch.log(-torch.expm1(-diagonal))
        raw = torch.tril(cholesky, diagonal=-1) + torch.diag_embed(raw_diagonal)
        self.raw_cholesky = nn.Parameter(raw)
        self.logits = nn.Parameter(initial.weights.detach().log().clone(),
                                   requires_grad=learn_weights)

    def forward(self):
        raw_diag = torch.diagonal(self.raw_cholesky, dim1=-2, dim2=-1)
        cholesky = (torch.tril(self.raw_cholesky, diagonal=-1)
                    + torch.diag_embed(F.softplus(raw_diag)+self.minimum_diag))*self.scale
        weights = self.logits.softmax(0)
        return GMM(weights, self.means*self.scale, cholesky @ cholesky.transpose(-1,-2))


class BarycenterObjective:
    def __init__(self, inputs, weights, bank, kind, mode):
        if mode not in {"avg", "min"}:
            raise ValueError("mode must be avg or min")
        if len(inputs) != len(weights) or np.any(np.asarray(weights) < 0) or not np.isclose(np.sum(weights), 1):
            raise ValueError("Invalid barycenter input weights")
        self.kind, self.mode, self.bank = kind, mode, bank
        self.inputs = [(gmm, float(weight)) for gmm, weight in zip(inputs, weights) if weight > 0]
        with torch.no_grad():
            self.projected_inputs = [project_gaussians(gmm.means, gmm.covariances, bank, kind)
                                     for gmm, _ in self.inputs]

    def __call__(self, candidate):
        # Projection locations enter the hard cost only via ranks and fibers.
        # Do not differentiate integer ordering; do differentiate masses below.
        with torch.no_grad():
            y = project_gaussians(candidate.means, candidate.covariances, self.bank, self.kind)
        total = candidate.weights.new_zeros(())
        for (source, weight), x in zip(self.inputs, self.projected_inputs):
            directions, rows, cols, mass = _projected_plan_entries(
                x, y, source.weights, candidate.weights)
            costs = pair_cost_matrix(source, candidate)
            slice_costs = mass.new_zeros(self.bank.count).scatter_add(
                0, directions, mass*costs[rows, cols])
            value = slice_costs.mean() if self.mode == "avg" else slice_costs.min()
            total = total + weight*value
        return total


@dataclass
class OptimizationResult:
    gmm: GMM
    initial_objective: float
    objective: float
    iterations: int
    evaluations: int
    status: str
    history: list[dict]


def optimize_barycenter(inputs, weights, initial, bank, kind, mode, *,
                        steps=80, learning_rate=1., history_size=10,
                        tolerance_grad=1e-7, tolerance_change=1e-10,
                        coordinate_scale=1., learn_weights=True):
    if steps < 0 or learning_rate <= 0 or history_size < 1:
        raise ValueError("Invalid L-BFGS settings")
    model = ParameterizedGMM(initial, coordinate_scale=coordinate_scale,
                             learn_weights=learn_weights)
    objective = BarycenterObjective(inputs, weights, bank, kind, mode)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.LBFGS(parameters, lr=learning_rate, max_iter=1,
                                  max_eval=25, history_size=history_size,
                                  tolerance_grad=tolerance_grad,
                                  tolerance_change=tolerance_change,
                                  line_search_fn="strong_wolfe")
    with torch.no_grad():
        initial_cost = float(objective(model()))
    best_cost = initial_cost
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history = [{"iteration": 0, "objective_squared": initial_cost,
                "best_objective_squared": initial_cost, "evaluations": 0}]
    evaluations, iteration, stalls = 0, 0, 0
    status = "max_steps"
    gradient_norm = float("inf")

    def closure():
        nonlocal evaluations, best_cost, best_state, gradient_norm
        optimizer.zero_grad()
        value = objective(model())
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite barycenter loss")
        value.backward()
        gradients = torch.cat([p.grad.reshape(-1) for p in parameters if p.grad is not None])
        if not bool(torch.isfinite(gradients).all()):
            raise FloatingPointError("Nonfinite barycenter gradient")
        gradient_norm = float(gradients.detach().abs().max())
        evaluations += 1
        cost = float(value.detach())
        if cost < best_cost:
            best_cost = cost
            best_state = {name: entry.detach().clone() for name, entry in model.state_dict().items()}
        return value

    for iteration in range(1, steps+1):
        previous = best_cost
        try:
            optimizer.step(closure)
            with torch.no_grad():
                cost = float(objective(model()))
        except (FloatingPointError, ValueError, RuntimeError) as error:
            # A failed line search cannot erase a previously feasible candidate.
            status = f"numerical_stop: {type(error).__name__}: {error}"
            break
        if cost < best_cost:
            best_cost = cost
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        history.append({"iteration": iteration, "objective_squared": cost,
                        "best_objective_squared": best_cost, "evaluations": evaluations})
        if gradient_norm <= tolerance_grad:
            status = "gradient_tolerance"
            break
        if previous-best_cost <= tolerance_change*max(1., abs(previous)):
            stalls += 1
        else:
            stalls = 0
        if stalls >= 3:
            status = "stalled"
            break
    model.load_state_dict(best_state)
    result = detached_gmm(model())
    with torch.no_grad():
        best_cost = float(objective(result))
    if best_cost > initial_cost + 1e-9*max(1., abs(initial_cost)):
        raise RuntimeError("Best-iterate restoration failed")
    return OptimizationResult(result, initial_cost, best_cost, iteration,
                              evaluations, status, history)


def mw2_objective(inputs, weights, candidate):
    import ot
    value = 0.
    with torch.no_grad():
        for source, weight in zip(inputs, weights):
            if weight <= 0:
                continue
            costs = pair_cost_matrix(source, candidate).detach().cpu().numpy()
            a = source.weights.detach().cpu().numpy()
            b = candidate.weights.detach().cpu().numpy()
            value += float(weight)*float(ot.emd2(a, b, costs))
    return value


def mw2_distance_squared(source, target):
    return mw2_objective([source], [1.], target)
