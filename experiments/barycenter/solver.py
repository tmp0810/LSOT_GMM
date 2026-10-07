"""Variational avg/min LSOT barycenters with fixed slices and Adam/L-BFGS.

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
    gradient_inf: float
    gradient_tolerance_met: bool
    starts: int
    restarts: int
    start_summaries: list[dict]
    gradient_evaluations: int


def optimize_barycenter(inputs, weights, initial, bank, kind, mode, *,
                        steps=80, learning_rate=1., history_size=10,
                        tolerance_grad=1e-7, tolerance_change=1e-10,
                        coordinate_scale=1., learn_weights=True,
                        initializations=None, stall_patience=8, max_restarts=2,
                        warmup_steps=0, optimizer_kind="lbfgs", adam_learning_rate=.01):
    """Adam or safeguarded L-BFGS on the unchanged hard lifted objective.

    ``steps`` is a per-start budget. Every method can receive the same saved
    spatial starts. A rejected line-search trial may be the best candidate:
    accepting it explicitly clears the old curvature history. Gradients for
    stopping are always reevaluated at the retained candidate, never taken
    from the last trial. ``stalled`` is not a convergence certificate.

    Optional Adam warmup is explicit and defaults to zero; it can cross hard
    ordering boundaries before L-BFGS refinement. All selections still use
    the exact hard objective, with no entropy or soft-sorting surrogate.
    ``optimizer_kind='adam'`` uses only ``steps`` Adam updates per start,
    restores the best evaluated candidate, and never constructs L-BFGS.
    """
    if (steps < 0 or learning_rate <= 0 or history_size < 1 or stall_patience < 1
            or max_restarts < 0 or warmup_steps < 0 or tolerance_grad < 0
            or tolerance_change < 0 or adam_learning_rate <= 0
            or optimizer_kind not in {"adam", "lbfgs"}
            or (optimizer_kind == "adam" and warmup_steps != 0)):
        raise ValueError("Invalid optimizer settings")
    starting_points = list(initializations) if initializations is not None else [initial]
    if not starting_points or any(g.count != initial.count or g.dimension != initial.dimension
                                  for g in starting_points):
        raise ValueError("All starts must have the same component budget and dimension")
    objective = BarycenterObjective(inputs, weights, bank, kind, mode)
    model = ParameterizedGMM(initial, coordinate_scale=coordinate_scale,
                             learn_weights=learn_weights)
    # Uniform objective scaling preserves the minimizers and makes image and
    # synthetic stopping tolerances comparable in normalized spatial units.
    loss_scale = coordinate_scale**2
    with torch.no_grad():
        initial_cost = float(objective(model()))
    if not np.isfinite(initial_cost):
        raise FloatingPointError("Nonfinite initial barycenter loss")
    best_cost, best_gradient = initial_cost, float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_start = 0
    history, summaries = [], []
    evaluations, gradient_evaluations, iteration, restarts = 1, 0, 0, 0
    local_cost, local_state = initial_cost, best_state
    parameters = [p for p in model.parameters() if p.requires_grad]
    current_start = 0

    def snapshot():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def evaluate():
        nonlocal evaluations, gradient_evaluations, best_cost, best_state, best_start, best_gradient
        nonlocal local_cost, local_state
        model.zero_grad(set_to_none=True)
        value = objective(model())/loss_scale
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite barycenter loss")
        value.backward()
        gradients = torch.cat([p.grad.reshape(-1) for p in parameters if p.grad is not None])
        if not bool(torch.isfinite(gradients).all()):
            raise FloatingPointError("Nonfinite barycenter gradient")
        norm = float(gradients.detach().abs().max())
        cost = float(value.detach())*loss_scale
        evaluations += 1
        gradient_evaluations += 1
        if cost < local_cost:
            local_cost, local_state = cost, snapshot()
        if cost < best_cost:
            best_cost, best_state, best_start, best_gradient = cost, snapshot(), current_start, norm
        elif cost == best_cost:
            best_gradient = norm
        return value, cost, norm

    def new_lbfgs():
        return torch.optim.LBFGS(parameters, lr=learning_rate, max_iter=1,
                                 max_eval=25, history_size=history_size,
                                 tolerance_grad=tolerance_grad,
                                 # With max_iter=1, our outer monitor handles
                                 # small changes. PyTorch also compares g^T d
                                 # against tolerance_change; 1e-10 otherwise
                                 # prevents reaching a 1e-7 gradient even on
                                 # a smooth single-Gaussian objective.
                                 tolerance_change=0.,
                                 line_search_fn="strong_wolfe")

    def record(cost, gradient, step, phase):
        history.append({"iteration": iteration, "start": current_start, "step": step,
                        "phase": phase, "objective_squared": cost,
                        "best_objective_squared": best_cost,
                        "gradient_inf": gradient, "best_gradient_inf": best_gradient,
                        "evaluations": evaluations, "gradient_evaluations": gradient_evaluations,
                        "restarts": restarts})

    def gradient_backtracking():
        """Try full and block gradient steps before abandoning a start."""
        nonlocal evaluations, local_cost, local_state, best_cost, best_state, best_start
        model.load_state_dict(local_state)
        evaluate()
        origin, gradients = snapshot(), {n: p.grad.detach().clone()
                                         for n, p in model.named_parameters() if p.requires_grad}
        before = local_cost
        for block in (None, "means", "raw_cholesky", "logits"):
            if block == "logits" and not learn_weights:
                continue
            for rate in (1., .1, .01, .001, .0001, .00001, .000001):
                model.load_state_dict(origin)
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in gradients and (block is None or name == block):
                            parameter.add_(gradients[name], alpha=-learning_rate*rate)
                    cost = float(objective(model()))
                evaluations += 1
                if np.isfinite(cost) and cost < local_cost:
                    local_cost, local_state = cost, snapshot()
                    if cost < best_cost:
                        best_cost, best_state, best_start = cost, snapshot(), current_start
        model.load_state_dict(local_state)
        evaluate()  # Recompute the gradient of the retained candidate.
        return before-local_cost

    for current_start, candidate in enumerate(starting_points):
        model = ParameterizedGMM(candidate, coordinate_scale=coordinate_scale,
                                 learn_weights=learn_weights)
        parameters = [p for p in model.parameters() if p.requires_grad]
        local_cost, local_state = float("inf"), snapshot()
        start_iteration, start_evaluations = iteration, evaluations
        status, stalls, recoveries = "max_steps", 0, 0
        step = 0
        try:
            _, cost, gradient = evaluate()
            record(cost, gradient, 0, "initial")
            adam_steps = steps if optimizer_kind == "adam" else warmup_steps
            if adam_steps:
                warmup = torch.optim.Adam(parameters, lr=adam_learning_rate)
                for warm_step in range(1, adam_steps+1):
                    # evaluate() has computed gradients at the current point.
                    warmup.step()
                    _, cost, gradient = evaluate()
                    iteration += 1
                    record(cost, gradient, warm_step,
                           "adam" if optimizer_kind == "adam" else "adam_warmup")
                model.load_state_dict(local_state)
                _, cost, gradient = evaluate()
            optimizer = new_lbfgs() if optimizer_kind == "lbfgs" else None
            for step in range(1, (steps if optimizer_kind == "lbfgs" else 0)+1):
                if gradient <= tolerance_grad:
                    status = "gradient_tolerance"
                    break
                previous = local_cost
                optimizer.step(lambda: evaluate()[0])
                _, cost, gradient = evaluate()
                # Synchronize the actual iterate with the lowest line-search
                # candidate. Its curvature history belongs to another path.
                if cost > local_cost:
                    model.load_state_dict(local_state)
                    optimizer = new_lbfgs()
                    restarts += 1
                    _, cost, gradient = evaluate()
                iteration += 1
                record(cost, gradient, step, "lbfgs")
                threshold = tolerance_change*max(loss_scale, abs(previous))
                stalls = stalls+1 if previous-local_cost <= threshold else 0
                if stalls >= stall_patience and gradient > tolerance_grad:
                    if recoveries >= max_restarts:
                        status = "stalled"
                        break
                    gradient_backtracking()
                    optimizer = new_lbfgs()
                    recoveries += 1
                    restarts += 1
                    stalls = 0
                    _, cost, gradient = evaluate()
                    record(cost, gradient, step, "curvature_restart")
            model.load_state_dict(local_state)
            _, cost, gradient = evaluate()
            if gradient <= tolerance_grad:
                status = "gradient_tolerance"
        except (FloatingPointError, ValueError, RuntimeError) as error:
            status = f"numerical_stop: {type(error).__name__}: {error}"
            gradient = float("inf")
        summaries.append({"start": current_start, "objective_squared": local_cost,
                          "gradient_inf": gradient, "status": status,
                          "iterations": iteration-start_iteration,
                          "evaluations": evaluations-start_evaluations})

    current_start = best_start
    model.load_state_dict(best_state)
    _, best_cost, gradient = evaluate()
    record(best_cost, gradient, 0, "returned_best")
    result = detached_gmm(model())
    if best_cost > initial_cost + 1e-9*max(loss_scale, abs(initial_cost)):
        raise RuntimeError("Best-iterate restoration failed")
    met = gradient <= tolerance_grad
    status = "gradient_tolerance" if met else summaries[best_start]["status"]
    # The final gradient, rather than a line-search trial or start summary,
    # controls the only gradient-based stopping claim.
    if status == "gradient_tolerance" and not met:
        status = "stalled"
    return OptimizationResult(result, initial_cost, best_cost, iteration,
                              evaluations, status, history, gradient, met,
                              len(starting_points), restarts, summaries, gradient_evaluations)


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
