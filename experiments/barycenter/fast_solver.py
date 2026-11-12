"""Single-start Adam, fixed/fresh/adaptive slices, and validation selection.

Fresh finite-bank minima are NOT unbiased estimators of the slice infimum.
Every checkpoint is compared on one immutable, independent validation bank.
All sampling, validation and selection work belongs to the solver timer.
"""
from dataclasses import dataclass, fields
import time

import numpy as np
import torch
from torch.nn import functional as F

from lsot.gaussians import GMM
from lsot.plans import _monotone_entries, _projected_plan_entries
from lsot.projections import sample_projection_bank
from .fast_geometry import TensorGMM, pair_costs, project_components
from .solver import ParameterizedGMM, OptimizationResult


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def subset_bank(bank, indices):
    return type(bank)(*(getattr(bank, f.name)[indices] for f in fields(bank)))


def join_banks(first, second):
    return type(first)(*(torch.cat((getattr(first, f.name), getattr(second, f.name)))
                         for f in fields(first)))


class FastParameterizedGMM(ParameterizedGMM):
    def forward(self):
        raw = torch.diagonal(self.raw_cholesky, dim1=-2, dim2=-1)
        root = (torch.tril(self.raw_cholesky, diagonal=-1)
                + torch.diag_embed(F.softplus(raw)+self.minimum_diag))*self.scale
        return TensorGMM(self.logits.softmax(0), self.means*self.scale,
                         root @ root.transpose(-1, -2))


def plan_entries(x, y, a, b):
    """Padded monotone plans; exact proportional-fiber fallback for ties."""
    xs, ix = x.T.sort(dim=1, stable=True)
    ys, iy = y.T.sort(dim=1, stable=True)
    ties = (xs[:, 1:] == xs[:, :-1]).any() | (ys[:, 1:] == ys[:, :-1]).any()
    if bool(ties):
        return _projected_plan_entries(x, y, a, b)
    i, j, mass = _monotone_entries(a[ix], b[iy])
    directions = torch.arange(x.shape[1], device=x.device)[:, None].expand_as(mass)
    return (directions.flatten(), torch.gather(ix, 1, i).flatten(),
            torch.gather(iy, 1, j).flatten(), mass.flatten())


def edge_costs(source, target, rows, cols):
    """Deduplicate repeated edges before Gaussian geometry/backpropagation."""
    keys, inverse = (rows*target.count+cols).unique(return_inverse=True)
    return pair_costs(source, target, keys//target.count, keys % target.count)[inverse]


class FastObjective:
    def __init__(self, inputs, weights, bank, kind, mode, *, dense_threshold=4096):
        if mode not in {"min", "avg"}:
            raise ValueError("Expected min or avg")
        if len(weights) != len(inputs) or np.any(np.asarray(weights) < 0) or not np.isclose(sum(weights), 1):
            raise ValueError("Invalid input weights")
        self.inputs = [(g, float(w)) for g, w in zip(inputs, weights) if w > 0]
        self.kind, self.mode = kind, mode
        self.dense_threshold = dense_threshold
        self.last_winners = []
        self.set_bank(bank)

    def set_bank(self, bank):
        self.bank = bank
        with torch.no_grad():
            self.projected_inputs = [project_components(g, bank, self.kind) for g, _ in self.inputs]

    def __call__(self, candidate):
        with torch.no_grad():
            y = project_components(candidate, self.bank, self.kind)
        total = candidate.weights.new_zeros(())
        self.last_winners = []
        for (source, weight), x in zip(self.inputs, self.projected_inputs):
            if source.count*candidate.count <= self.dense_threshold:
                # For tiny 2-D problems a dense algebraic cost is cheaper than
                # sorting twice + deduplicating sparse edges. Same hard loss.
                directions, rows, cols, masses = plan_entries(x, y, source.weights, candidate.weights)
                costs = pair_costs(source, candidate)
                scores = masses.new_zeros(self.bank.count).scatter_add(0, directions, masses*costs[rows, cols])
                value = scores.min() if self.mode == "min" else scores.mean()
                if self.mode == "min":
                    self.last_winners.append(scores.detach().argmin().reshape(1))
            elif self.mode == "min":
                with torch.no_grad():
                    directions, rows, cols, masses = plan_entries(x, y, source.weights, candidate.weights)
                    values = edge_costs(source, candidate, rows, cols)
                    scores = masses.new_zeros(self.bank.count).scatter_add_(0, directions, masses*values)
                    # Match torch.min's equal sharing of gradients at exact ties.
                    winners = torch.where(scores == scores.min())[0]
                    self.last_winners.append(winners[:1])
                _, rows, cols, masses = plan_entries(x[:, winners], y[:, winners],
                                                    source.weights, candidate.weights)
                value = (masses*edge_costs(source, candidate, rows, cols)).sum()/len(winners)
            else:
                _, rows, cols, masses = plan_entries(x, y, source.weights, candidate.weights)
                value = (masses*edge_costs(source, candidate, rows, cols)).sum()/self.bank.count
            total = total+weight*value
        return total


class MW2Objective:
    """POT supplies cost-matrix gradients AND dual gradients for learned masses."""
    def __init__(self, inputs, weights):
        import ot
        if len(weights) != len(inputs) or np.any(np.asarray(weights) < 0) or not np.isclose(sum(weights), 1):
            raise ValueError("Invalid input weights")
        self.emd2 = ot.emd2
        self.inputs = [(g, float(w)) for g, w in zip(inputs, weights) if w > 0]

    def __call__(self, candidate):
        total = candidate.weights.new_zeros(())
        for source, weight in self.inputs:
            total = total+weight*self.emd2(source.weights, candidate.weights, pair_costs(source, candidate))
        return total


@dataclass
class FastResult(OptimizationResult):
    validation_objective: float
    validation_ms: float
    setup_ms: float
    selected_step: int
    checkpoints: list
    sampling: str


def _optimize_fast_stages(inputs, weights, initial, *, stage_steps=(), clock_start=None, kind="B1D", mode="min", steps=200,
                  learning_rate=.003, L=100, validation_L=512, seed=0,
                  sampling="fixed", refresh_every=10, refresh_fraction=.25,
                  validate_every=10, patience=5, relative_tolerance=1e-4,
                  coordinate_scale=1., learn_weights=True, checkpoints=(25, 50, 100, 200),
                  backend="fast", train_bank=None, validation_bank=None):
    """Adam only; checkpoints/early stopping use the fixed validation objective.

    Cached checkpoint states are evaluated offline by the benchmark driver.
    Numerical failure returns the best finite validated iterate and is recorded.
    backend='reference' is a controlled objective-kernel ablation, same loop.
    """
    if (steps < 0 or min(L, validation_L, validate_every, refresh_every) < 1
            or learning_rate <= 0 or coordinate_scale <= 0 or patience < 0
            or relative_tolerance < 0 or not 0 < refresh_fraction <= 1
            or sampling not in {"fixed", "fresh", "partial"}
            or backend not in {"fast", "reference"}):
        raise ValueError("Invalid fast optimizer settings")
    if backend == "reference" and sampling == "partial":
        raise ValueError("Winner-preserving partial refresh requires the fast objective")
    if mode == "mw2" and sampling != "fixed":
        raise ValueError("MW2-variational has no slices; use sampling=fixed")
    device = initial.weights.device
    synchronize(device)
    setup_started = time.perf_counter()
    started = setup_started if clock_start is None else clock_start
    model = FastParameterizedGMM(initial, coordinate_scale=coordinate_scale, learn_weights=learn_weights)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scale = coordinate_scale**2
    if mode == "mw2":
        objective = validation = MW2Objective(inputs, weights)
    else:
        from .solver import BarycenterObjective
        cls = FastObjective if backend == "fast" else BarycenterObjective
        def sample(count, offset):
            return sample_projection_bank(initial.dimension, count, kind=kind,
                                           seed=seed+offset, device=device, dtype=initial.weights.dtype)
        bank = train_bank if train_bank is not None else sample(L, 0)
        vbank = validation_bank if validation_bank is not None else sample(validation_L, 1000003)
        objective = cls(inputs, weights, bank, kind, mode)
        validation = cls(inputs, weights, vbank, kind, mode)
    synchronize(device)
    setup_ms = (time.perf_counter()-setup_started)*1000
    history, saved = [], []
    best_state, best_value, selected_step = None, float("inf"), 0
    validation_ms, evaluations, gradients, stale = 0., 0, 0, 0
    initial_cost, initial_validation = None, None
    status, completed, previous_winners = "max_steps", 0, []
    significant_best = float("inf")

    def evaluate_checkpoint(step, gradient=float("nan"), training_value=None):
        nonlocal best_state, best_value, selected_step, validation_ms, evaluations, stale
        nonlocal significant_best, initial_cost, initial_validation
        synchronize(device)
        tick = time.perf_counter()
        with torch.no_grad():
            candidate = model()
            # Full invariant validation is off the per-update path.
            GMM(candidate.weights, candidate.means, candidate.covariances)
            v = float(validation(candidate))
            t = float(objective(candidate)) if training_value is None else float(training_value)
        evaluations += 1+int(training_value is None)
        if not np.isfinite(v) or not np.isfinite(t):
            raise FloatingPointError("Nonfinite checkpoint objective")
        if initial_cost is None:
            initial_cost, initial_validation = t, v
        if v < best_value:
            best_value, selected_step = v, step
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        if not np.isfinite(significant_best) or v < significant_best-relative_tolerance*max(abs(significant_best), 1e-12*scale):
            significant_best, stale = v, 0
        elif step % validate_every == 0:
            stale += 1
        synchronize(device)
        validation_ms += (time.perf_counter()-tick)*1000
        elapsed = (time.perf_counter()-started)*1000
        history.append({"iteration": step, "start": 0, "phase": "validation", "elapsed_ms": elapsed,
                        "objective_squared": v, "best_objective_squared": best_value,
                        "training_objective_squared": t, "gradient_inf": float("nan"),
                        "training_gradient_before_update": gradient,
                        "best_gradient_inf": float("nan"), "tolerance_grad": 0.,
                        "evaluations": evaluations, "gradient_evaluations": gradients,
                        "bank_epoch": step if sampling == "fresh" else max(0, (step-1)//refresh_every) if sampling == "partial" else 0})
        if step in checkpoints or step in {0, steps}:
            # Keep on device; transfers/metric evaluation occur after the solve.
            saved.append({"iteration": step, "elapsed_ms": elapsed, "validation_objective": best_value,
                          "state": {k: p.clone() for k, p in best_state.items()},
                          "selected_step": selected_step})
        return v

    evaluate_checkpoint(0)
    try:
        for step in range(1, steps+1):
            if mode != "mw2" and sampling != "fixed":
                refresh = sampling == "fresh" or (step > 1 and (step-1) % refresh_every == 0)
                if refresh:
                    if sampling == "fresh":
                        bank = sample(L, 10_000_019+step)
                    else:
                        nnew = max(1, int(round(L*refresh_fraction)))
                        retained = torch.cat(previous_winners).unique() if previous_winners else torch.empty(0, dtype=torch.long, device=device)
                        # Retain every current pair's winner when possible, then fill deterministically.
                        nkeep = min(L-1, max(L-nnew, len(retained))) if L > 1 else 0
                        candidates = torch.cat((retained, torch.arange(bank.count, device=device)))
                        seen, ordered = set(), []
                        for idx in candidates.tolist():
                            if idx not in seen:
                                seen.add(idx)
                                ordered.append(idx)
                        indices = torch.tensor(ordered[:nkeep], device=device, dtype=torch.long)
                        bank = join_banks(subset_bank(bank, indices), sample(L-nkeep, 10_000_019+step))
                    if isinstance(objective, FastObjective):
                        objective.set_bank(bank)
                    else:
                        objective = cls(inputs, weights, bank, kind, mode)
            optimizer.zero_grad(set_to_none=True)
            value = objective(model())/scale
            evaluations += 1
            value.backward()
            gradients += 1
            previous_winners = getattr(objective, "last_winners", [])
            optimizer.step()
            completed = step
            if step % validate_every == 0 or step in checkpoints or step == steps or step in stage_steps:
                norm = float(torch.stack([p.grad.detach().abs().max() for p in model.parameters()
                                         if p.grad is not None]).max())
                if not np.isfinite(norm):
                    raise FloatingPointError("Nonfinite parameter gradient")
                evaluate_checkpoint(step, norm)
                if step in stage_steps:
                    # Yield with the model, Adam moments and bank still alive.
                    yield {"score": best_value, "iterations": completed,
                           "evaluations": evaluations, "gradient_evaluations": gradients,
                           "history": history, "validation_ms": validation_ms,
                           "setup_ms": setup_ms, "status": "stage_complete"}
                if patience and stale >= patience:
                    status = "validation_plateau"
                    break
    except (FloatingPointError, ValueError, RuntimeError) as exc:
        status = f"numerical_stop: {type(exc).__name__}: {exc}"
    model.load_state_dict(best_state)
    optimizer.zero_grad(set_to_none=True)
    final_value = validation(model())/scale
    final_value.backward()
    gradients += 1
    evaluations += 1
    gradient = float(torch.stack([p.grad.detach().abs().max() for p in model.parameters()
                                 if p.grad is not None]).max())
    candidate = model()
    result = GMM(*(p.detach().clone() for p in (candidate.weights, candidate.means, candidate.covariances)))
    # Always include the selected solution at termination for quality/time plots.
    synchronize(device)
    elapsed = (time.perf_counter()-started)*1000
    saved.append({"iteration": completed, "elapsed_ms": elapsed, "validation_objective": best_value,
                  "state": {k: p.detach().clone() for k, p in model.state_dict().items()}, "selected_step": selected_step})
    history.append({**history[-1], "iteration": completed, "phase": "returned_best", "elapsed_ms": elapsed,
                    "objective_squared": best_value, "best_objective_squared": best_value,
                    "gradient_inf": gradient, "best_gradient_inf": gradient,
                    "evaluations": evaluations, "gradient_evaluations": gradients})
    yield FastResult(result, initial_validation, best_value, completed, evaluations, status,
                      history, gradient, False, 1, 0,
                      [{"start": 0, "iterations": completed, "status": status,
                        "objective_squared": best_value, "selected_step": selected_step}],
                      gradients, best_value, validation_ms, setup_ms, selected_step, saved, sampling)


def checkpoint_gmm(initial, checkpoint, coordinate_scale=1., device="cpu"):
    model = FastParameterizedGMM(initial, coordinate_scale=coordinate_scale)
    model.load_state_dict(checkpoint["state"])
    with torch.no_grad():
        candidate = model()
    return GMM(*(p.detach().to(device).clone() for p in
                 (candidate.weights, candidate.means, candidate.covariances)))


def optimize_fast(inputs, weights, initial, **kwargs):
    """Single uninterrupted Adam solve."""
    return next(_optimize_fast_stages(inputs, weights, initial, **kwargs))


def optimize_halving(inputs, weights, initializations, *, probe_steps=(25, 50), **kwargs):
    """4 -> 2 -> 1 starts, ranking best fixed-bank validation objectives.

    Stages use cumulative steps and preserve live Adam and projection banks.
    Fresh directions have identical seeds across starts. Partial banks can
    diverge because winners depend on each candidate. All work is timed.
    """
    steps = kwargs.get("steps", 200)
    if len(probe_steps) != 2 or not 0 < probe_steps[0] < probe_steps[1] < steps:
        raise ValueError("Require 0 < probe_steps[0] < probe_steps[1] < steps")
    if len(initializations) not in (1, 4):
        raise ValueError("Halving requires four starts (one for a single Gaussian)")
    if kwargs.get("patience", 0):
        raise ValueError("Halving requires patience=0")
    kwargs["patience"] = 0
    synchronize(initializations[0].weights.device)
    started = time.perf_counter()
    workers = {i: _optimize_fast_stages(inputs, weights, initial, stage_steps=probe_steps,
                                      clock_start=started, **kwargs)
               for i, initial in enumerate(initializations)}
    latest, active, decisions = {}, list(workers), []
    for cut, keep in zip(probe_steps, (2, 1)):
        for i in active:
            # A numerical stop can terminate a start before its scheduled cut.
            if i not in latest or isinstance(latest[i], dict):
                latest[i] = next(workers[i])
        def score(i):
            r = latest[i]
            return r["score"] if isinstance(r, dict) else r.objective
        survivors = sorted(active, key=lambda i: (score(i), i))[:keep]
        decisions.append({"iteration": cut, "scores": {i: score(i) for i in active},
                          "survivors": survivors})
        for i in active:
            if i not in survivors:
                workers[i].close()
        active = survivors
    selected = active[0]
    result = latest[selected]
    if isinstance(result, dict):
        result = next(workers[selected])
    latest[selected] = result
    workers[selected].close()
    histories, summaries = [], []
    total_evals = total_grads = 0
    validation_ms = setup_ms = 0.
    for i, r in latest.items():
        get = r.get if isinstance(r, dict) else lambda key, r=r: getattr(r, key)
        histories.extend(dict(row, start=i) for row in get("history"))
        summaries.append({"start": i, "iterations": get("iterations"),
                          "status": get("status") if i == selected else "pruned",
                          "objective_squared": r["score"] if isinstance(r, dict) else r.objective,
                          "selected": i == selected})
        total_evals += get("evaluations")
        total_grads += get("gradient_evaluations")
        validation_ms += get("validation_ms")
        setup_ms += get("setup_ms")
    result.history = sorted(histories, key=lambda row: row["elapsed_ms"])
    result.starts, result.start_summaries = len(initializations), summaries
    result.evaluations, result.gradient_evaluations = total_evals, total_grads
    result.validation_ms, result.setup_ms = validation_ms, setup_ms
    result.halving_decisions = decisions
    result.total_updates = sum(row["iterations"] for row in summaries)
    result.selected_start = selected
    return result
