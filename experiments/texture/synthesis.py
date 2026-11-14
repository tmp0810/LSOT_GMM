"""Fit patch mixtures, construct component couplings, and apply T_mean."""
from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
import ot
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
import torch

from lsot import BarycentricMap, GMM, SparsePlan, average_lsot, minimum_lsot
from lsot.gaussians import gaussian_pair_costs
from lsot.projections import PROJECTION_KINDS
from .data import aggregate_patches


METHODS = ("MW2",) + tuple(f"{mode}-LSOT-{kind}" for mode in ("min", "avg")
                          for kind in PROJECTION_KINDS)


def fit_gmm(patches, components, seed, *, iterations, tolerance, regularization):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = GaussianMixture(
            n_components=components, covariance_type="full", init_params="kmeans",
            n_init=1, random_state=seed, max_iter=iterations, tol=tolerance,
            reg_covar=regularization,
        ).fit(patches)
    model = GMM.from_numpy(fit.weights_, fit.means_, fit.covariances_)
    return model, dict(converged=bool(fit.converged_), iterations=int(fit.n_iter_),
                       lower_bound=float(fit.lower_bound_), samples=len(patches),
                       warnings=[str(item.message) for item in caught])


def mw2_plan(source, target):
    """Exact component LP, using the same Gaussian cost kernel as min-LSOT."""
    rows = torch.arange(source.count).repeat_interleave(target.count)
    cols = torch.arange(target.count).repeat(source.count)
    cost = gaussian_pair_costs(source, target, rows, cols).reshape(source.count, target.count)
    coupling, info = ot.emd(source.weights.numpy(), target.weights.numpy(), cost.numpy(),
                            numThreads=1, log=True)
    if info["warning"] is not None:
        raise RuntimeError(f"MW2 LP did not solve successfully: {info['warning']}")
    plan = SparsePlan.from_dense(torch.from_numpy(coupling))
    return plan, float(info["cost"])


@dataclass
class Synthesis:
    image: np.ndarray
    plan: SparsePlan
    timings: dict
    projection_index: int | None
    plan_cost_squared: float | None


@torch.inference_mode()
def synthesize(source, target, source_patches, shape, patch_size, method, bank, batch_size):
    """Time plan, affine-map setup, patch transformation and overlap-add."""
    start = perf_counter()
    direction = None
    cost = None
    if method == "MW2":
        plan, cost = mw2_plan(source, target)
    else:
        mode, _, kind = method.split("-")
        if mode == "min":
            result = minimum_lsot(source, target, bank, kind)
            plan, direction = result.plan, result.projection_index
            cost = float(result.cost_squared)
        elif mode == "avg":
            plan = average_lsot(source, target, bank, kind)
        else:
            raise ValueError(f"Unknown method: {method}")
    planned = perf_counter()
    mapper = BarycentricMap.from_plan(source, target, plan)
    setup = perf_counter()
    mapped = mapper.transform(source_patches, batch_size=batch_size)
    applied = perf_counter()
    image = aggregate_patches(mapped, shape, patch_size)
    if not torch.isfinite(image).all():
        raise FloatingPointError(f"{method} produced a nonfinite image")
    pixels = image.clamp_(0, 1).numpy()
    end = perf_counter()
    return Synthesis(pixels, plan, dict(
        plan_ms=1000*(planned-start), map_setup_ms=1000*(setup-planned),
        map_apply_ms=1000*(applied-setup), aggregate_ms=1000*(end-applied),
        method_ms=1000*(end-start)), direction, cost)
