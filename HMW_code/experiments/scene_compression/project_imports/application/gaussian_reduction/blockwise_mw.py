"""Experimental GHAP-partitioned, fixed-capacity MW barycentric reduction.

Not original GHAP: its free-weight CTD reduction has closed-form assignment.
The exact/mean-U2/initial-plan arms share EVERYTHING except solve_plan().
Uses upstream KD partition/export/donor semantics and the existing BW updater.
No images, cameras, renderer, reference plans or quality metrics enter here.
"""
from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np
import torch

from application.common import SparseCoupling, as_application_mixture
from application.gaussian_reduction.gsplat_bridge import (
    extract_gaussian_model, torch_bures_projection,
)


METHODS = {"block_mw_exact", "block_u2_mean", "block_initial_plan"}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def principal_root(covariance):
    values, vectors = torch.linalg.eigh(covariance)
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)


def bw_edges(x, cov, roots, y, target_cov, rows, columns):
    """True Gaussian W2 squared, float64; no root-gauge substitution."""
    ri = torch.as_tensor(np.array(rows), device=x.device, dtype=torch.long)
    ci = torch.as_tensor(np.array(columns), device=x.device, dtype=torch.long)
    r = roots[ri]
    middle = r @ target_cov[ci] @ r
    middle = (middle + middle.transpose(-1, -2)) * 0.5
    cross_trace = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum(-1)
    shape = (cov[ri].diagonal(dim1=-2, dim2=-1).sum(-1)
             + target_cov[ci].diagonal(dim1=-2, dim2=-1).sum(-1) - 2 * cross_trace)
    return ((x[ri] - y[ci]).square().sum(-1) + shape.clamp_min(0)).cpu().numpy()


def bw_matrix(x, cov, roots, y, target_cov, chunk=65536):
    n, m = len(x), len(y)
    output = np.empty(n * m, dtype=np.float64)
    for start in range(0, n * m, chunk):
        flat = np.arange(start, min(start + chunk, n * m))
        output[start:start + len(flat)] = bw_edges(
            x, cov, roots, y, target_cov, flat // m, flat % m)
    return output.reshape(n, m)


def make_plan(a, b, rows, columns, masses, costs=None, diagnostics=None):
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    masses = np.asarray(masses, dtype=np.float64)
    costs = np.zeros_like(masses) if costs is None else np.asarray(costs)
    residual = max(np.max(np.abs(np.bincount(rows, weights=masses, minlength=len(a)) - a)),
                   np.max(np.abs(np.bincount(columns, weights=masses, minlength=len(b)) - b)))
    if residual > 1e-9 or np.any(masses < 0) or not np.isfinite(costs).all():
        raise RuntimeError("invalid block coupling")
    return SparseCoupling(rows, columns, masses, costs, np.arange(len(a)),
                          np.arange(len(b)), a, b, float(masses @ costs),
                          float(residual), diagnostics or {})


def occupancy_initialization(x, a, ids):
    """Fixed Voronoi capacity from source geometry; reserve each sampled atom.

    A sampled atom is assigned to its own target even in exact position ties.
    This guarantees strictly positive capacities without fake mass/floors.
    """
    distance = (x[:, None] - x[ids][None]).square().sum(-1)
    assignment = distance.argmin(1).cpu().numpy()
    assignment[ids] = np.arange(len(ids))
    b = np.bincount(assignment, weights=a, minlength=len(ids))
    if np.any(b <= 0):
        raise RuntimeError("initial capacities must be positive")
    return b, make_plan(a, b, np.arange(len(a)), assignment, a)


def solve_plan(method, a, b, x, cov, roots, y, target_cov, initial, fast_solver, source):
    """Only solver-dependent seam; exact result checked with primal/dual audit."""
    if method == "block_initial_plan":
        return initial, {"ground_seconds": 0.0, "solver_seconds": 0.0}
    started = perf_counter()
    if method == "block_u2_mean":
        target = as_application_mixture(y.cpu().numpy(), target_cov.cpu().numpy(), b)
        plan = fast_solver(source, target)
        return plan, {"ground_seconds": 0.0, "solver_seconds": perf_counter() - started}
    if method != "block_mw_exact":
        raise ValueError(method)
    import ot
    cost = bw_matrix(x, cov, roots, y, target_cov)
    ground_seconds = perf_counter() - started
    started = perf_counter()
    scale = max(float(cost.max()), np.finfo(np.float64).tiny)
    scaled = np.ascontiguousarray(cost / scale)
    dense, log = ot.emd(a, b, scaled, numItermax=1000000, numThreads=1, log=True)
    if log["result_code"] != 1 or log["warning"] is not None:
        raise RuntimeError(f"EMD failed to certify optimality: {log}")
    primal = float(np.sum(dense * scaled))
    dual = float(a @ log["u"] + b @ log["v"])
    violation = float(np.max(log["u"][:, None] + log["v"][None, :] - scaled))
    if abs(primal - dual) > 1e-8 or violation > 1e-8:
        raise RuntimeError("EMD primal/dual certificate failed")
    rows, columns = np.nonzero(dense > 0)
    certificate = {"scaled_primal_dual_gap": abs(primal - dual),
                   "scaled_dual_violation": max(0.0, violation), "emd_status": 1}
    plan = make_plan(a, b, rows, columns, dense[rows, columns], cost[rows, columns], certificate)
    return plan, {"ground_seconds": ground_seconds,
                  "solver_seconds": perf_counter() - started, **certificate}


def reduce_block(x, cov, a, ids, *, method, outer_limit=10, inner_limit=64):
    roots = principal_root(cov)
    b, initial = occupancy_initialization(x, a, ids)
    y, target_cov = x[ids].clone(), cov[ids].clone()
    source = as_application_mixture(x.cpu().numpy(), cov.cpu().numpy(), a)
    fast_solver = None
    if method == "block_u2_mean":
        from application.gaussian_reduction.fixed_u2_coupling import FixedU2MeanCouplingSolver
        fast_solver = FixedU2MeanCouplingSolver()
    previous = initial
    history, converged, reason = [], False, "outer_limit"
    spatial_scale = float(((x - (x * torch.as_tensor(a, device=x.device)[:, None]).sum(0))
                           .square().sum(-1) * torch.as_tensor(a, device=x.device)).sum().sqrt())
    for step in range(1, outer_limit + 1):
        plan, timing = solve_plan(method, a, b, x, cov, roots, y, target_cov,
                                  initial, fast_solver, source)
        check_start = perf_counter()
        costs = bw_edges(x, cov, roots, y, target_cov, plan.rows, plan.columns)
        before = float(plan.masses @ costs)
        old_cost = float(previous.masses @ bw_edges(
            x, cov, roots, y, target_cov, previous.rows, previous.columns))
        rejected = before > old_cost + 1e-10 * max(1.0, old_cost)
        if rejected:
            plan, before = previous, old_cost
        guard_seconds = perf_counter() - check_start
        started = perf_counter()
        new_y, new_cov, inner = torch_bures_projection(
            x, cov, target_cov, plan, maximum_iterations=inner_limit,
            relative_tolerance=5e-4, relative_eigenvalue_floor=1e-12)
        synchronize(x.device)
        update_seconds = perf_counter() - started
        started = perf_counter()
        after = float(plan.masses @ bw_edges(
            x, cov, roots, new_y, new_cov, plan.rows, plan.columns))
        update_ok = after <= before + 1e-9 * max(1.0, before)
        mean_change = float(torch.linalg.vector_norm(new_y - y, dim=1).max()) / max(spatial_scale, 1e-12)
        cov_change = float((torch.linalg.matrix_norm(new_cov - target_cov)
                           / torch.linalg.matrix_norm(target_cov).clamp_min(1e-30)).max())
        audit_seconds = perf_counter() - started
        history.append({"step": step, "objective_before": before, "objective_after": after,
                        "proposal_rejected": rejected, "update_accepted": update_ok,
                        "mean_relative_change": mean_change, "cov_max_relative_change": cov_change,
                        "guard_seconds": guard_seconds, "audit_seconds": audit_seconds,
                        "update_seconds": update_seconds, "edges": len(plan.masses),
                        "marginal_residual": plan.marginal_residual, **timing, **dict(inner)})
        if not update_ok:
            reason = "numerical_update_rejected"
            break
        y, target_cov, previous = new_y, new_cov, plan
        relative_improvement = abs(before - after) / max(before, 1e-12)
        if (mean_change <= 1e-4 and cov_change <= 5e-4
                and relative_improvement <= 1e-4 and inner["barycenter_converged"]):
            converged, reason = True, "geometry_and_objective_stable"
            break
    # Appearance donors reproduce upstream block-local mean+covariance cost.
    donor_cost = ((x[:, None] - y[None]).square().sum(-1)
                  + (cov[:, None] - target_cov[None]).square().sum((-1, -2)))
    donors = donor_cost.argmin(0).cpu().numpy()
    return y, target_cov, donors, {"history": history, "converged": converged,
        "stop_reason": reason, "source_count": len(x), "target_count": len(y),
        "initial_ids": ids.tolist(), "capacity_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
        "capacity_min": float(b.min()), "capacity_max": float(b.max())}


def blockwise_payload(model, target_components, seed, *, method,
                      outer_limit=10, inner_limit=64, depth=10):
    from gmm_sampler import balanced_kd_split, decompose_covariance_torch
    started = perf_counter()
    snapshot = extract_gaussian_model(model)
    if snapshot.mixture.support_size != len(snapshot.xyz):
        raise ValueError("pilot requires positive opacity for all original atoms")
    _, blocks = balanced_kd_split(snapshot.xyz.cpu().numpy(), np.arange(len(snapshot.xyz)), max_depth=depth)
    if target_components < len(blocks):
        raise ValueError("target budget is smaller than number of spatial blocks")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    all_y, all_cov, all_donors, records = [], [], [], []
    left = target_components
    synchronize(snapshot.xyz.device)
    prep_seconds = perf_counter() - started
    for k, block in enumerate(blocks):
        block_started = perf_counter()
        m = left // (len(blocks) - k)
        left -= m
        if m > len(block):
            raise ValueError("block target budget exceeds its source count")
        ids = torch.randperm(len(block), generator=generator)[:m].numpy()
        index = torch.as_tensor(block, device=snapshot.xyz.device)
        x, cov = snapshot.xyz[index].double(), snapshot.covariances[index].double()
        a = snapshot.mixture.weights[block].copy()
        a /= a.sum()
        y, reduced_cov, donors, record = reduce_block(
            x, cov, a, ids, method=method, outer_limit=outer_limit, inner_limit=inner_limit)
        all_y.append(y.float())
        all_cov.append(reduced_cov.float())
        all_donors.append(block[donors])
        synchronize(snapshot.xyz.device)
        record.update(block=k, runtime_seconds=perf_counter() - block_started,
                      original_block_mass=float(snapshot.mixture.weights[block].sum()))
        records.append(record)
        if k % 16 == 0 or k == len(blocks) - 1:
            print(f"BLOCK_MW_PROGRESS block={k+1}/{len(blocks)} method={method} "
                  f"elapsed={perf_counter()-started:.2f}s steps={len(record['history'])} "
                  f"converged={record['converged']}", flush=True)
    joined_cov = torch.cat(all_cov)
    scaling, rotation, _ = decompose_covariance_torch(joined_cov)
    donors = torch.as_tensor(np.concatenate(all_donors), device=snapshot.xyz.device)
    payload = {"xyz": torch.cat(all_y), "scaling": torch.log(scaling), "rotation": rotation,
               "f_dc": snapshot.features_dc[donors].clone(),
               "f_rest": snapshot.features_rest[donors].clone(),
               "opacity": model._opacity.detach()[donors].clone()}
    if any(not bool(torch.isfinite(v).all()) for v in payload.values()):
        raise RuntimeError("nonfinite payload")
    synchronize(snapshot.xyz.device)
    return payload, {"algorithm": method, "method_role": "exploratory_fixed_capacity_MW_reduction",
        "is_original_ghap": False, "source_prep_seconds": prep_seconds,
        "block_count": len(blocks), "converged_blocks": sum(r["converged"] for r in records),
        "block_records": records, "appearance": "upstream_block_local_mean_cov_donor",
        "capacity": "initial_spatial_voronoi_frozen", "depth": depth,
        "outer_limit": outer_limit, "inner_limit": inner_limit,
        "covariance_export": "upstream_absolute_eigenvalue_floor_1e-6",
        "coordinate_preconditioning": "none", "geometry_dtype": "float64"}
