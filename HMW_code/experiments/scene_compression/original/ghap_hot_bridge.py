"""One compaction seam shared by official GHAP and NC-HOT.

This file is copied unchanged into the root of an official GHAP checkout by
``deploy.sh``.  Keeping all model conversion, exact-budget checks, statistics,
and post-compaction freezing here makes the patch against GHAP deliberately
small and auditable.

The NC-HOT imports are lazy: the GHAP baseline can run with only the upstream
dependencies, while ``hot_t4``/``hot_t0`` and the explicitly labelled
``hot_spatial_h1_control`` additionally require this project's root on
``PYTHONPATH``.
"""

from __future__ import annotations

import json
import inspect
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch


OFFICIAL_GHAP_COMMIT = "27b3a581c861069cc97e5629e216dc3d1f42a98a"
GEOMETRY_GROUPS = frozenset({"xyz", "scaling", "rotation"})
HOT_APPEARANCE_PROJECTIONS = frozenset(
    {"coupling_mean", "nearest_incoming_donor"}
)
SPATIAL_H1_CONTROL_METHOD = "hot_spatial_h1_control"
FIXED_U2_MEAN_METHOD = "hot_fixed_u2_mean"
SUPPORTED_METHODS = frozenset(
    {"ghap", "hot_t4", "hot_t0", SPATIAL_H1_CONTROL_METHOD, FIXED_U2_MEAN_METHOD, "random",
     "block_mw_exact", "block_u2_mean", "block_initial_plan"}
)
PARAMETER_FIELDS = {
    "xyz": "_xyz",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "scaling": "_scaling",
    "rotation": "_rotation",
    "opacity": "_opacity",
}


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to real-first (wxyz) quaternions.

    This is the numerically stable largest-component construction used by
    PyTorch3D, expressed only with PyTorch primitives.  It is intentionally
    provided as a fallback so official GHAP does not require the large
    ``pytorch3d`` package solely for one conversion routine.
    """

    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape (..., 3, 3)")
    if not (matrix.is_floating_point() and torch.isfinite(matrix).all()):
        raise ValueError("matrix must be a finite floating-point tensor")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    component_magnitudes = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack(
                (
                    component_magnitudes[..., 0] ** 2,
                    m21 - m12,
                    m02 - m20,
                    m10 - m01,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m21 - m12,
                    component_magnitudes[..., 1] ** 2,
                    m10 + m01,
                    m02 + m20,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m02 - m20,
                    m10 + m01,
                    component_magnitudes[..., 2] ** 2,
                    m12 + m21,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m10 - m01,
                    m20 + m02,
                    m21 + m12,
                    component_magnitudes[..., 3] ** 2,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    denominator = 2.0 * component_magnitudes.clamp_min(0.1)[..., :, None]
    candidates = candidates / denominator
    selected = component_magnitudes.argmax(dim=-1)
    gather_index = selected[..., None, None].expand(*selected.shape, 1, 4)
    quaternion = candidates.gather(-2, gather_index).squeeze(-2)
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternion.dtype).eps)


def _validate_target_components(target_components: int, source_count: int) -> int:
    if isinstance(target_components, bool) or not isinstance(
        target_components, (int, np.integer)
    ):
        raise ValueError("target_components must be an integer")
    target = int(target_components)
    if not 1 <= target <= int(source_count):
        raise ValueError(
            "target_components must satisfy 1 <= target_components <= source count"
        )
    return target


def _replace_parameter(gaussians: Any, name: str, value: torch.Tensor) -> None:
    replaced = gaussians.replace_tensor_to_optimizer(value.contiguous(), name)
    setattr(gaussians, PARAMETER_FIELDS[name], replaced[name])


def _install_payload(gaussians: Any, payload: Mapping[str, torch.Tensor]) -> Any:
    for name in ("xyz", "f_dc", "f_rest", "scaling", "rotation", "opacity"):
        _replace_parameter(gaussians, name, payload[name])
    count = int(payload["xyz"].shape[0])
    device = payload["xyz"].device
    gaussians.xyz_gradient_accum = torch.zeros((count, 1), device=device)
    gaussians.denom = torch.zeros((count, 1), device=device)
    gaussians.max_radii2D = torch.zeros(count, device=device)
    return gaussians


def _ghap_payload(
    gaussians: Any, target_components: int, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    # Imported only for this branch so HOT does not accidentally depend on the
    # baseline implementation and GHAP remains at the pinned official code.
    from gmm_sampler import gaussian_model_reduction

    reduced = gaussian_model_reduction(gaussians, int(target_components), seed)
    payload = {
        "xyz": reduced._xyz.detach(),
        "f_dc": reduced._features_dc.detach(),
        "f_rest": reduced._features_rest.detach(),
        "scaling": reduced._scaling.detach(),
        "rotation": reduced._rotation.detach(),
        "opacity": reduced._opacity.detach(),
    }
    return payload, {
        "algorithm": "official_ghap_kd_gmr_ctd",
        "official_commit": OFFICIAL_GHAP_COMMIT,
    }


def _print_reduction_progress(record: Mapping[str, Any]) -> None:
    print("HOT_REDUCTION_ITERATION " + json.dumps(_jsonable(record), sort_keys=True),
          flush=True)


def _hot_payload(
    gaussians: Any,
    target_components: int,
    seed: int,
    *,
    refinement_iterations: int,
    initialization: str,
    outer_iterations: int,
    barycenter_iterations: int,
    appearance_projection: str,
    fixed_u2_mean: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    # The production bridge performs the million-scale chunked Torch Bures
    # updates, re-solves the final plan, and projects activated opacity before
    # applying inverse-sigmoid.  Keep that logic in its single canonical file.
    from application.gaussian_reduction.gsplat_bridge import reduce_gaussian_model

    solver_options = {}
    checkpoint_root = os.environ.get("HOT_REDUCTION_CHECKPOINT_DIR")
    if checkpoint_root:
        from application.gaussian_reduction.ghap_integration.per_round_checkpoints import make_checkpoint_callback
        solver_options["checkpoint_callback"] = make_checkpoint_callback(
            gaussians, checkpoint_root
        )
    if fixed_u2_mean:
        from application.gaussian_reduction.fixed_u2_coupling import FixedU2MeanCouplingSolver
        solver_options["coupling_solver"] = FixedU2MeanCouplingSolver()
        if os.environ.get("HOT_REDUCTION_PROGRESS") == "1":
            solver_options["progress_callback"] = _print_reduction_progress
    result = reduce_gaussian_model(
        gaussians,
        target_components=target_components,
        refinement_iterations=refinement_iterations,
        seed=seed,
        initialization=initialization,
        maximum_outer_iterations=outer_iterations,
        barycenter_maximum_iterations=barycenter_iterations,
        appearance_projection=appearance_projection,
        **solver_options,
    )
    coupling = result.coupling
    payload = dict(result.tensors)
    reduction_stats = {
        "algorithm": (
            "sa_nc_hot_h6_t4_plan"
            if refinement_iterations == 4
            else "sa_nc_hot_h6_t0_mean_plan"
        ),
        "initialization": initialization,
        "appearance_projection": appearance_projection,
        "outer_iterations": int(result.iterations),
        "converged": bool(result.converged),
        "stop_reason": result.stop_reason,
        "active_coupling_edges": int(coupling.active_edges),
        "marginal_residual": float(coupling.marginal_residual),
        "final_coupling_surrogate_objective": float(coupling.objective),
        "reduction_diagnostics": _jsonable(result.diagnostics),
        "coupling_diagnostics": _jsonable(coupling.diagnostics),
        "history": _jsonable(result.history),
    }
    if fixed_u2_mean:
        reduction_stats.update(
            algorithm="fixed_u2_h6_mean_root_coupling",
            method_role="fixed_u2_main_mean_coupling",
            coupling_solver_calls=result.diagnostics["coupling_solver_calls"],
        )
    return payload, reduction_stats


def _spatial_h1_control_payload(
    gaussians: Any,
    target_components: int,
    seed: int,
    *,
    outer_iterations: int,
    barycenter_iterations: int,
    appearance_projection: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run the spatial-H1 diagnostic through the canonical 3DGS bridge.

    This branch is intentionally separate from ``_hot_payload`` so selecting
    the diagnostic control cannot alter the production H6/T4 or H6/T0 paths.
    The coupling solver and its equal-mass initialization use the same
    source-conditioned three-dimensional Hilbert order.
    """

    from application.gaussian_reduction.diagnostics.spatial_h1_control import (
        METHOD as SPATIAL_SOLVER_METHOD,
        SpatialH1DiagnosticControlSolver,
    )
    from application.gaussian_reduction.gsplat_bridge import reduce_gaussian_model

    solver = SpatialH1DiagnosticControlSolver()
    result = reduce_gaussian_model(
        gaussians,
        target_components=target_components,
        seed=seed,
        initialization="hilbert_equal_mass",
        maximum_outer_iterations=outer_iterations,
        barycenter_maximum_iterations=barycenter_iterations,
        appearance_projection=appearance_projection,
        coupling_solver=solver,
    )
    coupling = result.coupling
    payload = dict(result.tensors)
    reduction_stats = {
        "algorithm": SPATIAL_SOLVER_METHOD,
        "method_role": "diagnostic_control",
        "is_production_h6_t4": False,
        "initialization": "hilbert_equal_mass",
        "initialization_order": "source_preconditioned_spatial_hilbert",
        "appearance_projection": appearance_projection,
        "outer_iterations": int(result.iterations),
        "coupling_solver_calls": int(
            result.diagnostics.get("coupling_solver_calls", result.iterations + 1)
        ),
        "converged": bool(result.converged),
        "stop_reason": result.stop_reason,
        "active_coupling_edges": int(coupling.active_edges),
        "marginal_residual": float(coupling.marginal_residual),
        "final_coupling_surrogate_objective": float(coupling.objective),
        "reduction_diagnostics": _jsonable(result.diagnostics),
        "coupling_diagnostics": _jsonable(coupling.diagnostics),
        "history": _jsonable(result.history),
    }
    return payload, reduction_stats


def _random_payload(
    gaussians: Any, target_components: int, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    generator = np.random.default_rng(seed)
    indices = torch.as_tensor(
        generator.choice(
            int(gaussians.get_xyz.shape[0]), size=target_components, replace=False
        ),
        device=gaussians._xyz.device,
        dtype=torch.long,
    )
    payload = {
        "xyz": gaussians._xyz.detach()[indices],
        "f_dc": gaussians._features_dc.detach()[indices],
        "f_rest": gaussians._features_rest.detach()[indices],
        "scaling": gaussians._scaling.detach()[indices],
        "rotation": gaussians._rotation.detach()[indices],
        "opacity": gaussians._opacity.detach()[indices],
    }
    return payload, {"algorithm": "uniform_random_without_replacement"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_stats(path: str | os.PathLike[str] | None, record: Mapping[str, Any]) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if destination.suffix.lower() == ".json" else "a"
    with destination.open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def enforce_geometry_freeze(gaussians: Any) -> None:
    """Freeze xyz/scaling/rotation parameters and keep their group LR at zero."""

    for name in GEOMETRY_GROUPS:
        parameter = getattr(gaussians, "_" + name)
        parameter.requires_grad_(False)
        parameter.grad = None
    optimizer = getattr(gaussians, "optimizer", None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            if group.get("name") in GEOMETRY_GROUPS:
                group["lr"] = 0.0


def update_learning_rate_compat(gaussians: Any, iteration: int) -> Any:
    """Call either GHAP's two-argument or upstream 3DGS's one-argument API."""

    update = gaussians.update_learning_rate
    # ``update`` is bound, so official/upstream variants expose two/one
    # parameters respectively.  Signature inspection avoids swallowing a real
    # TypeError raised inside the scheduler.
    parameter_count = len(inspect.signature(update).parameters)
    if parameter_count >= 2:
        return update(iteration, False)
    return update(iteration)


def compact_gaussians(
    gaussians: Any,
    *,
    method: str,
    target_components: int,
    seed: int = 42,
    iteration: int | None = None,
    stats_path: str | os.PathLike[str] | None = None,
    hot_initialization: str = "opacity_random",
    hot_outer_iterations: int = 5,
    hot_barycenter_iterations: int = 20,
    hot_appearance_projection: str = "coupling_mean",
) -> tuple[Any, dict[str, Any]]:
    """Compact a live 3DGS model through one exact-budget integration seam."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)}")
    if (
        method == SPATIAL_H1_CONTROL_METHOD
        and hot_initialization != "hilbert_equal_mass"
    ):
        raise ValueError(
            "hot_spatial_h1_control requires "
            "hot_initialization='hilbert_equal_mass'"
        )
    if (
        method in {"hot_t4", "hot_t0", SPATIAL_H1_CONTROL_METHOD, FIXED_U2_MEAN_METHOD}
        and hot_appearance_projection not in HOT_APPEARANCE_PROJECTIONS
    ):
        raise ValueError(
            "hot_appearance_projection must be one of "
            f"{sorted(HOT_APPEARANCE_PROJECTIONS)}"
        )
    source_count = int(gaussians.get_xyz.shape[0])
    target_components = _validate_target_components(
        target_components, source_count
    )
    if hot_outer_iterations < 1 or hot_barycenter_iterations < 1:
        raise ValueError("HOT iteration limits must be positive")

    device = gaussians._xyz.device
    using_cuda = device.type == "cuda" and torch.cuda.is_available()
    if using_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    record: dict[str, Any] = {
        "status": "running",
        "method": method,
        "iteration": iteration,
        "seed": int(seed),
        "source_components": source_count,
        "target_components_requested": target_components,
        "target_components": target_components,
        "device": str(device),
    }
    try:
        if method == "ghap":
            payload, method_stats = _ghap_payload(
                gaussians, target_components, seed
            )
        elif method in {"block_mw_exact", "block_u2_mean", "block_initial_plan"}:
            from application.gaussian_reduction.blockwise_mw import blockwise_payload
            payload, method_stats = blockwise_payload(
                gaussians, target_components, seed, method=method,
                outer_limit=int(hot_outer_iterations),
                inner_limit=int(hot_barycenter_iterations),
            )
        elif method in {"hot_t4", "hot_t0", FIXED_U2_MEAN_METHOD}:
            payload, method_stats = _hot_payload(
                gaussians,
                target_components,
                seed,
                refinement_iterations=4 if method == "hot_t4" else 0,
                initialization=hot_initialization,
                outer_iterations=int(hot_outer_iterations),
                barycenter_iterations=int(hot_barycenter_iterations),
                appearance_projection=hot_appearance_projection,
                fixed_u2_mean=(method == FIXED_U2_MEAN_METHOD),
            )
        elif method == SPATIAL_H1_CONTROL_METHOD:
            payload, method_stats = _spatial_h1_control_payload(
                gaussians,
                target_components,
                seed,
                outer_iterations=int(hot_outer_iterations),
                barycenter_iterations=int(hot_barycenter_iterations),
                appearance_projection=hot_appearance_projection,
            )
        else:
            payload, method_stats = _random_payload(
                gaussians, target_components, seed
            )
        output_count = int(payload["xyz"].shape[0])
        valid_budget = (
            1 <= output_count <= target_components
            if method == "ghap"
            else output_count == target_components
        )
        if not valid_budget:
            raise RuntimeError(
                f"{method} returned {output_count} components; budget request "
                f"was {target_components}"
            )
        _install_payload(gaussians, payload)
        if int(gaussians.get_xyz.shape[0]) != output_count:
            raise RuntimeError("installed 3DGS model count differs from method output")
        if using_cuda:
            torch.cuda.synchronize(device)
        record.update(
            {
                "status": "ok",
                "target_components": output_count,
                "output_components": output_count,
                "exact_budget_preserved": output_count == target_components,
                "reduction_ratio_realized": output_count / source_count,
                **method_stats,
            }
        )
    except Exception as error:
        if using_cuda:
            torch.cuda.synchronize(device)
        record.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        record["runtime_seconds"] = perf_counter() - started
        if using_cuda:
            record["peak_gpu_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
        _write_stats(stats_path, record)
    return gaussians, record
