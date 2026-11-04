"""Minimal bridge from graphdeco 3D Gaussian Splatting to GMM reduction.

The bridge deliberately leaves the renderer and training loop untouched.  It
reads a ``GaussianModel`` once, alternates the application's fixed balanced
coupling with reverse-conditional Bures barycentric updates, transfers the SH
and activated opacity attributes through the last coupling, and returns the
six tensors consumed by graphdeco's ``replace_tensor_to_optimizer`` method.

No PyTorch3D dependency is used.  Full covariances are converted back to
3DGS log-scales and real-first quaternions with ``torch.linalg.eigh`` and a
small batched matrix-to-quaternion implementation.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from application.common import ApplicationMixture, SparseCoupling, as_application_mixture
from application.gaussian_reduction.main_coupling import MainCouplingSolver

try:  # Keep documentation/import tooling usable without the server torch env.
    import torch
except ImportError:  # pragma: no cover - exercised only in lightweight envs.
    torch = None  # type: ignore[assignment]


_PARAMETER_TO_ATTRIBUTE = MappingProxyType(
    {
        "xyz": "_xyz",
        "f_dc": "_features_dc",
        "f_rest": "_features_rest",
        "opacity": "_opacity",
        "scaling": "_scaling",
        "rotation": "_rotation",
    }
)
_EIGH_CHUNK_SIZE = 32_768
_INITIALIZATIONS = frozenset({"opacity_random", "hilbert_equal_mass"})
_APPEARANCE_PROJECTIONS = frozenset(
    {"coupling_mean", "nearest_incoming_donor"}
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("the 3DGS bridge requires PyTorch")
    return torch


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _nonnegative_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _full_covariances(packed_or_full: Any) -> Any:
    """Expand graphdeco's ``[xx,xy,xz,yy,yz,zz]`` storage when needed."""

    torch_module = _require_torch()
    covariance = packed_or_full.detach()
    if covariance.ndim == 3 and covariance.shape[1:] == (3, 3):
        return 0.5 * (covariance + covariance.transpose(-1, -2))
    if covariance.ndim != 2 or covariance.shape[1] != 6:
        raise ValueError(
            "GaussianModel.get_covariance must return shape (N,6) or (N,3,3)"
        )
    result = torch_module.empty(
        (len(covariance), 3, 3), dtype=covariance.dtype, device=covariance.device
    )
    result[:, 0, 0] = covariance[:, 0]
    result[:, 0, 1] = result[:, 1, 0] = covariance[:, 1]
    result[:, 0, 2] = result[:, 2, 0] = covariance[:, 2]
    result[:, 1, 1] = covariance[:, 3]
    result[:, 1, 2] = result[:, 2, 1] = covariance[:, 4]
    result[:, 2, 2] = covariance[:, 5]
    return result


@dataclass(frozen=True)
class _GaussianSnapshot:
    mixture: ApplicationMixture
    xyz: Any
    covariances: Any
    opacity: Any
    features_dc: Any
    features_rest: Any


def extract_gaussian_model(model: Any) -> _GaussianSnapshot:
    """Take a detached source snapshot from a graphdeco ``GaussianModel``."""

    torch_module = _require_torch()
    required = (
        "get_xyz",
        "get_covariance",
        "get_opacity",
        "_features_dc",
        "_features_rest",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"GaussianModel is missing required fields: {missing}")
    with torch_module.no_grad():
        xyz = model.get_xyz.detach()
        # ``get_covariance`` is analytically ``L L^T``, but graphdeco packs it
        # in float32.  Re-expanding very anisotropic splats can therefore leave
        # tiny negative eigenvalues (roundoff only), while the production
        # principal-root routine deliberately rejects non-PSD inputs.  Use the
        # same scale-local floor as the downstream Bures update once, on the
        # server device, so the NumPy snapshot and Torch projection operate on
        # one consistent SPD source geometry.
        covariance = _project_spd(
            _full_covariances(model.get_covariance()).detach(), 1.0e-6
        )
        opacity = model.get_opacity.detach().reshape(-1)
        features_dc = model._features_dc.detach()
        features_rest = model._features_rest.detach()
    count = int(xyz.shape[0])
    if xyz.shape != (count, 3) or covariance.shape != (count, 3, 3):
        raise ValueError("3DGS geometry must have N means and N full 3x3 covariances")
    if opacity.shape != (count,):
        raise ValueError("3DGS opacity must contain one scalar per Gaussian")
    if features_dc.shape[0] != count or features_rest.shape[0] != count:
        raise ValueError("3DGS SH features must match the Gaussian count")
    if not bool(torch_module.isfinite(xyz).all().item()):
        raise ValueError("3DGS means must be finite")
    if not bool(torch_module.isfinite(covariance).all().item()):
        raise ValueError("3DGS covariances must be finite")
    if not bool(torch_module.isfinite(opacity).all().item()):
        raise ValueError("3DGS activated opacities must be finite")
    if bool((opacity < 0.0).any().item()):
        raise ValueError("3DGS activated opacities must be nonnegative")

    mixture = as_application_mixture(
        xyz.double().cpu().numpy(),
        covariance.double().cpu().numpy(),
        opacity.double().cpu().numpy(),
    )
    return _GaussianSnapshot(
        mixture=mixture,
        xyz=xyz,
        covariances=covariance,
        opacity=opacity,
        features_dc=features_dc,
        features_rest=features_rest,
    )


def _target_from_tensors(means: Any, covariances: Any, count: int) -> ApplicationMixture:
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    return as_application_mixture(
        means.double().cpu().numpy(),
        covariances.double().cpu().numpy(),
        weights,
    )


def _initial_target(
    snapshot: _GaussianSnapshot,
    target_components: int,
    seed: int,
    *,
    initial_original_indices: Any | None = None,
) -> tuple[Any, Any, np.ndarray]:
    source = snapshot.mixture
    if target_components > source.support_size:
        raise ValueError("target_components cannot exceed positive-opacity Gaussians")
    if initial_original_indices is not None:
        original_indices = np.asarray(initial_original_indices)
        if original_indices.shape != (target_components,):
            raise ValueError(
                "initial_original_indices must contain target_components entries"
            )
        if original_indices.dtype.kind not in "iu" or original_indices.dtype.kind == "b":
            raise ValueError("initial_original_indices must contain integers")
        original_indices = np.ascontiguousarray(original_indices, dtype=np.int64)
        if len(np.unique(original_indices)) != target_components:
            raise ValueError("initial_original_indices must be distinct")
        positive_source_indices = np.asarray(source.original_indices, dtype=np.int64)
        positions = np.searchsorted(positive_source_indices, original_indices)
        in_range = positions < len(positive_source_indices)
        if not bool(np.all(in_range)) or not bool(
            np.all(
                positive_source_indices[positions[in_range]]
                == original_indices[in_range]
            )
        ):
            raise ValueError(
                "initial_original_indices must reference positive-opacity Gaussians"
            )
    else:
        if target_components == source.support_size:
            compact_indices = np.arange(source.support_size, dtype=np.int64)
        else:
            compact_indices = np.asarray(
                np.random.default_rng(seed).choice(
                    source.support_size,
                    size=target_components,
                    replace=False,
                    p=source.weights,
                ),
                dtype=np.int64,
            )
        original_indices = np.asarray(
            source.original_indices[compact_indices], dtype=np.int64
        )
    tensor_indices = _require_torch().as_tensor(
        original_indices, device=snapshot.xyz.device, dtype=_require_torch().long
    )
    return (
        snapshot.xyz.index_select(0, tensor_indices).clone(),
        snapshot.covariances.index_select(0, tensor_indices).clone(),
        original_indices,
    )


def _edge_arrays(
    coupling: SparseCoupling,
    source_count: int,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns, masses = coupling.active_triplets()
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    masses = np.asarray(masses, dtype=np.float64)
    if not (rows.shape == columns.shape == masses.shape):
        raise ValueError("coupling COO arrays must be aligned")
    if len(rows) < 1:
        raise ValueError("coupling must contain positive-mass edges")
    if int(rows.min()) < 0 or int(rows.max()) >= source_count:
        raise ValueError("coupling source index is outside the GaussianModel")
    if int(columns.min()) < 0 or int(columns.max()) >= target_count:
        raise ValueError("coupling target index is outside the reduction budget")
    if not bool(np.all(np.isfinite(masses))) or bool(np.any(masses <= 0.0)):
        raise ValueError("active coupling masses must be finite and positive")
    return rows, columns, masses


def _project_spd(matrices: Any, relative_floor: float) -> Any:
    torch_module = _require_torch()
    symmetric = 0.5 * (matrices + matrices.transpose(-1, -2))
    flat = symmetric.reshape(-1, 3, 3)
    output = torch_module.empty_like(flat)
    # cuSOLVER's Jacobi batched eigensolver rejects very large batch counts
    # (the Truck checkpoint contains >2M matrices). Chunking preserves every
    # per-matrix result while bounding both the batch count and workspace.
    for start in range(0, len(flat), _EIGH_CHUNK_SIZE):
        stop = min(start + _EIGH_CHUNK_SIZE, len(flat))
        eigenvalues, eigenvectors = torch_module.linalg.eigh(flat[start:stop])
        scale = eigenvalues.abs().amax(dim=-1, keepdim=True).clamp_min(
            torch_module.finfo(eigenvalues.dtype).tiny
        )
        eigenvalues = eigenvalues.clamp_min(scale * relative_floor)
        reconstructed = (
            eigenvectors * eigenvalues.unsqueeze(-2)
        ) @ eigenvectors.transpose(-1, -2)
        output[start:stop] = 0.5 * (
            reconstructed + reconstructed.transpose(-1, -2)
        )
    return output.reshape_as(symmetric)


def _spd_root(matrices: Any, relative_floor: float, *, inverse: bool = False) -> Any:
    torch_module = _require_torch()
    symmetric = 0.5 * (matrices + matrices.transpose(-1, -2))
    flat = symmetric.reshape(-1, 3, 3)
    output = torch_module.empty_like(flat)
    for start in range(0, len(flat), _EIGH_CHUNK_SIZE):
        stop = min(start + _EIGH_CHUNK_SIZE, len(flat))
        eigenvalues, eigenvectors = torch_module.linalg.eigh(flat[start:stop])
        scale = eigenvalues.abs().amax(dim=-1, keepdim=True).clamp_min(
            torch_module.finfo(eigenvalues.dtype).tiny
        )
        eigenvalues = eigenvalues.clamp_min(scale * relative_floor)
        factors = (
            torch_module.rsqrt(eigenvalues)
            if inverse
            else torch_module.sqrt(eigenvalues)
        )
        reconstructed = (
            eigenvectors * factors.unsqueeze(-2)
        ) @ eigenvectors.transpose(-1, -2)
        output[start:stop] = 0.5 * (
            reconstructed + reconstructed.transpose(-1, -2)
        )
    return output.reshape_as(symmetric)


def _column_masses(
    columns: np.ndarray,
    masses: np.ndarray,
    target_count: int,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    torch_module = _require_torch()
    result = torch_module.zeros(target_count, dtype=dtype, device=device)
    for start in range(0, len(columns), 1_048_576):
        stop = min(start + 1_048_576, len(columns))
        column = torch_module.as_tensor(columns[start:stop], device=device, dtype=torch_module.long)
        mass = torch_module.as_tensor(masses[start:stop], device=device, dtype=dtype)
        result.index_add_(0, column, mass)
    if bool((result <= 0.0).any().item()):
        raise ValueError("balanced coupling left an empty target column")
    return result


def _project_tensor(
    values: Any,
    rows: np.ndarray,
    columns: np.ndarray,
    masses: np.ndarray,
    column_masses: Any,
    target_count: int,
    edge_chunk_size: int,
) -> Any:
    """Reverse-conditionally project any tensor whose first axis is source."""

    torch_module = _require_torch()
    output = torch_module.zeros(
        (target_count, *values.shape[1:]), dtype=values.dtype, device=values.device
    )
    mass_shape = (-1,) + (1,) * (values.ndim - 1)
    denominator_shape = (-1,) + (1,) * (values.ndim - 1)
    for start in range(0, len(rows), edge_chunk_size):
        stop = min(start + edge_chunk_size, len(rows))
        row = torch_module.as_tensor(
            rows[start:stop], device=values.device, dtype=torch_module.long
        )
        column = torch_module.as_tensor(
            columns[start:stop], device=values.device, dtype=torch_module.long
        )
        mass = torch_module.as_tensor(masses[start:stop], device=values.device, dtype=values.dtype)
        contribution = values.index_select(0, row) * mass.reshape(mass_shape)
        output.index_add_(0, column, contribution)
    return output / column_masses.to(values.dtype).reshape(denominator_shape)


def _nearest_incoming_donor_rows(
    source_means: Any,
    source_covariances: Any,
    target_means: Any,
    target_covariances: Any,
    rows: np.ndarray,
    columns: np.ndarray,
    target_count: int,
    edge_chunk_size: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Select one deterministic source donor on every incoming plan column.

    Edges are ranked by the GHAP Eq. (2)-style surrogate geometry cost used
    for this ablation (this is not GHAP Algorithm 2 appearance assignment),

    ``||mu_i-mu_j||^2 + ||Sigma_i-Sigma_j||_F^2``.

    The first chunked pass finds each column's exact minimum cost.  A second
    pass chooses the smallest original source row among all exactly tied
    minima.  The two-pass construction makes the result independent of edge
    storage order and chunk boundaries without materializing all edge costs.
    """

    torch_module = _require_torch()
    edge_chunk_size = _positive_integer(edge_chunk_size, "edge_chunk_size")
    if source_means.ndim != 2 or source_means.shape[1] != 3:
        raise ValueError("source_means must have shape (N,3)")
    if source_covariances.shape != (len(source_means), 3, 3):
        raise ValueError("source_covariances must have shape (N,3,3)")
    if target_means.shape != (target_count, 3):
        raise ValueError("target_means must have shape (M,3)")
    if target_covariances.shape != (target_count, 3, 3):
        raise ValueError("target_covariances must have shape (M,3,3)")
    if rows.shape != columns.shape or rows.ndim != 1 or len(rows) < 1:
        raise ValueError("rows and columns must be aligned nonempty vectors")

    device = source_means.device
    dtype = source_means.dtype
    best_cost = torch_module.full(
        (target_count,), torch_module.inf, dtype=dtype, device=device
    )

    def edge_cost(start: int, stop: int) -> tuple[Any, Any, Any]:
        row = torch_module.as_tensor(
            rows[start:stop], device=device, dtype=torch_module.long
        )
        column = torch_module.as_tensor(
            columns[start:stop], device=device, dtype=torch_module.long
        )
        mean_delta = source_means.index_select(0, row) - target_means.index_select(
            0, column
        )
        covariance_delta = source_covariances.index_select(
            0, row
        ) - target_covariances.index_select(0, column)
        cost = mean_delta.square().sum(dim=1) + covariance_delta.square().sum(
            dim=(1, 2)
        )
        return row, column, cost

    with torch_module.no_grad():
        for start in range(0, len(rows), edge_chunk_size):
            stop = min(start + edge_chunk_size, len(rows))
            _, column, cost = edge_cost(start, stop)
            best_cost.scatter_reduce_(
                0, column, cost, reduce="amin", include_self=True
            )
        # One device-to-host synchronization audits both finite edge costs and
        # complete column coverage.  Avoid synchronizing once per edge chunk;
        # that overhead is material in the million-Gaussian timing ablation.
        if not bool(torch_module.isfinite(best_cost).all().item()):
            raise FloatingPointError(
                "incoming-donor costs must be finite and every target must "
                "have an incoming edge"
            )

        sentinel = int(len(source_means))
        best_row = torch_module.full(
            (target_count,), sentinel, dtype=torch_module.long, device=device
        )
        for start in range(0, len(rows), edge_chunk_size):
            stop = min(start + edge_chunk_size, len(rows))
            row, column, cost = edge_cost(start, stop)
            tied = cost == best_cost.index_select(0, column)
            candidates = torch_module.where(
                tied, row, torch_module.full_like(row, sentinel)
            )
            best_row.scatter_reduce_(
                0, column, candidates, reduce="amin", include_self=True
            )
        if bool((best_row == sentinel).any().item()):
            raise RuntimeError("failed to select an incoming donor for every target")

    return best_row, MappingProxyType(
        {
            "donor_cost": "squared_mean_l2_plus_squared_covariance_frobenius",
            "donor_candidates": "positive_edges_of_final_geometry_coupling",
            "donor_tie_break": "smallest_original_source_index",
            "donor_selection_passes": 2,
            "donor_selection_edge_visits": int(2 * len(rows)),
            "donor_selection_chunk_size": edge_chunk_size,
            "donor_source_indices_are_original_host_indices": True,
        }
    )


def torch_bures_projection(
    source_means: Any,
    source_covariances: Any,
    current_target_covariances: Any,
    coupling: SparseCoupling,
    *,
    maximum_iterations: int = 25,
    relative_tolerance: float = 1.0e-5,
    relative_eigenvalue_floor: float = 1.0e-6,
    edge_chunk_size: int = 262_144,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Chunked CUDA/CPU reverse-conditional full Bures projection for d=3."""

    torch_module = _require_torch()
    maximum_iterations = _positive_integer(maximum_iterations, "maximum_iterations")
    edge_chunk_size = _positive_integer(edge_chunk_size, "edge_chunk_size")
    relative_tolerance = _nonnegative_finite(relative_tolerance, "relative_tolerance")
    relative_eigenvalue_floor = _nonnegative_finite(
        relative_eigenvalue_floor, "relative_eigenvalue_floor"
    )
    if relative_eigenvalue_floor == 0.0:
        raise ValueError("relative_eigenvalue_floor must be positive")
    if source_means.ndim != 2 or source_means.shape[1] != 3:
        raise ValueError("source_means must have shape (N,3)")
    if source_covariances.shape != (len(source_means), 3, 3):
        raise ValueError("source_covariances must have shape (N,3,3)")
    target_count = int(current_target_covariances.shape[0])
    if current_target_covariances.shape != (target_count, 3, 3):
        raise ValueError("current_target_covariances must have shape (M,3,3)")
    rows, columns, masses = _edge_arrays(coupling, len(source_means), target_count)
    column_masses = _column_masses(
        columns,
        masses,
        target_count,
        device=source_means.device,
        dtype=source_means.dtype,
    )

    with torch_module.no_grad():
        means = _project_tensor(
            source_means,
            rows,
            columns,
            masses,
            column_masses,
            target_count,
            edge_chunk_size,
        )
        # Products S^(1/2) C S^(1/2) can have the square of C's condition
        # number. A covariance-sized floor on this product destroys even the
        # singleton fixed point. Compute covariance algebra in float64 and
        # clamp only negative roundoff eigenvalues in its nonnegative root.
        output_covariance_dtype = current_target_covariances.dtype
        work_covariances = source_covariances.to(dtype=torch_module.float64)
        current = _project_spd(
            current_target_covariances.to(dtype=torch_module.float64),
            relative_eigenvalue_floor,
        )
        covariance_column_masses = _column_masses(
            columns, masses, target_count, device=source_means.device,
            dtype=torch_module.float64,
        )
        converged = False
        relative_change = math.inf
        completed = 0
        for iteration in range(1, maximum_iterations + 1):
            root = _spd_root(current, relative_eigenvalue_floor)
            inverse_root = _spd_root(current, relative_eigenvalue_floor, inverse=True)
            average = torch_module.zeros_like(current)
            for start in range(0, len(rows), edge_chunk_size):
                stop = min(start + edge_chunk_size, len(rows))
                row = torch_module.as_tensor(
                    rows[start:stop], device=source_means.device, dtype=torch_module.long
                )
                column = torch_module.as_tensor(
                    columns[start:stop], device=source_means.device, dtype=torch_module.long
                )
                mass = torch_module.as_tensor(
                    masses[start:stop], device=source_means.device, dtype=current.dtype
                )
                selected_root = root.index_select(0, column)
                selected_covariance = work_covariances.index_select(0, row)
                middle = selected_root @ selected_covariance @ selected_root
                middle_root = _spd_root(middle, 0.0)
                alpha = mass / covariance_column_masses.index_select(0, column)
                average.index_add_(0, column, middle_root * alpha[:, None, None])
            proposal = inverse_root @ average @ average @ inverse_root
            proposal = _project_spd(proposal, relative_eigenvalue_floor)
            difference = torch_module.linalg.matrix_norm(
                proposal - current, ord="fro", dim=(-2, -1)
            )
            scale = torch_module.linalg.matrix_norm(
                current, ord="fro", dim=(-2, -1)
            ).clamp_min(torch_module.finfo(current.dtype).tiny)
            relative_change = float((difference / scale).amax().item())
            current = proposal
            completed = iteration
            if relative_change <= relative_tolerance:
                converged = True
                break
    return means, current.to(dtype=output_covariance_dtype), MappingProxyType(
        {
            "barycenter_iterations": completed,
            "barycenter_converged": converged,
            "barycenter_max_relative_change": relative_change,
            "active_edges": int(len(rows)),
            "edge_chunk_size": edge_chunk_size,
            "projection_device": str(source_means.device),
            "projection_dtype": str(source_means.dtype),
            "covariance_work_dtype": "torch.float64",
            "intermediate_root_floor": "negative_roundoff_only",
        }
    )


def matrix_to_quaternion_wxyz(matrices: Any) -> Any:
    """Convert proper rotation matrices to normalized real-first quaternions."""

    torch_module = _require_torch()
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("matrices must have shape (N,3,3)")
    count = len(matrices)
    result = torch_module.empty((count, 4), dtype=matrices.dtype, device=matrices.device)
    diagonal = torch_module.stack(
        (matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 2, 2]), dim=1
    )
    trace = diagonal.sum(dim=1)
    trace_case = trace > 0.0
    eps = torch_module.finfo(matrices.dtype).eps

    if bool(trace_case.any().item()):
        selected = matrices[trace_case]
        scale = 2.0 * torch_module.sqrt((trace[trace_case] + 1.0).clamp_min(eps))
        result[trace_case, 0] = 0.25 * scale
        result[trace_case, 1] = (selected[:, 2, 1] - selected[:, 1, 2]) / scale
        result[trace_case, 2] = (selected[:, 0, 2] - selected[:, 2, 0]) / scale
        result[trace_case, 3] = (selected[:, 1, 0] - selected[:, 0, 1]) / scale

    remaining = ~trace_case
    for axis in range(3):
        case = remaining & (diagonal[:, axis] >= diagonal.max(dim=1).values)
        if not bool(case.any().item()):
            continue
        selected = matrices[case]
        if axis == 0:
            scale = 2.0 * torch_module.sqrt(
                (1.0 + selected[:, 0, 0] - selected[:, 1, 1] - selected[:, 2, 2]).clamp_min(eps)
            )
            result[case, 0] = (selected[:, 2, 1] - selected[:, 1, 2]) / scale
            result[case, 1] = 0.25 * scale
            result[case, 2] = (selected[:, 0, 1] + selected[:, 1, 0]) / scale
            result[case, 3] = (selected[:, 0, 2] + selected[:, 2, 0]) / scale
        elif axis == 1:
            scale = 2.0 * torch_module.sqrt(
                (1.0 + selected[:, 1, 1] - selected[:, 0, 0] - selected[:, 2, 2]).clamp_min(eps)
            )
            result[case, 0] = (selected[:, 0, 2] - selected[:, 2, 0]) / scale
            result[case, 1] = (selected[:, 0, 1] + selected[:, 1, 0]) / scale
            result[case, 2] = 0.25 * scale
            result[case, 3] = (selected[:, 1, 2] + selected[:, 2, 1]) / scale
        else:
            scale = 2.0 * torch_module.sqrt(
                (1.0 + selected[:, 2, 2] - selected[:, 0, 0] - selected[:, 1, 1]).clamp_min(eps)
            )
            result[case, 0] = (selected[:, 1, 0] - selected[:, 0, 1]) / scale
            result[case, 1] = (selected[:, 0, 2] + selected[:, 2, 0]) / scale
            result[case, 2] = (selected[:, 1, 2] + selected[:, 2, 1]) / scale
            result[case, 3] = 0.25 * scale
        remaining = remaining & ~case
    result = result / torch_module.linalg.vector_norm(result, dim=1, keepdim=True).clamp_min(eps)
    return torch_module.where(result[:, :1] < 0.0, -result, result)


def covariance_to_3dgs_parameters(
    covariances: Any,
    *,
    relative_eigenvalue_floor: float = 1.0e-6,
) -> tuple[Any, Any]:
    """Return ``(log_scaling, quaternion_wxyz)`` reproducing covariances."""

    torch_module = _require_torch()
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
        raise ValueError("covariances must have shape (N,3,3)")
    relative_eigenvalue_floor = _nonnegative_finite(
        relative_eigenvalue_floor, "relative_eigenvalue_floor"
    )
    if relative_eigenvalue_floor == 0.0:
        raise ValueError("relative_eigenvalue_floor must be positive")
    symmetric = 0.5 * (covariances + covariances.transpose(-1, -2))
    eigenvalues, rotations = torch_module.linalg.eigh(symmetric)
    scale = eigenvalues.abs().amax(dim=-1, keepdim=True).clamp_min(
        torch_module.finfo(eigenvalues.dtype).tiny
    )
    eigenvalues = eigenvalues.clamp_min(scale * relative_eigenvalue_floor)
    negative_orientation = torch_module.linalg.det(rotations) < 0.0
    rotations = rotations.clone()
    rotations[negative_orientation, :, -1] *= -1.0
    log_scaling = 0.5 * torch_module.log(eigenvalues)
    return log_scaling, matrix_to_quaternion_wxyz(rotations)


def _canonical_plan(
    coupling: SparseCoupling,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce a nonnegative plan to sorted integer edge keys."""

    rows, columns, masses = coupling.active_triplets()
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    masses = np.asarray(masses, dtype=np.float64)
    if len(rows) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    maximum_row = int(rows.max())
    if maximum_row > (np.iinfo(np.int64).max - (target_count - 1)) // target_count:
        raise OverflowError("coupling edge key exceeds int64")
    keys = rows * target_count + columns
    if len(keys) > 1 and bool(np.any(keys[1:] < keys[:-1])):
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        masses = masses[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    return keys[starts], np.add.reduceat(masses, starts)


def _coupling_l1_change(
    first: SparseCoupling,
    second: SparseCoupling,
    target_count: int,
) -> float:
    """Exact L1 change between sparse plans without a Python edge dictionary."""

    first_keys, first_masses = _canonical_plan(first, target_count)
    second_keys, second_masses = _canonical_plan(second, target_count)
    if len(first_keys) == 0 or len(second_keys) == 0:
        return float(first_masses.sum() + second_masses.sum())
    positions = np.searchsorted(second_keys, first_keys)
    in_range = positions < len(second_keys)
    common_first = np.flatnonzero(in_range)
    common_positions = positions[in_range]
    matches = second_keys[common_positions] == first_keys[common_first]
    shared_mass = float(
        np.minimum(
            first_masses[common_first[matches]], second_masses[common_positions[matches]]
        ).sum()
    )
    return max(
        0.0,
        float(first_masses.sum() + second_masses.sum() - 2.0 * shared_mass),
    )


def _materialize_render_tensors(
    snapshot: _GaussianSnapshot,
    means: Any,
    covariances: Any,
    final_coupling: SparseCoupling,
    *,
    appearance_projection: str,
    relative_eigenvalue_floor: float,
    edge_chunk_size: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], float]:
    """Shared final/iteration export; geometry and appearance use one plan."""

    torch_module = _require_torch()
    source_count, target_components = len(snapshot.xyz), len(means)
    rows, columns, masses = _edge_arrays(
        final_coupling, source_count, target_components
    )
    appearance_started = perf_counter()
    appearance_diagnostics: Mapping[str, Any]
    with torch_module.no_grad():
        if appearance_projection == "coupling_mean":
            column_masses = _column_masses(
                columns,
                masses,
                target_components,
                device=snapshot.xyz.device,
                dtype=snapshot.xyz.dtype,
            )
            features_dc = _project_tensor(
                snapshot.features_dc,
                rows,
                columns,
                masses,
                column_masses,
                target_components,
                edge_chunk_size,
            )
            features_rest = _project_tensor(
                snapshot.features_rest,
                rows,
                columns,
                masses,
                column_masses,
                target_components,
                edge_chunk_size,
            )
            activated_opacity = _project_tensor(
                snapshot.opacity[:, None],
                rows,
                columns,
                masses,
                column_masses,
                target_components,
                edge_chunk_size,
            )
            appearance_diagnostics = MappingProxyType(
                {
                    "appearance_projection": "coupling_mean",
                    "appearance_source": "reverse_conditional_coupling_mean",
                }
            )
        else:
            donor_rows, donor_diagnostics = _nearest_incoming_donor_rows(
                snapshot.xyz,
                snapshot.covariances,
                means,
                covariances,
                rows,
                columns,
                target_components,
                edge_chunk_size,
            )
            features_dc = snapshot.features_dc.index_select(0, donor_rows).clone()
            features_rest = snapshot.features_rest.index_select(
                0, donor_rows
            ).clone()
            activated_opacity = snapshot.opacity.index_select(
                0, donor_rows
            )[:, None].clone()
            appearance_diagnostics = MappingProxyType(
                {
                    "appearance_projection": "nearest_incoming_donor",
                    "appearance_source": "one_source_per_target_column",
                    **dict(donor_diagnostics),
                }
            )
        opacity_epsilon = max(
            1.0e-6, float(torch_module.finfo(activated_opacity.dtype).eps)
        )
        activated_opacity = activated_opacity.clamp(
            opacity_epsilon, 1.0 - opacity_epsilon
        )
        raw_opacity = torch_module.log(activated_opacity) - torch_module.log1p(
            -activated_opacity
        )
        log_scaling, rotation = covariance_to_3dgs_parameters(
            covariances,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
        )
    appearance_seconds = perf_counter() - appearance_started

    tensors = MappingProxyType(
        {
            "xyz": means.to(dtype=snapshot.xyz.dtype).contiguous(),
            "f_dc": features_dc.to(dtype=snapshot.features_dc.dtype).contiguous(),
            "f_rest": features_rest.to(dtype=snapshot.features_rest.dtype).contiguous(),
            "opacity": raw_opacity.to(dtype=snapshot.opacity.dtype).contiguous(),
            "scaling": log_scaling.to(dtype=snapshot.xyz.dtype).contiguous(),
            "rotation": rotation.to(dtype=snapshot.xyz.dtype).contiguous(),
        }
    )
    return tensors, appearance_diagnostics, appearance_seconds


@dataclass(frozen=True)
class GaussianModelReduction:
    """Reduced 3DGS tensors plus the coupling and convergence audit."""

    tensors: Mapping[str, Any]
    reduced_mixture: ApplicationMixture
    coupling: SparseCoupling
    initial_source_indices: np.ndarray
    converged: bool
    stop_reason: str
    iterations: int
    history: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]


def reduce_gaussian_model(
    model: Any,
    *,
    target_components: int,
    refinement_iterations: int = 4,
    seed: int = 0,
    initialization: str = "opacity_random",
    maximum_outer_iterations: int = 3,
    outer_relative_tolerance: float = 5.0e-4,
    coupling_l1_tolerance: float = 1.0e-3,
    relative_surrogate_tolerance: float = 5.0e-4,
    barycenter_maximum_iterations: int = 8,
    barycenter_relative_tolerance: float = 5.0e-4,
    relative_eigenvalue_floor: float = 1.0e-6,
    edge_chunk_size: int = 262_144,
    appearance_projection: str = "coupling_mean",
    coupling_solver: (
        Callable[[ApplicationMixture, ApplicationMixture], SparseCoupling] | None
    ) = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint_callback: (
        Callable[[Mapping[str, Any], Mapping[str, Any]], bool | None] | None
    ) = None,
) -> GaussianModelReduction:
    """Reduce a trained 3DGS model to exactly ``target_components`` atoms.

    Normal calls instantiate :class:`MainCouplingSolver` with either ``T=4``
    (the main method) or ``T=0`` (the implicit six-plan mean ablation).  The
    injectable solver exists only to make the graphdeco boundary unit-testable.
    ``initialization="opacity_random"`` preserves the original weighted
    sampling behavior.  ``"hilbert_equal_mass"`` prepares the production
    solver's cached source orders first and selects one distinct source atom
    near the midpoint of every equal-mass bin in its first Hilbert order.

    The defaults are a bounded million-Gaussian pilot configuration (three
    outer updates and eight Bures fixed-point steps per update).  Hitting a
    cap is reported as such; only simultaneous geometry, coupling-L1,
    surrogate-objective, and inner-barycenter stability sets ``converged``.
    An optional read-only callback receives each completed outer iteration;
    it does not change the solver, update, or stopping criteria.
    ``checkpoint_callback(tensors, record)`` receives isolated, renderable
    tensor copies after every completed update. Returning True requests a
    graceful stop; it never marks an unstable iteration as converged.
    Its materialization/copy/I/O time is excluded from the explicit
    ``elapsed_algorithm_seconds`` field, not from total wall time.
    """

    torch_module = _require_torch()
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable or None")
    target_components = _positive_integer(target_components, "target_components")
    if not isinstance(initialization, str) or initialization not in _INITIALIZATIONS:
        raise ValueError(
            f"initialization must be one of {sorted(_INITIALIZATIONS)}"
        )
    if (
        not isinstance(appearance_projection, str)
        or appearance_projection not in _APPEARANCE_PROJECTIONS
    ):
        raise ValueError(
            "appearance_projection must be one of "
            f"{sorted(_APPEARANCE_PROJECTIONS)}"
        )
    maximum_outer_iterations = _positive_integer(
        maximum_outer_iterations, "maximum_outer_iterations"
    )
    outer_relative_tolerance = _nonnegative_finite(
        outer_relative_tolerance, "outer_relative_tolerance"
    )
    coupling_l1_tolerance = _nonnegative_finite(
        coupling_l1_tolerance, "coupling_l1_tolerance"
    )
    relative_surrogate_tolerance = _nonnegative_finite(
        relative_surrogate_tolerance, "relative_surrogate_tolerance"
    )
    snapshot_started = perf_counter()
    snapshot = extract_gaussian_model(model)
    source_count = int(snapshot.xyz.shape[0])
    solver = coupling_solver or MainCouplingSolver(
        refinement_iterations=refinement_iterations
    )
    initial_original_indices = None
    if initialization == "hilbert_equal_mass":
        selector = getattr(solver, "hilbert_equal_mass_initial_indices", None)
        if not callable(selector):
            raise TypeError(
                "hilbert_equal_mass initialization requires a coupling solver "
                "with hilbert_equal_mass_initial_indices(source, M)"
            )
        initial_original_indices = selector(
            snapshot.mixture, target_components
        )
    means, covariances, initial_indices = _initial_target(
        snapshot,
        target_components,
        int(seed),
        initial_original_indices=initial_original_indices,
    )
    target = _target_from_tensors(means, covariances, target_components)
    snapshot_seconds = perf_counter() - snapshot_started

    history: list[Mapping[str, Any]] = []
    converged = False
    stop_reason = "maximum_outer_iterations"
    coupling_seconds = 0.0
    projection_seconds = 0.0
    checkpoint_overhead_seconds = 0.0
    checkpoint_payload = None
    last_coupling_l1_change = math.inf
    last_relative_surrogate_change = math.inf
    coupling_started = perf_counter()
    current_coupling = solver(snapshot.mixture, target)
    coupling_seconds += perf_counter() - coupling_started
    projection_coupling = current_coupling
    scene_scale = torch_module.linalg.vector_norm(
        snapshot.xyz.amax(dim=0) - snapshot.xyz.amin(dim=0)
    ).clamp_min(torch_module.finfo(snapshot.xyz.dtype).tiny)
    for outer_iteration in range(1, maximum_outer_iterations + 1):
        # This is the plan that generates both the next geometry and, if this
        # is the final iteration, its projected appearance attributes.
        projection_coupling = current_coupling
        projection_started = perf_counter()
        proposal_means, proposal_covariances, barycentric = torch_bures_projection(
            snapshot.xyz,
            snapshot.covariances,
            covariances,
            projection_coupling,
            maximum_iterations=barycenter_maximum_iterations,
            relative_tolerance=barycenter_relative_tolerance,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
            edge_chunk_size=edge_chunk_size,
        )
        projection_seconds += perf_counter() - projection_started
        mean_change = float(
            (
                torch_module.linalg.vector_norm(proposal_means - means, dim=1)
                / scene_scale
            ).amax().item()
        )
        covariance_scale = torch_module.linalg.matrix_norm(
            covariances, ord="fro", dim=(-2, -1)
        ).clamp_min(torch_module.finfo(covariances.dtype).tiny)
        covariance_change = float(
            (
                torch_module.linalg.matrix_norm(
                    proposal_covariances - covariances, ord="fro", dim=(-2, -1)
                )
                / covariance_scale
            )
            .amax()
            .item()
        )
        relative_change = max(mean_change, covariance_change)
        proposal_target = _target_from_tensors(
            proposal_means, proposal_covariances, target_components
        )
        # Re-solving here makes the retained/final plan a plan for the updated
        # representatives.  It also gives a real coupling-stability stopping
        # signal, rather than declaring convergence from geometry alone.
        coupling_started = perf_counter()
        refreshed_coupling = solver(snapshot.mixture, proposal_target)
        coupling_seconds += perf_counter() - coupling_started
        coupling_l1_change = _coupling_l1_change(
            current_coupling, refreshed_coupling, target_components
        )
        objective_scale = max(
            abs(float(current_coupling.objective)),
            abs(float(refreshed_coupling.objective)),
            np.finfo(np.float64).tiny,
        )
        relative_surrogate_change = abs(
            float(refreshed_coupling.objective)
            - float(current_coupling.objective)
        ) / objective_scale
        last_coupling_l1_change = coupling_l1_change
        last_relative_surrogate_change = relative_surrogate_change
        history.append(
            MappingProxyType(
                {
                    "outer_iteration": outer_iteration,
                    "coupling_surrogate_objective_before": float(
                        current_coupling.objective
                    ),
                    "coupling_surrogate_objective_after": float(
                        refreshed_coupling.objective
                    ),
                    "relative_surrogate_objective_change": float(
                        relative_surrogate_change
                    ),
                    "coupling_l1_change": float(coupling_l1_change),
                    "coupling_marginal_residual": float(
                        refreshed_coupling.marginal_residual
                    ),
                    "active_edges": int(refreshed_coupling.active_edges),
                    "maximum_relative_mean_change": mean_change,
                    "maximum_relative_covariance_change": covariance_change,
                    "maximum_relative_geometry_change": relative_change,
                    "elapsed_reduction_seconds": perf_counter() - snapshot_started,
                    "elapsed_algorithm_seconds": (
                        perf_counter() - snapshot_started - checkpoint_overhead_seconds
                    ),
                    **dict(barycentric),
                }
            )
        )
        means, covariances = proposal_means, proposal_covariances
        target = proposal_target
        current_coupling = refreshed_coupling
        if progress_callback is not None:
            progress_callback(history[-1])
        stop_requested = False
        if checkpoint_callback is not None:
            checkpoint_started = perf_counter()
            checkpoint_payload = _materialize_render_tensors(
                snapshot, means, covariances, projection_coupling,
                appearance_projection=appearance_projection,
                relative_eigenvalue_floor=relative_eigenvalue_floor,
                edge_chunk_size=edge_chunk_size,
            )
            if snapshot.xyz.is_cuda:
                torch_module.cuda.synchronize(snapshot.xyz.device)
            materialization_seconds = perf_counter() - checkpoint_started
            isolated_tensors = MappingProxyType({
                name: value.detach().clone()
                for name, value in checkpoint_payload[0].items()
            })
            checkpoint_record = MappingProxyType({
                **dict(history[-1]),
                "render_payload_seconds": materialization_seconds,
                "geometry_and_appearance_same_plan": True,
            })
            requested = checkpoint_callback(isolated_tensors, checkpoint_record)
            if requested is not None and not isinstance(requested, bool):
                raise TypeError("checkpoint_callback must return bool or None")
            stop_requested = requested is True
            del isolated_tensors
            if snapshot.xyz.is_cuda:
                torch_module.cuda.synchronize(snapshot.xyz.device)
            checkpoint_overhead_seconds += perf_counter() - checkpoint_started
        if (
            relative_change <= outer_relative_tolerance
            and coupling_l1_change <= coupling_l1_tolerance
            and relative_surrogate_change <= relative_surrogate_tolerance
            and bool(barycentric["barycenter_converged"])
        ):
            converged = True
            stop_reason = "geometry_coupling_and_surrogate_stable"
            break
        if stop_requested:
            stop_reason = "checkpoint_callback_stop_requested"
            break
    # At an iteration cap the look-ahead plan ``current_coupling`` can differ
    # from the plan that actually generated ``means/covariances``.  Returning
    # the latter keeps geometry, SH and opacity one coherent barycentric
    # projection.  The exact L1 difference to the freshly resolved look-ahead
    # plan remains in diagnostics and determines whether convergence is true.
    final_coupling = projection_coupling

    if checkpoint_callback is None:
        tensors, appearance_diagnostics, appearance_seconds = _materialize_render_tensors(
            snapshot, means, covariances, final_coupling,
            appearance_projection=appearance_projection,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
            edge_chunk_size=edge_chunk_size,
        )
    else:
        tensors, appearance_diagnostics, appearance_seconds = checkpoint_payload
    reduced = _target_from_tensors(means, covariances, target_components)
    total_seconds = perf_counter() - snapshot_started
    method = final_coupling.diagnostics.get(
        "method", f"injected_{type(solver).__name__}"
    )
    diagnostics = MappingProxyType(
        {
            "method": method,
            "source_gaussians": source_count,
            "positive_opacity_source_gaussians": snapshot.mixture.support_size,
            "target_gaussians": target_components,
            "exact_budget_preserved": target_components == len(means),
            "target_weights": "uniform_fixed_capacity",
            "source_weights": "normalized_activated_opacity",
            "initialization": initialization,
            "geometry_update": "reverse_conditional_full_bures_barycenter",
            "appearance_update": (
                "reverse_conditional_coupling_projection"
                if appearance_projection == "coupling_mean"
                else "nearest_incoming_donor_copy"
            ),
            "appearance_projection": appearance_projection,
            "appearance_projection_seconds": appearance_seconds,
            **dict(appearance_diagnostics),
            "opacity_projection_domain": "activated_then_inverse_sigmoid",
            "target_ids_resampled_between_iterations": False,
            "uses_dense_mixture_wasserstein_or_lp": False,
            "snapshot_seconds": snapshot_seconds,
            "coupling_seconds": coupling_seconds,
            "barycentric_projection_seconds": projection_seconds,
            "total_reduction_seconds": total_seconds,
            "checkpoint_overhead_seconds": checkpoint_overhead_seconds,
            "checkpoint_callback_enabled": checkpoint_callback is not None,
            "coupling_solver_calls": len(history) + 1,
            "maximum_outer_iterations": maximum_outer_iterations,
            "barycenter_maximum_iterations": barycenter_maximum_iterations,
            "barycenter_relative_tolerance": barycenter_relative_tolerance,
            "active_edges": int(final_coupling.active_edges),
            "coupling_marginal_residual": float(final_coupling.marginal_residual),
            "outer_relative_tolerance": outer_relative_tolerance,
            "coupling_l1_tolerance": coupling_l1_tolerance,
            "relative_surrogate_tolerance": relative_surrogate_tolerance,
            "relative_eigenvalue_floor": relative_eigenvalue_floor,
            "stop_reason": stop_reason,
            "lookahead_coupling_was_resolved_for_final_geometry": True,
            "final_coupling_generated_final_geometry": True,
            "appearance_and_geometry_use_same_coupling": True,
            "final_coupling_l1_change": float(last_coupling_l1_change),
            "final_relative_surrogate_change": float(
                last_relative_surrogate_change
            ),
            "final_geometry_exactly_projected_by_final_coupling": True,
            "final_geometry_plan_consistency": (
                "same_plan_for_geometry_and_attributes; freshly_resolved_"
                "lookahead_plan_used_only_for_convergence_audit"
            ),
        }
    )
    frozen_initial_indices = np.ascontiguousarray(initial_indices.copy())
    frozen_initial_indices.setflags(write=False)
    return GaussianModelReduction(
        tensors=tensors,
        reduced_mixture=reduced,
        coupling=final_coupling,
        initial_source_indices=frozen_initial_indices,
        converged=converged,
        stop_reason=stop_reason,
        iterations=len(history),
        history=tuple(history),
        diagnostics=diagnostics,
    )


def apply_reduction_to_gaussian_model(
    model: Any,
    reduction: GaussianModelReduction,
    *,
    reset_densification_statistics: bool = True,
) -> Any:
    """Install reduced tensors using graphdeco's optimizer-safe replacement."""

    torch_module = _require_torch()
    if not hasattr(model, "replace_tensor_to_optimizer"):
        raise TypeError("GaussianModel must provide replace_tensor_to_optimizer")
    missing = set(_PARAMETER_TO_ATTRIBUTE).difference(reduction.tensors)
    if missing:
        raise ValueError(f"reduction is missing 3DGS tensors: {sorted(missing)}")
    count = int(reduction.tensors["xyz"].shape[0])
    for name, attribute in _PARAMETER_TO_ATTRIBUTE.items():
        tensor = reduction.tensors[name]
        if tensor.shape[0] != count:
            raise ValueError(f"{name} does not preserve the exact target budget")
        replaced = model.replace_tensor_to_optimizer(tensor, name)
        if name not in replaced:
            raise RuntimeError(
                f"replace_tensor_to_optimizer did not return the {name!r} parameter"
            )
        setattr(model, attribute, replaced[name])
    if reset_densification_statistics:
        device = reduction.tensors["xyz"].device
        dtype = reduction.tensors["xyz"].dtype
        model.max_radii2D = torch_module.zeros(count, dtype=dtype, device=device)
        model.xyz_gradient_accum = torch_module.zeros(
            (count, 1), dtype=dtype, device=device
        )
        model.denom = torch_module.zeros((count, 1), dtype=dtype, device=device)
    return model


def reduce_and_apply_gaussian_model(model: Any, **kwargs: Any) -> GaussianModelReduction:
    """One-line integration seam for GHAP's ``subsampling`` call site."""

    reduction = reduce_gaussian_model(model, **kwargs)
    apply_reduction_to_gaussian_model(model, reduction)
    return reduction


__all__ = [
    "GaussianModelReduction",
    "apply_reduction_to_gaussian_model",
    "covariance_to_3dgs_parameters",
    "extract_gaussian_model",
    "matrix_to_quaternion_wxyz",
    "reduce_and_apply_gaussian_model",
    "reduce_gaussian_model",
    "torch_bures_projection",
]
