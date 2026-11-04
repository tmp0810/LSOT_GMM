"""Source-conditioned spatial-H1 control for 3DGS reduction diagnostics.

This module is deliberately *not* a paper method or a replacement for the
production H6/T4 coupling.  It is a diagnostic control with one transparent
construction:

1. compute an opacity-mass-weighted source centre and one isotropic source RMS;
2. apply that same affine preconditioner to source and target xyz;
3. take one 20-bit, three-dimensional Hilbert order on each side; and
4. match the ordered masses with the existing exact NW scan.

The returned edge costs are the repository's full compact ``Phi_2`` surrogate
costs.  Only the *order* is spatial.  Consequently the coupling can be injected
into ``reduce_gaussian_model`` for a normal Bures projection and 3DGS writeback,
while its diagnostics remain explicitly labelled ``diagnostic_control``.

``reduce_gaussian_model(maximum_outer_iterations=1)`` still resolves a second,
fresh lookahead plan by design.  Do not reuse the old plan to suppress that
call: doing so would falsify ``lookahead_resolved`` and convergence semantics.
For a strict single-solve component timing, invoke this solver once and time
``torch_bures_projection`` separately in a diagnostic driver; use the standard
bridge run when a complete model writeback is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any

import numpy as np

from application.common import ApplicationMixture, SparseCoupling
from experiments.nc_hot_bw import Mixture, compile_fixed_q2_map, prepare_compact_q2
from experiments.nc_hot_bw.candidate_graph_ot import (
    direct_single_order_transport_trusted,
)
from experiments.nc_hot_bw.hilbert import hilbert_order_fast


AMBIENT_DIMENSION = 3
HILBERT_BITS = 20
H = 1
Q = 2
METHOD = "diagnostic_control_spatial_h1"
DEFAULT_DIAGNOSTIC_CHUNK_SIZE = 262_144
SKIP_LOOKAHEAD_TIMING_GUIDANCE = (
    "The standard bridge resolves an initial and a fresh lookahead coupling "
    "even when maximum_outer_iterations=1. For strict one-solve timing, time "
    "SpatialH1DiagnosticControlSolver once and torch_bures_projection "
    "separately; never reuse stale costs or claim lookahead_resolved."
)
_IDENTITY_3D = np.eye(AMBIENT_DIMENSION, dtype=np.float64)
_IDENTITY_3D.setflags(write=False)


def _freeze(array: Any, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(array, dtype=dtype).copy())
    result.setflags(write=False)
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(normalized)


def _validate_mixture(
    mixture: ApplicationMixture, *, name: str, require_readonly: bool = False
) -> None:
    if not isinstance(mixture, ApplicationMixture):
        raise TypeError(f"{name} must be an ApplicationMixture")
    if mixture.ambient_dimension != AMBIENT_DIMENSION:
        raise ValueError(f"{name} must contain d=3 Gaussian components")
    if require_readonly and any(
        array.flags.writeable
        for array in (
            mixture.means,
            mixture.covariances,
            mixture.weights,
            mixture.original_indices,
        )
    ):
        raise ValueError(
            "source arrays must be read-only before identity-keyed caching"
        )


def _as_core_mixture(mixture: ApplicationMixture) -> Mixture:
    return Mixture(mixture.means, mixture.covariances, mixture.weights)


def _source_isotropic_preconditioner(
    source: ApplicationMixture,
) -> tuple[np.ndarray, float, bool]:
    """Return source-mass centre and scalar radius RMS.

    The scale is ``sqrt(E_a[||x-center||^2])``.  Dividing all three axes by this
    same scalar preserves spatial anisotropy and makes the expected squared
    norm of the preconditioned source equal to one.  A unit fallback is used
    only when all positive-mass source means are numerically identical.
    """

    center = np.einsum(
        "i,ij->j", source.weights, source.means, optimize=True
    )
    centered = source.means - center
    squared_radius = np.einsum(
        "ij,ij->i", centered, centered, optimize=True
    )
    mean_squared_radius = float(np.dot(source.weights, squared_radius))
    if not np.isfinite(mean_squared_radius) or mean_squared_radius < 0.0:
        raise RuntimeError("source spatial RMS is not finite and nonnegative")
    rms = float(np.sqrt(max(mean_squared_radius, 0.0)))
    fallback = rms <= np.finfo(np.float64).tiny
    if fallback:
        rms = 1.0
    center = np.ascontiguousarray(center, dtype=np.float64)
    center.setflags(write=False)
    return center, float(rms), bool(fallback)


def _spatial_order(
    means: np.ndarray, center: np.ndarray, rms: float
) -> tuple[np.ndarray, float]:
    normalized = np.ascontiguousarray((means - center) / rms, dtype=np.float64)
    return hilbert_order_fast(normalized, _IDENTITY_3D, bits=HILBERT_BITS)


@dataclass(frozen=True)
class _PreparedSpatialSource:
    owner: ApplicationMixture
    compact: Any
    order: np.ndarray
    center: np.ndarray
    isotropic_rms: float
    degenerate_rms_fallback: bool
    collision_fraction: float
    representation_ms: float
    order_ms: float


class SpatialH1DiagnosticControlSolver:
    """Callable source-to-target H1 diagnostic control.

    The source compact representation and spatial order are cached by immutable
    ``ApplicationMixture`` identity, matching the repeated-target use pattern
    of the 3DGS bridge.  This class has no H6/T4 refinement setting because a
    single NW forest has no cycle-reallocation freedom.
    """

    def __init__(
        self, *, diagnostic_chunk_size: int = DEFAULT_DIAGNOSTIC_CHUNK_SIZE
    ) -> None:
        self.diagnostic_chunk_size = _positive_integer(
            diagnostic_chunk_size, "diagnostic_chunk_size"
        )
        self.compiled_q2 = compile_fixed_q2_map(AMBIENT_DIMENSION)
        self._prepared_source: _PreparedSpatialSource | None = None
        self._source_cache_builds = 0
        self._source_cache_hits = 0

    @property
    def source_cache_builds(self) -> int:
        return int(self._source_cache_builds)

    @property
    def source_cache_hits(self) -> int:
        return int(self._source_cache_hits)

    def clear_source_cache(self) -> None:
        self._prepared_source = None

    def prepare_source(self, source: ApplicationMixture) -> bool:
        """Prepare an immutable source and return whether its cache was hit."""

        _validate_mixture(source, name="source", require_readonly=True)
        cached = self._prepared_source
        if cached is not None and cached.owner is source:
            self._source_cache_hits += 1
            return True

        started = perf_counter_ns()
        compact = prepare_compact_q2(
            _as_core_mixture(source), self.compiled_q2, fast_d3_roots=True
        )
        represented = perf_counter_ns()
        center, rms, fallback = _source_isotropic_preconditioner(source)
        order, collision = _spatial_order(source.means, center, rms)
        ordered = perf_counter_ns()
        frozen_order = _freeze(order, np.int64)
        self._prepared_source = _PreparedSpatialSource(
            owner=source,
            compact=compact,
            order=frozen_order,
            center=center,
            isotropic_rms=rms,
            degenerate_rms_fallback=fallback,
            collision_fraction=float(collision),
            representation_ms=(represented - started) / 1.0e6,
            order_ms=(ordered - represented) / 1.0e6,
        )
        self._source_cache_builds += 1
        return False

    def hilbert_equal_mass_initial_indices(
        self,
        source: ApplicationMixture,
        target_components: int,
    ) -> np.ndarray:
        """Select distinct source atoms at spatial-Hilbert mass quantiles.

        This is the initialization hook expected by ``gsplat_bridge``.  It uses
        the same source-conditioned three-dimensional order as this control,
        so the following first coupling call hits the source cache.
        """

        count = _positive_integer(target_components, "target_components")
        _validate_mixture(source, name="source", require_readonly=True)
        if count > source.support_size:
            raise ValueError(
                "target_components cannot exceed positive source support"
            )
        self.prepare_source(source)
        cached = self._prepared_source
        if cached is None:
            raise RuntimeError("source preparation cache was not populated")
        ordered_weights = source.weights[cached.order]
        cumulative = np.cumsum(ordered_weights, dtype=np.float64)
        quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
        raw_positions = np.searchsorted(cumulative, quantiles, side="left")

        selected_positions = np.empty(count, dtype=np.int64)
        lower = 0
        for target_index in range(count):
            upper = source.support_size - (count - target_index)
            position = max(lower, min(int(raw_positions[target_index]), upper))
            selected_positions[target_index] = position
            lower = position + 1
        selected = _freeze(
            source.original_indices[cached.order[selected_positions]], np.int64
        )
        if len(np.unique(selected)) != count:
            raise RuntimeError("spatial Hilbert initialization lost exact budget")
        return selected

    def __call__(
        self, source: ApplicationMixture, target: ApplicationMixture
    ) -> SparseCoupling:
        """Return one exactly balanced spatial-H1 diagnostic coupling."""

        _validate_mixture(source, name="source", require_readonly=True)
        _validate_mixture(target, name="target")
        total_started = perf_counter_ns()
        source_cache_hit = self.prepare_source(source)
        cached = self._prepared_source
        if cached is None:
            raise RuntimeError("source preparation cache was not populated")

        target_started = perf_counter_ns()
        target_compact = prepare_compact_q2(
            _as_core_mixture(target), self.compiled_q2, fast_d3_roots=True
        )
        target_represented = perf_counter_ns()
        target_order, target_collision = _spatial_order(
            target.means, cached.center, cached.isotropic_rms
        )
        target_ordered = perf_counter_ns()

        solve_started = perf_counter_ns()
        result = direct_single_order_transport_trusted(
            cached.compact.weights,
            cached.order,
            cached.compact.features,
            target_compact.weights,
            target_order,
            target_compact.features,
            requested_refinement_iterations=0,
            relative_epsilon=1.0 / 8.0,
        )
        solved = perf_counter_ns()

        rows = np.asarray(result.rows, dtype=np.int64)
        columns = np.asarray(result.columns, dtype=np.int64)
        masses = np.asarray(result.masses, dtype=np.float64)
        costs = np.asarray(result.costs, dtype=np.float64)
        if not (rows.shape == columns.shape == masses.shape == costs.shape):
            raise RuntimeError("direct H1 result fields have inconsistent shapes")
        if bool(np.any(masses <= 0.0)) or not bool(np.all(np.isfinite(masses))):
            raise RuntimeError("direct H1 result contains invalid active masses")
        if bool(np.any(costs < 0.0)) or not bool(np.all(np.isfinite(costs))):
            raise RuntimeError("direct H1 result contains invalid surrogate costs")

        row_sum = np.bincount(rows, weights=masses, minlength=source.support_size)
        column_sum = np.bincount(
            columns, weights=masses, minlength=target.support_size
        )
        source_residual = float(np.max(np.abs(row_sum - source.weights)))
        target_residual = float(np.max(np.abs(column_sum - target.weights)))
        marginal_residual = max(source_residual, target_residual)
        if marginal_residual > 3.0e-12:
            raise RuntimeError(
                "spatial H1 diagnostic control lost balanced marginals: "
                f"{marginal_residual:.3e}"
            )

        diagnostic_started = perf_counter_ns()
        mean_objective = 0.0
        shape_objective = 0.0
        maximum_cost_error = 0.0
        for start in range(0, len(rows), self.diagnostic_chunk_size):
            stop = min(start + self.diagnostic_chunk_size, len(rows))
            delta = (
                cached.compact.features[rows[start:stop]]
                - target_compact.features[columns[start:stop]]
            )
            mean_squared = np.einsum(
                "ij,ij->i", delta[:, :3], delta[:, :3], optimize=True
            )
            shape_squared = np.einsum(
                "ij,ij->i", delta[:, 3:], delta[:, 3:], optimize=True
            )
            block_mass = masses[start:stop]
            mean_objective += float(np.dot(block_mass, mean_squared))
            shape_objective += float(np.dot(block_mass, shape_squared))
            reconstructed = mean_squared + shape_squared
            maximum_cost_error = max(
                maximum_cost_error,
                float(np.max(np.abs(reconstructed - costs[start:stop]))),
            )
        objective = float(np.dot(masses, costs))
        reconstructed_objective = mean_objective + shape_objective
        objective_scale = max(1.0, abs(objective), abs(reconstructed_objective))
        if abs(objective - reconstructed_objective) > 2.0e-12 * objective_scale:
            raise RuntimeError("compact Phi_2 cost decomposition is inconsistent")
        diagnosed = perf_counter_ns()

        source_representation_ms = (
            0.0 if source_cache_hit else cached.representation_ms
        )
        source_order_ms = 0.0 if source_cache_hit else cached.order_ms
        target_representation_ms = (target_represented - target_started) / 1.0e6
        target_order_ms = (target_ordered - target_represented) / 1.0e6
        solve_ms = (solved - solve_started) / 1.0e6
        diagnostic_ms = (diagnosed - diagnostic_started) / 1.0e6
        diagnostics: dict[str, float | int | str | bool] = {
            "method": METHOD,
            "method_role": "diagnostic_control",
            "is_diagnostic_control": True,
            "is_production_h6_t4": False,
            "h": H,
            "q": Q,
            "hilbert_dimension": AMBIENT_DIMENSION,
            "hilbert_bits": HILBERT_BITS,
            "ordering_ground": "source_preconditioned_xyz_only",
            "common_preconditioner": True,
            "preconditioner_fit_on": "source_normalized_opacity_mass",
            "isotropic_rms_semantics": "sqrt(E_source[||x-center||^2])",
            "source_center_x": float(cached.center[0]),
            "source_center_y": float(cached.center[1]),
            "source_center_z": float(cached.center[2]),
            "source_isotropic_rms": float(cached.isotropic_rms),
            "degenerate_rms_unit_fallback": cached.degenerate_rms_fallback,
            "source_hilbert_collision_fraction": cached.collision_fraction,
            "target_hilbert_collision_fraction": float(target_collision),
            "transport": "single_order_northwest_corner_scan",
            "direction": "source_to_target",
            "balanced_marginals": True,
            "source_marginal_residual": source_residual,
            "target_marginal_residual": target_residual,
            "marginal_residual": marginal_residual,
            "active_edges": int(len(masses)),
            "source_components": source.support_size,
            "target_components": target.support_size,
            "raw_edges": int(result.raw_edges),
            "union_edges": int(result.union_edges),
            "output_edges": int(result.output_edges),
            "core_active_edges_above_1e-13": int(result.active_edges),
            "graph_components": int(result.graph_components),
            "cycle_rank": 0,
            "core_mode": str(result.mode),
            "requested_refinement_iterations": 0,
            "effective_refinement_iterations": 0,
            "pre_repair_marginal_residual": float(
                result.pre_repair_marginal_residual
            ),
            "residual_completion_mass": float(result.residual_mass),
            "cost_semantics": "squared_compact_phi2_feature_l2_surrogate",
            "cost_ordering_relation": "full_phi2_cost_spatial_xyz_order_only",
            "costs_are_exact_gaussian_bw": False,
            "objective_is_exact_mixture_wasserstein": False,
            "exact_bw_repricing_performed": False,
            "surrogate_objective": objective,
            "mean_xyz_surrogate_objective": mean_objective,
            "shape_surrogate_objective": shape_objective,
            "normalized_xyz_ordering_objective": (
                mean_objective / (cached.isotropic_rms * cached.isotropic_rms)
            ),
            "mean_xyz_objective_fraction": (
                mean_objective / objective if objective > 0.0 else 0.0
            ),
            "shape_objective_fraction": (
                shape_objective / objective if objective > 0.0 else 0.0
            ),
            "maximum_edge_cost_reconstruction_error": maximum_cost_error,
            "source_cache_hit": bool(source_cache_hit),
            "source_cache_builds": int(self._source_cache_builds),
            "source_cache_hits": int(self._source_cache_hits),
            "source_representation_ms_online": source_representation_ms,
            "source_order_ms_online": source_order_ms,
            "target_representation_ms": target_representation_ms,
            "target_order_ms": target_order_ms,
            "direct_nw_solve_ms": solve_ms,
            "post_solve_cost_diagnostic_ms": diagnostic_ms,
            "online_total_ms": (diagnosed - total_started) / 1.0e6,
            "no_dense_cost_or_plan": True,
            "skip_lookahead_timing_guidance": SKIP_LOOKAHEAD_TIMING_GUIDANCE,
        }
        return SparseCoupling(
            rows=_freeze(source.original_indices[rows], np.int64),
            columns=_freeze(target.original_indices[columns], np.int64),
            masses=_freeze(masses, np.float64),
            costs=_freeze(costs, np.float64),
            source_indices=source.original_indices,
            target_indices=target.original_indices,
            source_weights=source.weights,
            target_weights=target.weights,
            objective=objective,
            marginal_residual=marginal_residual,
            diagnostics=MappingProxyType(diagnostics),
        )


__all__ = [
    "AMBIENT_DIMENSION",
    "H",
    "HILBERT_BITS",
    "METHOD",
    "SKIP_LOOKAHEAD_TIMING_GUIDANCE",
    "SpatialH1DiagnosticControlSolver",
]
