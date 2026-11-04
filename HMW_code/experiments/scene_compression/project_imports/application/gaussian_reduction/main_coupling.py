"""Fixed production coupling used by the 3DGS reduction application.

This module is intentionally a thin adapter around the current speed-first
NC-HOT implementation.  It fixes the method at radial--Haar ``q=2``, ``H=6``,
``ell=2`` and 30 Hilbert bits.  ``T=4`` is the application method and ``T=0``
is its arithmetic-mean-of-NW-plans control.

The returned ``SparseCoupling.costs`` and ``SparseCoupling.objective`` are
*surrogate* quantities: squared Euclidean costs in the norm-completed compact
``Phi_2`` feature chart and their mass-weighted sum.  They are not exact
Gaussian Bures--Wasserstein edge costs or a dense mixture-Wasserstein LP
objective.  The coupling itself is nevertheless an exactly repaired balanced
plan and can therefore drive the reduction's barycentric projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any

import numpy as np

from application.common.contracts import ApplicationMixture, SparseCoupling
from experiments.nc_hot_bw import (
    Mixture,
    ProjectedOrders,
    compile_fixed_q2_map,
    fixed_projection_bank,
    nc_hot_q2_h9_style_prepared,
    prepare_compact_q2,
    prepare_projected_orders,
)


Q = 2
H = 6
VIEW_DIMENSION = 2
BITS = 30
MAIN_REFINEMENT_ITERATIONS = 4
MEAN_PLAN_REFINEMENT_ITERATIONS = 0
RELATIVE_EPSILON = 1.0 / 8.0
AMBIENT_DIMENSION = 3


def _freeze(array: Any, dtype: Any) -> np.ndarray:
    """Return an immutable, contiguous snapshot for the application boundary."""

    result = np.ascontiguousarray(np.asarray(array, dtype=dtype).copy())
    result.setflags(write=False)
    return result


def _as_core_mixture(mixture: ApplicationMixture) -> Mixture:
    return Mixture(mixture.means, mixture.covariances, mixture.weights)


def _validate_application_mixture(
    mixture: ApplicationMixture, *, name: str
) -> None:
    if not isinstance(mixture, ApplicationMixture):
        raise TypeError(f"{name} must be an ApplicationMixture")
    if mixture.ambient_dimension != AMBIENT_DIMENSION:
        raise ValueError(
            f"{name} must contain d={AMBIENT_DIMENSION} Gaussians for the "
            "fixed 3DGS coupling"
        )
    if name == "source" and any(
        array.flags.writeable
        for array in (
            mixture.means,
            mixture.covariances,
            mixture.weights,
            mixture.original_indices,
        )
    ):
        raise ValueError(
            "source ApplicationMixture arrays must be read-only before they "
            "can back the identity-keyed preparation cache"
        )


@dataclass(frozen=True)
class _PreparedSource:
    owner: ApplicationMixture
    orders: ProjectedOrders
    representation_ms: float
    order_ms: float


class MainCouplingSolver:
    """Callable balanced-coupling solver for iterative 3DGS reduction.

    Parameters
    ----------
    refinement_iterations:
        Must be ``4`` for the main sparse refined coupling or ``0`` for the
        implicit arithmetic mean of the six weighted-NW plans.  No other
        setting is exposed by this application adapter.

    Notes
    -----
    A reduction repeatedly changes only its target representatives.  This
    class therefore caches Stage I (compact representation) and Stage II
    (projected Hilbert orders) for the *same source object*.  Passing another
    source object safely replaces that one-entry cache.  ApplicationMixture
    snapshots are read-only, so identity is a sufficient and allocation-free
    cache key on the intended iterative path.
    """

    def __init__(
        self, *, refinement_iterations: int = MAIN_REFINEMENT_ITERATIONS
    ) -> None:
        if isinstance(refinement_iterations, (bool, np.bool_)):
            raise ValueError("refinement_iterations must be 4 or 0, not bool")
        try:
            selected_iterations = operator.index(refinement_iterations)
        except TypeError as error:
            raise ValueError("refinement_iterations must be 4 or 0") from error
        if selected_iterations not in (
            MAIN_REFINEMENT_ITERATIONS,
            MEAN_PLAN_REFINEMENT_ITERATIONS,
        ):
            raise ValueError("refinement_iterations must be 4 or 0")

        # Both objects are method constants.  Their construction is deliberately
        # outside every online query and never inspects a source/target pair.
        self.compiled_q2 = compile_fixed_q2_map(AMBIENT_DIMENSION)
        self.projection_bank = fixed_projection_bank(
            self.compiled_q2.feature_dimension,
            H,
            VIEW_DIMENSION,
        )
        self.refinement_iterations = selected_iterations
        self._prepared_source: _PreparedSource | None = None
        self._source_cache_builds = 0
        self._source_cache_hits = 0

    @property
    def source_cache_builds(self) -> int:
        return int(self._source_cache_builds)

    @property
    def source_cache_hits(self) -> int:
        return int(self._source_cache_hits)

    def clear_source_cache(self) -> None:
        """Discard the one-entry source cache without changing method constants."""

        self._prepared_source = None

    def prepare_source(self, source: ApplicationMixture) -> bool:
        """Prepare ``source`` if needed and return whether the cache was hit."""

        _validate_application_mixture(source, name="source")
        cached = self._prepared_source
        if cached is not None and cached.owner is source:
            self._source_cache_hits += 1
            return True

        started = perf_counter_ns()
        prepared = prepare_compact_q2(
            _as_core_mixture(source), self.compiled_q2, fast_d3_roots=True
        )
        represented = perf_counter_ns()
        orders = prepare_projected_orders(
            prepared,
            self.projection_bank,
            bits=BITS,
        )
        ordered = perf_counter_ns()
        self._prepared_source = _PreparedSource(
            owner=source,
            orders=orders,
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
        """Select ``M`` distinct source atoms at first-order mass quantiles.

        The first cached projected Hilbert order defines a deterministic
        one-dimensional traversal.  Midpoints of ``M`` equal-mass bins are
        located on that traversal, then minimally shifted along the order to
        make the selected atoms distinct while reserving enough atoms for all
        remaining bins.  The returned values are immutable *original host*
        indices, so zero-weight filtering remains invisible to the 3DGS host.

        Calling this method prepares the source through the normal one-entry
        cache.  A following coupling call on the same source therefore reuses
        both its compact representation and projected orders.
        """

        _validate_application_mixture(source, name="source")
        if isinstance(target_components, (bool, np.bool_)):
            raise ValueError("target_components must be a positive integer")
        try:
            count = operator.index(target_components)
        except TypeError as error:
            raise ValueError(
                "target_components must be a positive integer"
            ) from error
        count = int(count)
        if count < 1 or count > source.support_size:
            raise ValueError(
                "target_components must lie in [1, source.support_size]"
            )

        self.prepare_source(source)
        cached = self._prepared_source
        if cached is None:  # Defensive: prepare_source always fills the cache.
            raise RuntimeError("source preparation cache was not populated")
        first_order = np.asarray(cached.orders.orders[0], dtype=np.int64)
        ordered_weights = np.asarray(source.weights[first_order], dtype=np.float64)
        cumulative = np.cumsum(ordered_weights, dtype=np.float64)
        total = float(cumulative[-1])
        quantiles = (
            (np.arange(count, dtype=np.float64) + 0.5) * (total / count)
        )
        raw_positions = np.searchsorted(cumulative, quantiles, side="left")

        # A heavy atom may contain several quantile midpoints.  Project the
        # nondecreasing raw positions onto strictly increasing feasible
        # positions.  The upper bound leaves one source atom for every later
        # target and guarantees an exact distinct M without resampling.
        selected_positions = np.empty(count, dtype=np.int64)
        lower = 0
        source_count = source.support_size
        for target in range(count):
            upper = source_count - (count - target)
            position = int(raw_positions[target])
            position = max(lower, min(position, upper))
            selected_positions[target] = position
            lower = position + 1

        selected = _freeze(
            source.original_indices[first_order[selected_positions]], np.int64
        )
        if len(selected) != count or len(np.unique(selected)) != count:
            raise RuntimeError(
                "Hilbert equal-mass initialization lost the exact distinct budget"
            )
        return selected

    def __call__(
        self,
        source: ApplicationMixture,
        target: ApplicationMixture,
    ) -> SparseCoupling:
        """Return a sparse balanced source-to-target coupling.

        Exact BW costs are deliberately not computed here.  Geometry updates
        downstream use the returned masses, while this result's edge costs and
        objective retain the online compact-feature surrogate semantics.
        """

        _validate_application_mixture(source, name="source")
        _validate_application_mixture(target, name="target")
        source_cache_hit = self.prepare_source(source)
        cached = self._prepared_source
        if cached is None:  # Defensive: prepare_source always populates it.
            raise RuntimeError("source preparation cache was not populated")

        target_started = perf_counter_ns()
        prepared_target = prepare_compact_q2(
            _as_core_mixture(target), self.compiled_q2, fast_d3_roots=True
        )
        target_represented = perf_counter_ns()
        target_orders = prepare_projected_orders(
            prepared_target,
            self.projection_bank,
            bits=BITS,
        )
        target_ordered = perf_counter_ns()
        target_representation_ms = (target_represented - target_started) / 1.0e6
        target_order_ms = (target_ordered - target_represented) / 1.0e6

        # A cache miss belongs to this cold call; a cache hit contributes zero
        # source preparation work to the online prefix timer.
        source_representation_ms = (
            0.0 if source_cache_hit else cached.representation_ms
        )
        source_order_ms = 0.0 if source_cache_hit else cached.order_ms
        result, timing = nc_hot_q2_h9_style_prepared(
            cached.orders,
            target_orders,
            max_refinement_iterations=self.refinement_iterations,
            relative_epsilon=RELATIVE_EPSILON,
            prefix_timing={
                "feature_ms": source_representation_ms
                + target_representation_ms,
                "order_ms": source_order_ms + target_order_ms,
            },
        )

        rows = np.asarray(result.rows, dtype=np.int64)
        columns = np.asarray(result.columns, dtype=np.int64)
        masses = np.asarray(result.masses, dtype=np.float64)
        costs = np.asarray(result.costs, dtype=np.float64)
        if not (rows.shape == columns.shape == masses.shape == costs.shape):
            raise RuntimeError("core coupling fields have inconsistent shapes")
        if bool(np.any(masses < 0.0)) or not bool(np.all(np.isfinite(masses))):
            raise RuntimeError("core coupling returned invalid masses")
        if bool(np.any(costs < 0.0)) or not bool(np.all(np.isfinite(costs))):
            raise RuntimeError("core coupling returned invalid surrogate costs")

        row_sum = np.bincount(
            rows, weights=masses, minlength=source.support_size
        )
        column_sum = np.bincount(
            columns, weights=masses, minlength=target.support_size
        )
        marginal_residual = max(
            float(np.max(np.abs(row_sum - source.weights))),
            float(np.max(np.abs(column_sum - target.weights))),
        )
        # The core uses a 1e-9 absolute feasibility guard for million-scale
        # sparse plans; observed small problems remain near machine precision.
        if marginal_residual > 1.0e-9:
            raise RuntimeError(
                "main coupling lost balanced marginals: "
                f"{marginal_residual:.3e}"
            )

        surrogate_objective = float(np.dot(masses, costs))
        if not np.isclose(
            surrogate_objective,
            float(result.squared_objective),
            rtol=2.0e-13,
            atol=2.0e-15,
        ):
            raise RuntimeError("core surrogate objective does not match edge sum")

        method = (
            "sa_nc_hot_h6_t4_plan"
            if self.refinement_iterations == MAIN_REFINEMENT_ITERATIONS
            else "sa_nc_hot_h6_t0_mean_plan"
        )
        diagnostics: dict[str, float | int | str | bool] = {
            "method": method,
            "q": Q,
            "h": H,
            "view_dimension": VIEW_DIMENSION,
            "hilbert_bits": BITS,
            "requested_refinement_iterations": self.refinement_iterations,
            "effective_refinement_iterations": int(
                timing["effective_refinement_iterations"]
            ),
            "core_iterations_completed": int(result.iterations_completed),
            "relative_epsilon": RELATIVE_EPSILON,
            "selection_ground": "compact_feature_l2",
            "cost_semantics": "squared_compact_phi2_feature_l2_surrogate",
            "objective_semantics": (
                "mass_weighted_sum_of_squared_compact_phi2_feature_l2_"
                "surrogate_costs"
            ),
            "surrogate_objective": surrogate_objective,
            "reference_surrogate_objective": float(result.reference_objective),
            "costs_are_exact_gaussian_bw": False,
            "objective_is_exact_mixture_wasserstein": False,
            "exact_bw_repricing_performed": False,
            "no_dense_mw_or_lp": True,
            "balanced_marginals": True,
            "marginal_residual": marginal_residual,
            "core_mode": str(result.mode),
            "raw_edges": int(result.raw_edges),
            "union_edges": int(result.union_edges),
            "output_edges": int(result.output_edges),
            "active_edges": int(np.count_nonzero(masses > 0.0)),
            "core_active_edges_above_1e-13": int(result.active_edges),
            "graph_components": int(result.graph_components),
            "cycle_rank": int(result.cycle_rank),
            "pre_repair_marginal_residual": float(
                result.pre_repair_marginal_residual
            ),
            "residual_completion_mass": float(result.residual_mass),
            "source_cache_hit": bool(source_cache_hit),
            "source_cache_builds": int(self._source_cache_builds),
            "source_cache_hits": int(self._source_cache_hits),
            "source_cache_key_semantics": "application_mixture_object_identity",
            "source_cached_representation_ms": float(cached.representation_ms),
            "source_cached_order_ms": float(cached.order_ms),
            "target_representation_ms": float(target_representation_ms),
            "target_order_ms": float(target_order_ms),
            "radial_shape_frame": "fixed_radial_plus_seeded_haar_traceless_q2",
            "projected_view_bank": "fixed_nested_axis_then_haar_two_planes",
        }
        diagnostics.update(
            {
                f"pipeline_{key}": value
                for key, value in timing.items()
                if isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating))
            }
        )

        return SparseCoupling(
            rows=_freeze(source.original_indices[rows], np.int64),
            columns=_freeze(target.original_indices[columns], np.int64),
            masses=_freeze(masses, np.float64),
            costs=_freeze(costs, np.float64),
            source_indices=source.original_indices,
            target_indices=target.original_indices,
            source_weights=source.weights,
            target_weights=target.weights,
            objective=surrogate_objective,
            marginal_residual=marginal_residual,
            diagnostics=MappingProxyType(diagnostics),
        )


__all__ = [
    "AMBIENT_DIMENSION",
    "BITS",
    "H",
    "MAIN_REFINEMENT_ITERATIONS",
    "MEAN_PLAN_REFINEMENT_ITERATIONS",
    "Q",
    "RELATIVE_EPSILON",
    "VIEW_DIMENSION",
    "MainCouplingSolver",
]
