"""Iterative GMM reduction driven by a balanced component coupling.

The source mixture stays fixed.  A smaller set of Gaussian representatives is
initialized by weighted sampling and then updated by alternating

1. a balanced source-to-representative component coupling; and
2. one reverse-conditional Gaussian barycenter per representative.

Unlike free-target-marginal CTD reduction, the representative weights are
fixed capacities.  This distinction is intentional: if the target marginal is
free, zero-entropy CTD reduces to independent nearest-centre assignments and a
balanced coupling solver is no longer part of the algorithm.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from application.common import (
    ApplicationMixture,
    CandidateBackend,
    SparseCoupling,
    as_application_mixture,
)


Array = np.ndarray
CouplingSolver = Callable[[ApplicationMixture, ApplicationMixture], SparseCoupling]
_COVARIANCE_UPDATES = {"bures", "moment_matching"}


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


def _positive_finite(value: Any, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


def _freeze(array: Any, dtype: Any) -> Array:
    result = np.ascontiguousarray(np.asarray(array, dtype=dtype).copy())
    result.setflags(write=False)
    return result


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def _scaled_eigenvalue_floor(values: Array, relative_floor: float) -> float:
    scale = max(
        float(np.max(np.abs(values))),
        np.finfo(np.float64).tiny,
    )
    return max(relative_floor * scale, np.finfo(np.float64).tiny)


def _validate_psd_collection(matrices: Array, name: str) -> None:
    values = np.linalg.eigvalsh(
        0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    )
    scales = np.maximum(
        np.max(np.abs(values), axis=1),
        np.finfo(np.float64).tiny,
    )
    bad = values[:, 0] < -2.0e-11 * scales
    if bool(np.any(bad)):
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{name}[{index}] must be positive semidefinite; "
            f"minimum eigenvalue is {values[index, 0]:.6e}"
        )


def _project_spd(matrix: Array, relative_floor: float) -> Array:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    values = np.maximum(
        values,
        _scaled_eigenvalue_floor(values, relative_floor),
    )
    return _symmetrize((vectors * values) @ vectors.T)


def _regularize_mixture_covariances(
    mixture: ApplicationMixture,
    relative_floor: float,
) -> tuple[ApplicationMixture, Mapping[str, Any]]:
    """Create one consistent SPD snapshot used by all reduction stages."""

    _validate_psd_collection(mixture.covariances, "source.covariances")
    regularized = np.stack(
        [_project_spd(matrix, relative_floor) for matrix in mixture.covariances]
    )
    differences = np.linalg.norm(
        regularized - mixture.covariances,
        axis=(1, 2),
    )
    scales = np.maximum(
        np.linalg.norm(mixture.covariances, axis=(1, 2)),
        np.finfo(np.float64).tiny,
    )
    changed = differences > 32.0 * np.finfo(np.float64).eps * scales
    snapshot = ApplicationMixture(
        means=_freeze(mixture.means, np.float64),
        covariances=_freeze(regularized, np.float64),
        weights=_freeze(mixture.weights, np.float64),
        original_indices=_freeze(mixture.original_indices, np.int64),
        original_size=mixture.original_size,
    )
    return snapshot, MappingProxyType(
        {
            "relative_eigenvalue_floor": float(relative_floor),
            "regularized_covariance_count": int(np.count_nonzero(changed)),
            "maximum_relative_covariance_change": float(
                np.max(differences / scales)
            ),
        }
    )


def _psd_sqrt(matrix: Array, relative_floor: float = 0.0) -> Array:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    floor = (
        _scaled_eigenvalue_floor(values, relative_floor)
        if relative_floor > 0.0
        else 0.0
    )
    values = np.maximum(values, floor)
    return _symmetrize((vectors * np.sqrt(values)) @ vectors.T)


def _inverse_psd_sqrt(matrix: Array, relative_floor: float) -> Array:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    values = np.maximum(
        values,
        _scaled_eigenvalue_floor(values, relative_floor),
    )
    return _symmetrize((vectors * (1.0 / np.sqrt(values))) @ vectors.T)


def _gaussian_bw2(
    mean_a: Array,
    covariance_a: Array,
    mean_b: Array,
    covariance_b: Array,
) -> float:
    root_a = _psd_sqrt(covariance_a)
    middle = _symmetrize(root_a @ covariance_b @ root_a)
    value = float(
        np.dot(mean_a - mean_b, mean_a - mean_b)
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2.0 * np.trace(_psd_sqrt(middle))
    )
    scale = max(
        1.0,
        float(np.trace(covariance_a)),
        float(np.trace(covariance_b)),
    )
    if value < -1.0e-9 * scale:
        raise FloatingPointError(f"Gaussian BW squared became negative: {value}")
    return max(value, 0.0)


def _gaussian_bw2_edges(
    source: ApplicationMixture,
    target: ApplicationMixture,
    rows: Array,
    columns: Array,
    *,
    batch_size: int = 262_144,
) -> Array:
    """Vectorized exact Gaussian BW-squared prices for aligned COO edges.

    The reduction application routinely reprices millions of sparse edges.
    Calling ``_gaussian_bw2`` once per edge makes Python overhead dominate the
    coupling method, so this helper batches the same closed-form calculation.
    ``batch_size`` bounds the temporary ``(edges,d,d)`` arrays.
    """

    edge_rows = np.asarray(rows, dtype=np.int64)
    edge_columns = np.asarray(columns, dtype=np.int64)
    if edge_rows.ndim != 1 or edge_columns.shape != edge_rows.shape:
        raise ValueError("rows and columns must be aligned vectors")
    batch_size = _positive_integer(batch_size, "batch_size")
    output = np.empty(len(edge_rows), dtype=np.float64)
    for start in range(0, len(edge_rows), batch_size):
        stop = min(start + batch_size, len(edge_rows))
        selected_rows = edge_rows[start:stop]
        selected_columns = edge_columns[start:stop]
        covariance_a = source.covariances[selected_rows]
        covariance_b = target.covariances[selected_columns]
        values_a, vectors_a = np.linalg.eigh(covariance_a)
        values_a = np.maximum(values_a, 0.0)
        roots_a = np.matmul(
            vectors_a * np.sqrt(values_a)[:, None, :],
            np.swapaxes(vectors_a, -1, -2),
        )
        middle = np.matmul(
            np.matmul(roots_a, covariance_b), roots_a
        )
        middle = 0.5 * (middle + np.swapaxes(middle, -1, -2))
        middle_values = np.maximum(np.linalg.eigvalsh(middle), 0.0)
        mean_difference = (
            source.means[selected_rows] - target.means[selected_columns]
        )
        squared = (
            np.einsum("ni,ni->n", mean_difference, mean_difference)
            + np.trace(covariance_a, axis1=1, axis2=2)
            + np.trace(covariance_b, axis1=1, axis2=2)
            - 2.0 * np.sum(np.sqrt(middle_values), axis=1)
        )
        scale = np.maximum.reduce(
            (
                np.ones_like(squared),
                np.trace(covariance_a, axis1=1, axis2=2),
                np.trace(covariance_b, axis1=1, axis2=2),
            )
        )
        if bool(np.any(squared < -1.0e-9 * scale)):
            index = int(np.flatnonzero(squared < -1.0e-9 * scale)[0])
            raise FloatingPointError(
                "Gaussian BW squared became negative on edge "
                f"{start + index}: {squared[index]}"
            )
        output[start:stop] = np.maximum(squared, 0.0)
    return output


def _covariance_objective(
    candidate: Array,
    covariances: Array,
    weights: Array,
) -> float:
    zero = np.zeros(candidate.shape[0], dtype=np.float64)
    return float(
        sum(
            float(weight)
            * _gaussian_bw2(zero, covariance, zero, candidate)
            for covariance, weight in zip(covariances, weights)
        )
    )


def bures_covariance_barycenter(
    covariances: Any,
    weights: Any,
    *,
    initialization: Any | None = None,
    maximum_iterations: int = 100,
    relative_tolerance: float = 1.0e-9,
    eigenvalue_floor: float = 1.0e-10,
) -> tuple[Array, Mapping[str, Any]]:
    """Compute a full-SPD Bures barycenter by a safeguarded fixed point."""

    matrices = np.asarray(covariances, dtype=np.float64)
    probability = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("covariances must have shape (n,d,d)")
    if probability.shape != (len(matrices),) or len(matrices) < 1:
        raise ValueError("weights must match a nonempty covariance collection")
    if not bool(np.all(np.isfinite(matrices))) or not bool(
        np.all(np.isfinite(probability))
    ):
        raise ValueError("barycenter inputs must be finite")
    if bool(np.any(probability < 0.0)) or float(np.max(probability)) <= 0.0:
        raise ValueError("barycenter weights must be nonnegative with positive sum")
    maximum_iterations = _positive_integer(maximum_iterations, "maximum_iterations")
    relative_tolerance = _positive_finite(
        relative_tolerance, "relative_tolerance"
    )
    eigenvalue_floor = _positive_finite(eigenvalue_floor, "eigenvalue_floor")

    _validate_psd_collection(matrices, "covariances")
    matrices = np.stack(
        [_project_spd(matrix, eigenvalue_floor) for matrix in matrices]
    )
    probability = probability / float(np.max(probability))
    probability = probability / float(np.sum(probability))
    if initialization is None:
        current = np.tensordot(probability, matrices, axes=(0, 0))
    else:
        current = np.asarray(initialization, dtype=np.float64)
        if current.shape != matrices.shape[1:]:
            raise ValueError("initialization has the wrong covariance shape")
        if not bool(np.all(np.isfinite(current))):
            raise ValueError("initialization must be finite")
        _validate_psd_collection(current[None, :, :], "initialization")
    current = _project_spd(current, eigenvalue_floor)
    current_objective = _covariance_objective(current, matrices, probability)
    converged = False
    stopped_by_non_descent = False
    accepted_iterations = 0
    final_relative_change = math.inf

    for iteration in range(1, maximum_iterations + 1):
        root = _psd_sqrt(current, eigenvalue_floor)
        inverse_root = _inverse_psd_sqrt(current, eigenvalue_floor)
        average = np.zeros_like(current)
        for weight, covariance in zip(probability, matrices):
            if weight > 0.0:
                average += float(weight) * _psd_sqrt(
                    _symmetrize(root @ covariance @ root)
                )
        proposal = _project_spd(
            inverse_root @ average @ average @ inverse_root,
            eigenvalue_floor,
        )
        proposal_objective = _covariance_objective(
            proposal, matrices, probability
        )
        objective_scale = max(1.0, abs(current_objective))
        if proposal_objective > current_objective + 1.0e-10 * objective_scale:
            stopped_by_non_descent = True
            break
        final_relative_change = float(
            np.linalg.norm(proposal - current, ord="fro")
            / max(np.linalg.norm(current, ord="fro"), np.finfo(np.float64).tiny)
        )
        current = proposal
        current_objective = proposal_objective
        accepted_iterations = iteration
        if final_relative_change <= relative_tolerance:
            converged = True
            break

    return _freeze(current, np.float64), MappingProxyType(
        {
            "iterations": int(accepted_iterations),
            "converged": bool(converged),
            "stopped_by_non_descent": bool(stopped_by_non_descent),
            "relative_change": float(final_relative_change),
            "objective": float(current_objective),
        }
    )


def _original_to_compact(mixture: ApplicationMixture) -> dict[int, int]:
    return {
        int(original): compact
        for compact, original in enumerate(mixture.original_indices)
    }


def _positive_triplets(
    coupling: SparseCoupling,
    source: ApplicationMixture,
    target: ApplicationMixture,
) -> tuple[Array, Array, Array]:
    raw_rows = np.asarray(coupling.rows)
    raw_columns = np.asarray(coupling.columns)
    raw_masses = np.asarray(coupling.masses, dtype=np.float64)
    raw_costs = np.asarray(coupling.costs, dtype=np.float64)
    if (
        raw_rows.ndim != 1
        or raw_columns.ndim != 1
        or raw_masses.ndim != 1
        or raw_costs.ndim != 1
        or not (
            raw_rows.shape
            == raw_columns.shape
            == raw_masses.shape
            == raw_costs.shape
        )
    ):
        raise ValueError("coupling edge arrays must be one-dimensional and aligned")
    if raw_rows.dtype.kind not in "iu" or raw_columns.dtype.kind not in "iu":
        raise ValueError("coupling row and column indices must be integers")
    if not bool(np.all(np.isfinite(raw_masses))) or bool(
        np.any(raw_masses < 0.0)
    ):
        raise ValueError("coupling masses must be finite and nonnegative")
    if not bool(np.all(np.isfinite(raw_costs))):
        raise ValueError("coupling costs must be finite")
    coupling_source_indices = np.asarray(coupling.source_indices)
    coupling_target_indices = np.asarray(coupling.target_indices)
    coupling_source_weights = np.asarray(coupling.source_weights, dtype=np.float64)
    coupling_target_weights = np.asarray(coupling.target_weights, dtype=np.float64)
    if not np.array_equal(coupling_source_indices, source.original_indices):
        raise ValueError("coupling source indices do not match the current source")
    if not np.array_equal(coupling_target_indices, target.original_indices):
        raise ValueError("coupling target indices do not match the current target")
    if coupling_source_weights.shape != source.weights.shape or not np.allclose(
        coupling_source_weights, source.weights, atol=5.0e-8, rtol=0.0
    ):
        raise ValueError("coupling source weights do not match the current source")
    if coupling_target_weights.shape != target.weights.shape or not np.allclose(
        coupling_target_weights, target.weights, atol=5.0e-8, rtol=0.0
    ):
        raise ValueError("coupling target weights do not match the current target")

    rows, columns, masses = coupling.active_triplets()
    source_lookup = _original_to_compact(source)
    target_lookup = _original_to_compact(target)
    try:
        compact_rows = np.asarray(
            [source_lookup[int(row)] for row in rows], dtype=np.int64
        )
        compact_columns = np.asarray(
            [target_lookup[int(column)] for column in columns], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("coupling references a filtered or unknown component") from error
    masses = np.asarray(masses, dtype=np.float64)
    row_mass = np.bincount(
        compact_rows, weights=masses, minlength=source.support_size
    )
    column_mass = np.bincount(
        compact_columns, weights=masses, minlength=target.support_size
    )
    residual = max(
        float(np.max(np.abs(row_mass - source.weights))),
        float(np.max(np.abs(column_mass - target.weights))),
    )
    if residual > 5.0e-8:
        raise ValueError(
            f"coupling does not match current source/target marginals: {residual:.3e}"
        )
    return compact_rows, compact_columns, masses


def barycentric_projection(
    source: ApplicationMixture,
    target: ApplicationMixture,
    coupling: SparseCoupling,
    *,
    covariance_update: str = "bures",
    barycenter_maximum_iterations: int = 100,
    barycenter_relative_tolerance: float = 1.0e-9,
    eigenvalue_floor: float = 1.0e-10,
) -> tuple[ApplicationMixture, Mapping[str, Any]]:
    """Update target Gaussians from reverse conditionals of ``coupling``.

    For target component ``j``, the normalized weights are
    ``alpha_ij = pi_ij / target_weight_j``.  Means are Euclidean barycenters;
    covariances are Bures barycenters by default.  ``moment_matching`` is an
    explicit baseline and is not guaranteed to decrease the BW objective.
    """

    if source.ambient_dimension != target.ambient_dimension:
        raise ValueError("source and target dimensions must match")
    if covariance_update not in _COVARIANCE_UPDATES:
        raise ValueError(
            f"covariance_update must be one of {sorted(_COVARIANCE_UPDATES)}"
        )
    eigenvalue_floor = _positive_finite(eigenvalue_floor, "eigenvalue_floor")
    if covariance_update == "bures":
        barycenter_maximum_iterations = _positive_integer(
            barycenter_maximum_iterations,
            "barycenter_maximum_iterations",
        )
        barycenter_relative_tolerance = _positive_finite(
            barycenter_relative_tolerance,
            "barycenter_relative_tolerance",
        )
    _validate_psd_collection(source.covariances, "source.covariances")
    _validate_psd_collection(target.covariances, "target.covariances")
    rows, columns, masses = _positive_triplets(coupling, source, target)
    component_count = target.support_size
    means = np.zeros_like(target.means)
    covariances = np.zeros_like(target.covariances)
    inner_iterations: list[int] = []
    inner_residuals: list[float] = []
    all_converged = True

    order = np.argsort(columns, kind="stable")
    grouped_rows = rows[order]
    grouped_masses = masses[order]
    counts = np.bincount(columns, minlength=component_count)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )

    for column in range(component_count):
        start = int(offsets[column])
        stop = int(offsets[column + 1])
        if start == stop:
            raise RuntimeError("positive fixed target weight received no coupling mass")
        local_rows = grouped_rows[start:stop]
        local_mass = grouped_masses[start:stop]
        probability = local_mass / float(np.sum(local_mass))
        mean = np.tensordot(
            probability, source.means[local_rows], axes=(0, 0)
        )
        means[column] = mean
        if covariance_update == "bures":
            covariance, diagnostics = bures_covariance_barycenter(
                source.covariances[local_rows],
                probability,
                initialization=target.covariances[column],
                maximum_iterations=barycenter_maximum_iterations,
                relative_tolerance=barycenter_relative_tolerance,
                eigenvalue_floor=eigenvalue_floor,
            )
            inner_iterations.append(int(diagnostics["iterations"]))
            inner_residuals.append(float(diagnostics["relative_change"]))
            all_converged = all_converged and bool(diagnostics["converged"])
        else:
            mean_offsets = source.means[local_rows] - mean
            covariance = np.tensordot(
                probability,
                source.covariances[local_rows]
                + np.einsum("ni,nj->nij", mean_offsets, mean_offsets),
                axes=(0, 0),
            )
        covariances[column] = _project_spd(covariance, eigenvalue_floor)

    reduced = ApplicationMixture(
        means=_freeze(means, np.float64),
        covariances=_freeze(covariances, np.float64),
        weights=_freeze(target.weights, np.float64),
        original_indices=_freeze(target.original_indices, np.int64),
        original_size=target.original_size,
    )
    return reduced, MappingProxyType(
        {
            "covariance_update": covariance_update,
            "target_weights_fixed": True,
            "maximum_barycenter_iterations": int(max(inner_iterations, default=0)),
            "maximum_barycenter_relative_change": float(
                max(inner_residuals, default=0.0)
            ),
            "all_barycenters_converged": bool(
                covariance_update != "bures" or all_converged
            ),
        }
    )


def _coupling_objective(
    source: ApplicationMixture,
    target: ApplicationMixture,
    coupling: SparseCoupling,
) -> float:
    rows, columns, masses = _positive_triplets(coupling, source, target)
    costs = _gaussian_bw2_edges(source, target, rows, columns)
    return float(np.dot(masses, costs))


def _reprice_coupling(
    source: ApplicationMixture,
    target: ApplicationMixture,
    coupling: SparseCoupling,
) -> SparseCoupling:
    """Keep a feasible plan/support and price it against moved representatives."""

    source_lookup = _original_to_compact(source)
    target_lookup = _original_to_compact(target)
    try:
        rows = np.asarray(
            [source_lookup[int(row)] for row in coupling.rows], dtype=np.int64
        )
        columns = np.asarray(
            [target_lookup[int(column)] for column in coupling.columns],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("coupling support cannot be reused for this target") from error
    costs = _gaussian_bw2_edges(source, target, rows, columns)
    masses = np.asarray(coupling.masses, dtype=np.float64)
    if costs.shape != masses.shape:
        raise ValueError("coupling edge arrays must have matching shapes")
    objective = float(np.dot(masses, costs))
    diagnostics = dict(coupling.diagnostics)
    diagnostics.update(
        {
            "objective": objective,
            "repriced_after_representative_update": True,
        }
    )
    return SparseCoupling(
        rows=coupling.rows,
        columns=coupling.columns,
        masses=coupling.masses,
        costs=_freeze(costs, np.float64),
        source_indices=source.original_indices,
        target_indices=target.original_indices,
        source_weights=source.weights,
        target_weights=target.weights,
        objective=objective,
        marginal_residual=coupling.marginal_residual,
        diagnostics=MappingProxyType(diagnostics),
    )


def _coupling_l1(first: SparseCoupling, second: SparseCoupling) -> float:
    mass_by_edge: dict[tuple[int, int], float] = {}
    for rows, columns, masses, sign in (
        (*first.active_triplets(), 1.0),
        (*second.active_triplets(), -1.0),
    ):
        for row, column, mass in zip(rows, columns, masses):
            edge = (int(row), int(column))
            mass_by_edge[edge] = mass_by_edge.get(edge, 0.0) + sign * float(mass)
    return float(sum(abs(value) for value in mass_by_edge.values()))


def _representative_shift(
    previous: ApplicationMixture, proposal: ApplicationMixture
) -> tuple[float, float]:
    shifts = np.asarray(
        [
            _gaussian_bw2(mean_a, covariance_a, mean_b, covariance_b)
            for mean_a, covariance_a, mean_b, covariance_b in zip(
                previous.means,
                previous.covariances,
                proposal.means,
                proposal.covariances,
            )
        ],
        dtype=np.float64,
    )
    return float(np.max(shifts)), float(np.dot(previous.weights, shifts))


def _target_count(
    source_count: int,
    reduction_ratio: float | None,
    target_components: int | None,
) -> int:
    if (reduction_ratio is None) == (target_components is None):
        raise ValueError(
            "provide exactly one of reduction_ratio or target_components"
        )
    if target_components is not None:
        count = _positive_integer(target_components, "target_components")
        if count > source_count:
            raise ValueError("target_components cannot exceed source support")
        return count
    assert reduction_ratio is not None
    if not np.isfinite(reduction_ratio) or not (0.0 < reduction_ratio <= 1.0):
        raise ValueError("reduction_ratio must lie in (0, 1]")
    return max(1, min(source_count, int(math.ceil(reduction_ratio * source_count))))


def _initial_target(
    source: ApplicationMixture,
    target_components: int,
    *,
    seed: int,
    initial_indices: Sequence[int] | None,
    target_weights: str | Sequence[float],
) -> tuple[ApplicationMixture, Array, str]:
    if initial_indices is None:
        if target_components == source.support_size:
            selected = np.arange(source.support_size, dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            selected = np.asarray(
                rng.choice(
                    source.support_size,
                    size=target_components,
                    replace=False,
                    p=source.weights,
                ),
                dtype=np.int64,
            )
    else:
        raw_requested = np.asarray(initial_indices)
        if raw_requested.shape != (target_components,):
            raise ValueError("initial_indices must have target_components entries")
        if raw_requested.dtype.kind not in "iu" or raw_requested.dtype.kind == "b":
            raise ValueError("initial_indices must contain integer original indices")
        if raw_requested.dtype.kind == "u" and bool(
            np.any(raw_requested > np.iinfo(np.int64).max)
        ):
            raise ValueError("initial_indices exceed the int64 index range")
        requested = raw_requested.astype(np.int64, copy=False)
        if len(np.unique(requested)) != len(requested):
            raise ValueError("initial_indices must be unique")
        compact_by_original = _original_to_compact(source)
        try:
            selected = np.asarray(
                [compact_by_original[int(index)] for index in requested],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "initial_indices must reference positive-weight original components"
            ) from error

    if isinstance(target_weights, str):
        if target_weights == "uniform":
            if target_components == source.support_size:
                reduced_weights = source.weights[selected].copy()
                weight_rule = "source_at_full_ratio"
            else:
                reduced_weights = np.full(
                    target_components, 1.0 / target_components, dtype=np.float64
                )
                weight_rule = target_weights
        elif target_weights == "sampled":
            reduced_weights = source.weights[selected].copy()
            reduced_weights /= float(np.sum(reduced_weights))
            weight_rule = target_weights
        else:
            raise ValueError("target_weights string must be 'uniform' or 'sampled'")
    else:
        reduced_weights = np.asarray(target_weights, dtype=np.float64)
        if reduced_weights.shape != (target_components,):
            raise ValueError("explicit target_weights have the wrong shape")
        if not bool(np.all(np.isfinite(reduced_weights))) or bool(
            np.any(reduced_weights <= 0.0)
        ):
            raise ValueError("explicit target_weights must be finite and positive")
        reduced_weights = reduced_weights / float(np.max(reduced_weights))
        reduced_weights = reduced_weights / float(np.sum(reduced_weights))
        weight_rule = "explicit"

    target = as_application_mixture(
        source.means[selected],
        source.covariances[selected],
        reduced_weights,
    )
    return target, _freeze(source.original_indices[selected], np.int64), weight_rule


@dataclass(frozen=True)
class GaussianReductionResult:
    """Returned reduced mixture, final coupling, and convergence audit."""

    reduced: ApplicationMixture
    coupling: SparseCoupling
    initial_source_indices: Array
    converged: bool
    stop_reason: str
    iterations: int
    history: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]


def run_balanced_reduction(
    source: ApplicationMixture,
    coupling_solver: CouplingSolver,
    *,
    reduction_ratio: float | None = None,
    target_components: int | None = None,
    seed: int = 0,
    initial_indices: Sequence[int] | None = None,
    target_weights: str | Sequence[float] = "uniform",
    covariance_update: str = "bures",
    maximum_iterations: int = 50,
    relative_objective_tolerance: float = 1.0e-6,
    representative_tolerance: float = 1.0e-8,
    coupling_tolerance: float = 1.0e-8,
    monotonicity_tolerance: float = 1.0e-9,
    barycenter_maximum_iterations: int = 100,
    barycenter_relative_tolerance: float = 1.0e-9,
    eigenvalue_floor: float = 1.0e-10,
) -> GaussianReductionResult:
    """Reduce ``source`` with fixed target masses and repeated couplings."""

    if not callable(coupling_solver):
        raise TypeError("coupling_solver must be callable")
    maximum_iterations = _positive_integer(maximum_iterations, "maximum_iterations")
    outer_tolerances = np.asarray(
        [
            relative_objective_tolerance,
            representative_tolerance,
            coupling_tolerance,
            monotonicity_tolerance,
        ],
        dtype=np.float64,
    )
    if not bool(np.all(np.isfinite(outer_tolerances))) or bool(
        np.any(outer_tolerances < 0.0)
    ):
        raise ValueError("outer tolerances must be nonnegative")
    if covariance_update not in _COVARIANCE_UPDATES:
        raise ValueError(
            f"covariance_update must be one of {sorted(_COVARIANCE_UPDATES)}"
        )
    eigenvalue_floor = _positive_finite(eigenvalue_floor, "eigenvalue_floor")
    if covariance_update == "bures":
        barycenter_maximum_iterations = _positive_integer(
            barycenter_maximum_iterations,
            "barycenter_maximum_iterations",
        )
        barycenter_relative_tolerance = _positive_finite(
            barycenter_relative_tolerance,
            "barycenter_relative_tolerance",
        )

    source, regularization_diagnostics = _regularize_mixture_covariances(
        source, eigenvalue_floor
    )
    target_components = _target_count(
        source.support_size, reduction_ratio, target_components
    )
    current, selected, weight_rule = _initial_target(
        source,
        target_components,
        seed=seed,
        initial_indices=initial_indices,
        target_weights=target_weights,
    )
    current_coupling = coupling_solver(source, current)
    current_objective = _coupling_objective(source, current, current_coupling)
    history: list[Mapping[str, Any]] = [
        MappingProxyType(
            {
                "iteration": 0,
                "accepted": True,
                "objective": float(current_objective),
                "active_coupling_edges": int(current_coupling.active_edges),
            }
        )
    ]
    converged = False
    stop_reason = "maximum_iterations"
    accepted_iterations = 0
    rollback_count = 0
    retained_plan_count = 0

    for iteration in range(1, maximum_iterations + 1):
        proposal, projection_diagnostics = barycentric_projection(
            source,
            current,
            current_coupling,
            covariance_update=covariance_update,
            barycenter_maximum_iterations=barycenter_maximum_iterations,
            barycenter_relative_tolerance=barycenter_relative_tolerance,
            eigenvalue_floor=eigenvalue_floor,
        )
        frozen_objective = _coupling_objective(
            source, proposal, current_coupling
        )
        scale = max(1.0, abs(current_objective))
        max_shift, weighted_shift = _representative_shift(current, proposal)
        if (
            covariance_update == "bures"
            and frozen_objective
            > current_objective + monotonicity_tolerance * scale
        ):
            rollback_count += 1
            stop_reason = "barycentric_update_non_descent"
            history.append(
                MappingProxyType(
                    {
                        "iteration": iteration,
                        "accepted": False,
                        "rollback_reason": stop_reason,
                        "objective_before": float(current_objective),
                        "frozen_plan_objective_after_update": float(
                            frozen_objective
                        ),
                        "maximum_representative_bw2_step": max_shift,
                        **dict(projection_diagnostics),
                    }
                )
            )
            break

        refreshed_coupling = coupling_solver(source, proposal)
        refreshed_objective = _coupling_objective(
            source, proposal, refreshed_coupling
        )
        retained_previous_plan = False
        if (
            covariance_update == "bures"
            and refreshed_objective
            > frozen_objective + monotonicity_tolerance * scale
        ):
            proposal_coupling = _reprice_coupling(
                source, proposal, current_coupling
            )
            proposal_objective = float(proposal_coupling.objective)
            retained_previous_plan = True
            retained_plan_count += 1
        else:
            proposal_coupling = refreshed_coupling
            proposal_objective = refreshed_objective

        relative_decrease = float(
            max(current_objective - proposal_objective, 0.0) / scale
        )
        coupling_l1_change = _coupling_l1(current_coupling, proposal_coupling)
        history.append(
            MappingProxyType(
                {
                    "iteration": iteration,
                    "accepted": True,
                    "objective_before": float(current_objective),
                    "frozen_plan_objective_after_update": float(frozen_objective),
                    "refreshed_candidate_objective": float(refreshed_objective),
                    "objective_after_resolve": float(proposal_objective),
                    "retained_previous_plan": retained_previous_plan,
                    "relative_objective_decrease": relative_decrease,
                    "maximum_representative_bw2_step": max_shift,
                    "weighted_representative_bw2_step": weighted_shift,
                    "coupling_l1_change": coupling_l1_change,
                    "active_coupling_edges": int(proposal_coupling.active_edges),
                    **dict(projection_diagnostics),
                }
            )
        )
        current = proposal
        current_coupling = proposal_coupling
        current_objective = proposal_objective
        accepted_iterations += 1
        if (
            relative_decrease <= relative_objective_tolerance
            and max_shift <= representative_tolerance
            and coupling_l1_change <= coupling_tolerance
            and bool(projection_diagnostics["all_barycenters_converged"])
        ):
            converged = True
            stop_reason = "objective_representatives_and_coupling_stable"
            break

    diagnostics = MappingProxyType(
        {
            "mode": "balanced_fixed_target_weights",
            "source_components": int(source.support_size),
            "target_components": int(current.support_size),
            "reduction_ratio_requested": (
                None if reduction_ratio is None else float(reduction_ratio)
            ),
            "target_components_requested": int(target_components),
            "reduction_ratio_realized": float(
                current.support_size / source.support_size
            ),
            "target_weight_rule": weight_rule,
            "target_weights_fixed": True,
            "covariance_update": covariance_update,
            "relative_objective_tolerance": float(relative_objective_tolerance),
            "representative_bw2_tolerance": float(representative_tolerance),
            "coupling_l1_tolerance": float(coupling_tolerance),
            "final_objective": float(current_objective),
            "rollback_count": int(rollback_count),
            "retained_previous_plan_count": int(retained_plan_count),
            **dict(regularization_diagnostics),
            "boundary": (
                "fixed target capacities keep balanced component coupling "
                "nontrivial; free target weights are the separate CTD baseline"
            ),
        }
    )
    return GaussianReductionResult(
        reduced=current,
        coupling=current_coupling,
        initial_source_indices=selected,
        converged=converged,
        stop_reason=stop_reason,
        iterations=accepted_iterations,
        history=tuple(history),
        diagnostics=diagnostics,
    )


def reduce_gaussian_mixture(
    means: Any,
    covariances: Any,
    weights: Any,
    *,
    u2: Any,
    reduction_ratio: float | None = None,
    target_components: int | None = None,
    seed: int = 0,
    initial_indices: Sequence[int] | None = None,
    target_weights: str | Sequence[float] = "uniform",
    covariance_update: str = "bures",
    h: int = 8,
    bits: int = 20,
    maximum_iterations: int = 50,
    relative_objective_tolerance: float = 1.0e-6,
    representative_tolerance: float = 1.0e-8,
    coupling_tolerance: float = 1.0e-8,
    monotonicity_tolerance: float = 1.0e-9,
    barycenter_maximum_iterations: int = 100,
    barycenter_relative_tolerance: float = 1.0e-9,
    eigenvalue_floor: float = 1.0e-10,
    max_refinement_iterations: int | None = None,
) -> GaussianReductionResult:
    """Convenience entry point using the NC-HOT-BW candidate backend."""

    source = as_application_mixture(means, covariances, weights)
    backend = CandidateBackend(np.asarray(u2, dtype=np.float64), h=h, bits=bits)
    solver: CouplingSolver = partial(
        backend.solve_exact_bw,
        max_refinement_iterations=max_refinement_iterations,
    )
    return run_balanced_reduction(
        source,
        solver,
        reduction_ratio=reduction_ratio,
        target_components=target_components,
        seed=seed,
        initial_indices=initial_indices,
        target_weights=target_weights,
        covariance_update=covariance_update,
        maximum_iterations=maximum_iterations,
        relative_objective_tolerance=relative_objective_tolerance,
        representative_tolerance=representative_tolerance,
        coupling_tolerance=coupling_tolerance,
        monotonicity_tolerance=monotonicity_tolerance,
        barycenter_maximum_iterations=barycenter_maximum_iterations,
        barycenter_relative_tolerance=barycenter_relative_tolerance,
        eigenvalue_floor=eigenvalue_floor,
    )
