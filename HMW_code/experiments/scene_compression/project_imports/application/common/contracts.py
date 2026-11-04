"""Array contracts shared by the GMMflow and RegGS integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _readonly(array: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(array, dtype=dtype).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ApplicationMixture:
    """Validated, normalized Gaussian mixture with source-index provenance."""

    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    original_indices: np.ndarray
    original_size: int

    @property
    def support_size(self) -> int:
        return int(len(self.weights))

    @property
    def ambient_dimension(self) -> int:
        return int(self.means.shape[1])


def as_application_mixture(
    means: Any,
    covariances: Any,
    weights: Any,
) -> ApplicationMixture:
    """Snapshot inputs, filter exact zero weights and normalize total mass.

    Negative weights are rejected.  Exact zeros are removed because the core
    NW transport implementation intentionally requires strictly positive
    atoms; ``original_indices`` maps every returned edge back to its host
    component.  Caller-owned arrays are never modified.
    """

    raw_means = np.asarray(means, dtype=np.float64)
    raw_covariances = np.asarray(covariances, dtype=np.float64)
    raw_weights = np.asarray(weights, dtype=np.float64)
    if raw_means.ndim != 2 or len(raw_means) < 1:
        raise ValueError("means must have nonempty shape (n,d)")
    count, dimension = raw_means.shape
    if raw_covariances.shape != (count, dimension, dimension):
        raise ValueError("covariances must have shape (n,d,d) matching means")
    if raw_weights.shape != (count,):
        raise ValueError("weights must have shape (n,) matching means")
    if not (
        bool(np.all(np.isfinite(raw_means)))
        and bool(np.all(np.isfinite(raw_covariances)))
        and bool(np.all(np.isfinite(raw_weights)))
    ):
        raise ValueError("mixture arrays must contain only finite values")
    if bool(np.any(raw_weights < 0.0)):
        raise ValueError("weights must be nonnegative")
    keep = raw_weights > 0.0
    if not bool(np.any(keep)):
        raise ValueError("at least one component must have positive weight")
    selected_weights = raw_weights[keep]
    total = float(np.sum(selected_weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("positive weights must have a finite positive sum")
    selected_covariances = raw_covariances[keep]
    asymmetry = np.max(
        np.abs(selected_covariances - np.swapaxes(selected_covariances, -1, -2))
    )
    scale = max(1.0, float(np.max(np.abs(selected_covariances))))
    if asymmetry > 3.0e-10 * scale:
        raise ValueError("covariances must be symmetric")
    selected_covariances = 0.5 * (
        selected_covariances + np.swapaxes(selected_covariances, -1, -2)
    )
    return ApplicationMixture(
        means=_readonly(raw_means[keep], np.float64),
        covariances=_readonly(selected_covariances, np.float64),
        weights=_readonly(selected_weights / total, np.float64),
        original_indices=_readonly(np.flatnonzero(keep), np.int64),
        original_size=int(count),
    )


@dataclass(frozen=True)
class CandidateProblem:
    """Prepared candidate graph plus the data needed for host edge pricing."""

    source: ApplicationMixture
    target: ApplicationMixture
    prepared_source: Any
    prepared_target: Any
    graph: Any
    h: int
    bits: int
    u2_sha256: str
    order_bank_sha256: str
    timing_ms: Mapping[str, float]
    diagnostics: Mapping[str, float | int | str]

    @property
    def rows(self) -> np.ndarray:
        """Candidate row indices in the filtered source mixture."""

        return self.graph.rows

    @property
    def columns(self) -> np.ndarray:
        """Candidate column indices in the filtered target mixture."""

        return self.graph.columns


@dataclass(frozen=True)
class SparseCoupling:
    """Restricted balanced transport plan in original host indices."""

    rows: np.ndarray
    columns: np.ndarray
    masses: np.ndarray
    costs: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    source_weights: np.ndarray
    target_weights: np.ndarray
    objective: float
    marginal_residual: float
    diagnostics: Mapping[str, float | int | str | bool]

    @property
    def active_edges(self) -> int:
        return int(np.count_nonzero(self.masses > 0.0))

    def active_triplets(
        self, tolerance: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return original-index COO entries whose mass exceeds tolerance."""

        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        keep = self.masses > tolerance
        return self.rows[keep], self.columns[keep], self.masses[keep]

