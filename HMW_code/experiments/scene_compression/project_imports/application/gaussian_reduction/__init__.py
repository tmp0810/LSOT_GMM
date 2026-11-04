"""Balanced-coupling Gaussian-mixture reduction."""

from .reducer import (
    GaussianReductionResult,
    barycentric_projection,
    bures_covariance_barycenter,
    reduce_gaussian_mixture,
    run_balanced_reduction,
)

__all__ = [
    "GaussianReductionResult",
    "barycentric_projection",
    "bures_covariance_barycenter",
    "reduce_gaussian_mixture",
    "run_balanced_reduction",
]
