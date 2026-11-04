"""Shared contracts and the paper-faithful NC-HOT-BW candidate backend."""

from .calibration import canonical_test_u2, load_u2, u2_sha256
from .candidate_backend import CandidateBackend
from .contracts import (
    ApplicationMixture,
    CandidateProblem,
    SparseCoupling,
    as_application_mixture,
)

__all__ = [
    "ApplicationMixture",
    "CandidateBackend",
    "CandidateProblem",
    "SparseCoupling",
    "as_application_mixture",
    "canonical_test_u2",
    "load_u2",
    "u2_sha256",
]

