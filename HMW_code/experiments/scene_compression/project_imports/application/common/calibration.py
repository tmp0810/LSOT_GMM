"""Loading and provenance helpers for the frozen q=2 shape subspace."""

from __future__ import annotations

import hashlib
import operator
from pathlib import Path

import numpy as np


def u2_sha256(u2: np.ndarray) -> str:
    """Return the content hash used in application manifests."""

    array = np.ascontiguousarray(np.asarray(u2, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_u2(u2: np.ndarray, ambient_dimension: int | None) -> np.ndarray:
    matrix = np.asarray(u2, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("u2 must have shape (d*(d+1)//2, 2)")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("u2 must contain only finite values")
    if ambient_dimension is not None:
        expected = ambient_dimension * (ambient_dimension + 1) // 2
        if matrix.shape != (expected, 2):
            raise ValueError(
                f"u2 must have shape ({expected}, 2) for d={ambient_dimension}"
            )
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(2), rtol=0.0, atol=3.0e-10):
        raise ValueError("u2 columns must be orthonormal")
    result = np.ascontiguousarray(matrix.copy())
    result.setflags(write=False)
    return result


def load_u2(
    path: str | Path,
    *,
    ambient_dimension: int | None = None,
    key: str = "u2",
) -> np.ndarray:
    """Load a frozen calibration matrix from ``.npy`` or ``.npz``.

    Formal experiments must save the train/calibration scene IDs, seed and
    returned SHA-256 next to this matrix.  Loading never fits U2 from the
    current query, which prevents accidental test leakage.
    """

    location = Path(path)
    if location.suffix == ".npy":
        matrix = np.load(location, allow_pickle=False)
    elif location.suffix == ".npz":
        with np.load(location, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{location} does not contain array {key!r}")
            matrix = archive[key]
    else:
        raise ValueError("U2 path must end in .npy or .npz")
    return _validate_u2(matrix, ambient_dimension)


def canonical_test_u2(ambient_dimension: int) -> np.ndarray:
    """Create a deterministic U2 for smoke tests only.

    This is not a fitted subspace and must not be used for paper results.
    It simply selects the first two coordinates of the symmetric chart.
    """

    if isinstance(ambient_dimension, (bool, np.bool_)):
        raise ValueError("ambient_dimension must be a positive integer")
    try:
        ambient_dimension = operator.index(ambient_dimension)
    except TypeError as error:
        raise ValueError("ambient_dimension must be a positive integer") from error
    if ambient_dimension < 1:
        raise ValueError("ambient_dimension must be a positive integer")
    shape_dimension = ambient_dimension * (ambient_dimension + 1) // 2
    if shape_dimension < 2:
        raise ValueError("q=2 requires ambient_dimension >= 2")
    matrix = np.zeros((shape_dimension, 2), dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    matrix.setflags(write=False)
    return matrix
