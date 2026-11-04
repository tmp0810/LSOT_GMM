"""Stage 1: Gaussian parameters to full Cholesky Euclidean features."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GaussianMixture:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray


def normalize_weights(weights):
    # Same arithmetic/order as nc_hot_bw.representation.normalize_weights.
    values = np.ascontiguousarray(weights, dtype=np.float64).copy()
    if (values.ndim != 1 or len(values) < 1 or np.any(values <= 0)
            or not np.isfinite(values).all()):
        raise ValueError("require finite strictly positive weights")
    total = float(np.sum(values))
    if not np.isfinite(total):
        raise ValueError("weight total overflows float64")
    values /= total
    pivot = int(np.argmax(values))
    values[pivot] += 1.0 - float(np.sum(values))
    return values


def prepare_cholesky(mixture):
    """Return (mu, vecl(chol(Sigma))) and normalized positive weights.

    Coordinates and arithmetic match the historical benchmark on finite SPD
    inputs. The public wrapper additionally rejects nonsymmetric covariances.
    No q=2 compression, learned calibration, or spherical scoring is performed.
    """
    means = np.array(mixture.means, dtype=np.float64, copy=True, order="C")
    cov = np.array(mixture.covariances, dtype=np.float64, copy=True, order="C")
    weights = normalize_weights(mixture.weights)
    if means.ndim != 2 or min(means.shape) < 1:
        raise ValueError("require a nonempty (n,d) mean array")
    n, d = means.shape
    if cov.shape != (n, d, d) or weights.shape != (n,):
        raise ValueError("incompatible Gaussian mixture arrays")
    if not np.isfinite(means).all() or not np.isfinite(cov).all():
        raise ValueError("nonfinite Gaussian parameters")
    if not np.allclose(cov, cov.transpose(0, 2, 1), rtol=1e-12, atol=1e-14):
        raise ValueError("covariances must be symmetric; no silent symmetrization")
    factors = np.linalg.cholesky(cov)
    rows, cols = np.tril_indices(d)
    features = np.ascontiguousarray(np.column_stack((means, factors[:, rows, cols])))
    return features, weights
