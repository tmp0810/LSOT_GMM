"""Patch distribution and spatial statistics of reconstructed images."""
import numpy as np
import ot

from .data import extract_patches


def lag_covariances(image):
    """RGB cross-covariance at zero and valid signed spatial lags, no wrap."""
    h, w, _ = image.shape
    centered = image - image.mean(axis=(0, 1))
    offsets = [(0, 0)] + [(dy, dx) for r in (1, 2, 4, 8, 16)
                          for dy, dx in ((0, r), (r, 0), (r, r), (r, -r))
                          if dy < h and abs(dx) < w]
    values = []
    for dy, dx in offsets:
        a = centered[:h-dy, max(-dx, 0):w-max(dx, 0)].reshape(-1, 3)
        b = centered[dy:, max(dx, 0):w-max(-dx, 0)].reshape(-1, 3)
        values.append(a.T @ b / len(a))
    return np.stack(values)


class TextureMetrics:
    """Evaluate a generated RGB image, using fixed samples across methods/K."""
    def __init__(self, exemplar, target_patches, patch_size, seed, *, sw_samples, w_samples, directions):
        self.patch_size = patch_size
        self.n_sw = min(len(target_patches), sw_samples)
        self.n_w = min(len(target_patches), w_samples)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        # Independent target/output sample locations, held fixed across methods.
        sample_count = max(self.n_sw, self.n_w)
        take = rng.choice(len(target_patches), size=sample_count, replace=False)
        self.indices = rng.choice(len(target_patches), size=sample_count, replace=False)
        target = target_patches[take]
        self.directions = rng.standard_normal((directions, target.shape[1]))
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)
        self.target_projected = np.sort(target[:self.n_sw] @ self.directions.T, axis=0)
        self.target_w = np.ascontiguousarray(target[:self.n_w])
        self.uniform = np.full(self.n_w, 1. / self.n_w)
        self.target_cov = lag_covariances(exemplar)

    def __call__(self, image):
        patches = extract_patches(image, self.patch_size)[self.indices]
        projected = np.sort(patches[:self.n_sw] @ self.directions.T, axis=0)
        delta = projected - self.target_projected
        cost = ot.dist(np.ascontiguousarray(patches[:self.n_w]), self.target_w, metric="sqeuclidean")
        w2_squared = ot.emd2(self.uniform, self.uniform, cost, numThreads=1)
        cov_diff = lag_covariances(image) - self.target_cov
        return dict(patch_sw2=float(np.sqrt(np.mean(delta**2))),
                    patch_w2=float(np.sqrt(max(w2_squared, 0))),
                    autocov_rmse=float(np.sqrt(np.mean(cov_diff**2))))
