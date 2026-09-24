"""Image I/O, the two upstream example images, and EM fitting."""
from pathlib import Path
import hashlib
import time
from urllib.request import urlopen
import warnings

import numpy as np
from PIL import Image, ImageOps
from sklearn.mixture import GaussianMixture


REFERENCE_IMAGES = {
    "renoir.jpg": "96f8ffa9fa4714ceb9523f8bd75dbca9c39feb77",
    "gauguin.jpg": "16371058bf6c5a4f3164f169dd32bb1817577882",
}


def download_reference_images(directory):
    """Download the original notebook images and verify their Git blob hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in REFERENCE_IMAGES.items():
        destination = directory / name
        if destination.exists():
            data = destination.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/judelo/gmmot/master/im/{name}"
            with urlopen(url, timeout=60) as response:
                data = response.read()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != expected:
            raise ValueError(f"{name}: reference image hash differs from the reviewed upstream asset")
        if not destination.exists():
            destination.write_bytes(data)
    return directory / "renoir.jpg", directory / "gauguin.jpg"


def read_rgb(path, max_side=None):
    """Convert JPEG/PNG/etc. once to 8-bit RGB, then divide by 255 exactly once."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if max_side is not None:
            if max_side < 1:
                raise ValueError("max_side must be positive or null")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255.0


def save_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def fit_gmm(pixels, components, seed, *, fit_pixels=None, max_iter=100, tol=1e-3, reg_covar=1e-6):
    """Same defaults as the reference: full covariance, kmeans, n_init=1.

    Full data are used unless fit_pixels is explicitly supplied (smoke mode).
    The fixed random_state is the sole reproducibility addition to reference EM.
    """
    if components < 1 or components > len(pixels):
        raise ValueError("components must be between one and the number of pixels")
    training = pixels
    if fit_pixels is not None and fit_pixels < len(pixels):
        if fit_pixels < components:
            raise ValueError("fit_pixels must be at least the component count")
        ids = np.random.default_rng(seed).choice(len(pixels), fit_pixels, replace=False)
        training = pixels[ids]
    start = time.perf_counter()
    model = GaussianMixture(n_components=components, covariance_type="full", n_init=1,
                            init_params="kmeans", random_state=seed, max_iter=max_iter,
                            tol=tol, reg_covar=reg_covar).fit(training)
    elapsed = 1000 * (time.perf_counter() - start)
    if not model.converged_:
        warnings.warn("EM reached max_iter; convergence is recorded in metrics.csv", RuntimeWarning)
    return model, elapsed, len(training)


def guided_output(source, mapped, radius=10, epsilon=1e-4):
    """Original optional filter: source + guided_filter(mapped-source, source).

    Filter the UNCLIPPED displacement, independently in each RGB channel.
    """
    from gmmot import guided_filter
    out = np.empty_like(source)
    for channel in range(3):
        out[..., channel] = source[..., channel] + guided_filter(
            mapped[..., channel] - source[..., channel], source[..., channel], radius, epsilon)
    return out
