"""Exact synthetic parameters and notebook-equivalent image preprocessing."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture

from lsot.gaussians import GMM

ASSET_FILE = Path(__file__).with_name("assets") / "notebook_masks.json"
SHAPE_NAMES = ("redcross", "duck", "star", "batman")
NOTEBOOK_COMMIT = "0984edc826b113e35c3260b699a4ff49ab39d25f"
NOTEBOOK_URL = ("https://github.com/judelo/gmmot/blob/" + NOTEBOOK_COMMIT
                + "/python/GMM_OT_Barycenters.ipynb")


@dataclass
class Case:
    name: str
    inputs: list[GMM]
    bounds: tuple[float, float]
    density_grid: int
    metadata: dict


def gaussian_example():
    means = np.array([[-1., -1.], [-1., 1.], [np.sqrt(3) - 1, 0.]])
    covariances = .05 * np.array([
        [[1., .7], [.7, 1.]], [[1., .1], [.1, 1.]], [[1., 0.], [0., 1.]]])
    inputs = [GMM.from_numpy([1.], means[j:j+1], covariances[j:j+1])
              for j in range(3)]
    return Case("gaussian", inputs, (-2., 2.), 50,
                {"source": NOTEBOOK_URL, "source_cell": 3})


def synthetic_example():
    """Preserve the notebook's gmm order: 0, 1, 2, 3 (not definition order)."""
    inputs = [
        GMM.from_numpy([1/3]*3, [[.5, .75], [.5, .25], [.5, .5]],
                       .25*np.array([[[.1, 0], [0, .005]],
                                      [[.1, 0], [0, .005]],
                                      [[.060, .05], [.05, .05]]])),
        GMM.from_numpy([.25]*4, [[.25, .25], [.75, .75],
                                [.25, .75], [.75, .25]],
                       np.repeat((.01*np.eye(2))[None], 4, axis=0)),
        GMM.from_numpy([.25]*4, [[.5, .75], [.5, .25], [.25, .5], [.75, .5]],
                       .25*np.array([[[.1, 0], [0, .005]],
                                      [[.1, 0], [0, .005]],
                                      [[.005, 0], [0, .1]],
                                      [[.005, 0], [0, .1]]])),
        GMM.from_numpy([1/3]*3, [[.8, .7], [.2, .7], [.5, .3]],
                       [[[.02, .01], [.01, .01]],
                        [[.02, -.01], [-.01, .01]],
                        [[.06, 0], [0, .01]]]),
    ]
    return Case("synthetic", inputs, (0., 1.), 50,
                {"source": NOTEBOOK_URL, "source_cells": [8, 10, 12]})


def bilinear_weights(tx, ty):
    if not (0 <= tx <= 1 and 0 <= ty <= 1):
        raise ValueError("tx and ty must lie in [0,1]")
    return np.array([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty])


def grid_nodes(size):
    if size < 2:
        raise ValueError("grid size must be at least two")
    # Notebook display uses gmminterp[i + nb_images*j]: columns are tx.
    return [(row, col, col/(size-1), row/(size-1),
             bilinear_weights(col/(size-1), row/(size-1)))
            for row in range(size) for col in range(size)]


def bundled_masks():
    manifest = json.loads(ASSET_FILE.read_text())
    masks = {}
    for name in SHAPE_NAMES:
        entry = manifest["masks"][name]
        packed = np.frombuffer(zlib.decompress(base64.b64decode(entry["data"])),
                               dtype=np.uint8)
        mask = np.unpackbits(packed, bitorder="big").reshape(manifest["shape"]).astype(bool)
        digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        if digest != entry["mask_sha256"]:
            raise ValueError(f"Corrupt bundled mask: {name}")
        masks[name] = mask
    return masks, manifest


def image_cloud(image):
    """Notebook: u=1-blue; nonzero(u[::-1,:]); X=(column, flipped row).

    Every nonzero pixel has equal mass. No thresholding, resizing, spatial
    normalization, intensity weighting, or random pixel subsampling is used.
    """
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("The notebook expects an RGB/RGBA PNG")
    blue = pixels[..., 2].astype(np.float64)
    if np.issubdtype(pixels.dtype, np.integer):
        blue /= np.iinfo(pixels.dtype).max
    mask = (1-blue) != 0
    rows, cols = np.nonzero(mask[::-1, :])
    if not len(rows):
        raise ValueError("Image has no nonzero foreground pixels")
    return np.column_stack((cols, rows)).astype(np.float64), mask


def save_gmm(path, gmm, **extra):
    path = Path(path)
    temporary = path.with_name(path.stem+".tmp.npz")
    np.savez_compressed(temporary, weights=gmm.weights.detach().cpu().numpy(),
                        means=gmm.means.detach().cpu().numpy(),
                        covariances=gmm.covariances.detach().cpu().numpy(), **extra)
    temporary.replace(path)


def load_gmm(path, *, device="cpu"):
    with np.load(path) as data:
        return GMM.from_numpy(data["weights"], data["means"], data["covariances"],
                              device=device)


def image_example(output, *, image_dir=None, components=12, seed=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    masks, provenance = bundled_masks()
    inputs, entries = [], []
    for index, name in enumerate(SHAPE_NAMES):
        if image_dir is None:
            # Exact foreground support of each bundled recovered mask.
            pixels = np.repeat(np.where(masks[name], 0, 255)[..., None], 3, axis=2).astype(np.uint8)
            origin = "notebook-output-recovered-mask"
        else:
            path = Path(image_dir) / f"{name}.png"
            if not path.exists():
                raise FileNotFoundError(f"Missing original image: {path}")
            pixels = np.asarray(Image.open(path))
            origin = str(path.resolve())
        cloud, mask = image_cloud(pixels)
        if mask.shape != (128, 128):
            raise ValueError(f"{name}: the notebook setting requires a 128x128 image; no automatic resizing is applied")
        digest = hashlib.sha256(cloud.tobytes()).hexdigest()
        settings = {"cloud_sha256": digest, "components": components, "seed": seed,
                    "covariance_type": "full", "n_init": 1, "max_iter": 100,
                    "tol": .001, "reg_covar": 1e-6}
        cache, setting_file = output / f"{name}.npz", output / f"{name}.json"
        if cache.exists() and setting_file.exists() and json.loads(setting_file.read_text()) == settings:
            fitted = load_gmm(cache)
        else:
            fit = GaussianMixture(n_components=components, covariance_type="full",
                                  random_state=seed, n_init=1, max_iter=100,
                                  tol=.001, reg_covar=1e-6).fit(cloud)
            fitted = GMM.from_numpy(fit.weights_, fit.means_, fit.covariances_)
            save_gmm(cache, fitted, em_converged=fit.converged_, em_iterations=fit.n_iter_)
            setting_file.write_text(json.dumps(settings, indent=2)+"\n")
        Image.fromarray(np.repeat(np.where(mask, 0, 255)[..., None], 3, axis=2).astype(np.uint8)).save(output / f"{name}.png")
        np.save(output / f"{name}_cloud.npy", cloud)
        inputs.append(fitted)
        entries.append({"name": name, "input_index": index, "origin": origin,
                        "shape": list(mask.shape), "foreground_pixels": len(cloud), **settings})
    metadata = {"source": NOTEBOOK_URL, "source_cells": [18, 19, 20],
                "original_pngs_supplied": image_dir is not None,
                "data_provenance": provenance["provenance"] if image_dir is None else "User-supplied original PNGs",
                "em_note": "Notebook EM is unseeded; a fixed seed is added for reproducibility.",
                "images": entries}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return Case("images", inputs, (0., 128.), 128, metadata)
