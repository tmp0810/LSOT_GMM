"""Exemplars, ADSN, and overlapping RGB patches for a single scale."""
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn.functional import fold


ASSETS = Path(__file__).with_name("assets") / "stori_textures.json"


def texture_path(name, image_dir, *, download=True):
    """Resolve a local file or a pinned StoRI image in the download cache."""
    explicit = Path(name).expanduser()
    if explicit.is_file():
        return explicit, str(explicit.resolve())
    manifest = json.loads(ASSETS.read_text())
    known = manifest["files"]
    path = Path(image_dir) / name
    if name not in known:
        if path.is_file():
            return path, str(path.resolve())
        raise FileNotFoundError(f"Unknown texture {name!r}; pass a local image or a StoRI filename")

    url = (f"https://raw.githubusercontent.com/storimaging/Images/{manifest['revision']}/"
           f"{manifest['path']}/{name}")
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == known[name]:
        return path, url
    if not download:
        raise FileNotFoundError(f"Missing or changed StoRI image: {path}. Supply a local path or allow downloads.")
    try:
        with urlopen(url, timeout=30) as response:
            data = response.read(16 * 1024 * 1024 + 1)
    except OSError as exc:
        raise OSError(f"Could not download {name} from {url}. Use --images with a local file.") from exc
    if hashlib.sha256(data).hexdigest() != known[name]:
        raise ValueError(f"Checksum mismatch for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        file.write(data)
        temporary = Path(file.name)
    temporary.replace(path)
    return path, url


def load_rgb(path, max_side):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb.size
        if max_side is not None and max(width, height) > max_side:
            ratio = max_side / max(width, height)
            rgb = rgb.resize((max(1, round(width * ratio)), max(1, round(height * ratio))),
                             resample=Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.float64) / 255., (height, width)


def adsn(image, seed):
    """Stationary Gaussian synthesis with the exemplar's texton.

    As in StoRI, use one scalar noise field for all color channels and a
    padded convolution followed by a crop. Do not clip before fitting.
    """
    h, w, _ = image.shape
    mean = image.mean(axis=(0, 1))
    texton = (image - mean) / np.sqrt(h * w)
    padded = (2 * h, 2 * w)
    noise = np.random.default_rng(seed).standard_normal(padded)
    filtered = np.fft.rfft2(noise, axes=(0, 1))[..., None] * np.fft.rfft2(
        texton, s=padded, axes=(0, 1))
    return mean + np.fft.irfft2(filtered, s=padded, axes=(0, 1))[:h, :w]


def extract_patches(image, patch_size):
    """Return stride-one valid patches, row-major positions and CHW vectors."""
    if image.ndim != 3 or image.shape[2] != 3 or not 1 <= patch_size <= min(image.shape[:2]):
        raise ValueError("Patch size must fit an RGB image")
    views = np.lib.stride_tricks.sliding_window_view(image, (patch_size, patch_size), axis=(0, 1))
    return np.ascontiguousarray(views).reshape(-1, 3 * patch_size**2)


def aggregate_patches(patches, shape, patch_size):
    """Put each patch back at its source position and divide by overlap."""
    h, w = shape
    positions = (h - patch_size + 1) * (w - patch_size + 1)
    if min(h, w) < patch_size or patch_size < 1 or patches.shape != (positions, 3 * patch_size**2):
        raise ValueError("Patch array does not match the output shape")
    summed = fold(patches.T.unsqueeze(0), shape, patch_size)
    counts = fold(patches.new_ones(1, patch_size**2, positions), shape, patch_size)
    return (summed / counts)[0].permute(1, 2, 0).contiguous()


def save_rgb(path, image):
    pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)
