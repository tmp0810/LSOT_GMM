"""Evaluation projections are separate from Gaussian parameter projections."""
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(function, *, device, repeats, warmups):
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    for _ in range(warmups):
        function()
    times = []
    result = None
    for _ in range(repeats):
        result = None
        synchronize(device)
        start = time.perf_counter()
        result = function()
        synchronize(device)
        times.append(1000 * (time.perf_counter() - start))
    return result, times


def time_summary(times):
    return float(np.mean(times)), float(np.std(times, ddof=1)) if len(times) > 1 else 0.0


@dataclass
class ColorEvaluator:
    source_indices: np.ndarray
    target_indices: np.ndarray
    directions: np.ndarray
    sorted_target: np.ndarray

    @classmethod
    def create(cls, source_pixels, target_pixels, *, samples=4096, projections=128, seed=42):
        if samples < 1 or projections < 1:
            raise ValueError("evaluation budgets must be positive")
        rng = np.random.default_rng(seed)
        count = min(samples, len(source_pixels), len(target_pixels))
        source_ids = rng.choice(len(source_pixels), count, replace=False)
        target_ids = rng.choice(len(target_pixels), count, replace=False)
        directions = rng.normal(size=(projections, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        target_sorted = np.sort(target_pixels[target_ids] @ directions.T, axis=0)
        return cls(source_ids, target_ids, directions, target_sorted)

    def sw2(self, output):
        """Root SW2 on clipped, unquantized RGB; fixed pixels/directions per run."""
        colors = np.clip(output.reshape(-1, 3)[self.source_indices], 0, 1)
        sorted_output = np.sort(colors @ self.directions.T, axis=0)
        return float(np.sqrt(np.mean((sorted_output - self.sorted_target) ** 2)))


def save_comparison(path, source, target, outputs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [("Source", source), ("Target", target), *outputs]
    columns = min(3, len(items))
    rows = math.ceil(len(items) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (name, image) in zip(axes.flat, items):
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(name)
    figure.tight_layout()
    figure.savefig(Path(path), dpi=160, bbox_inches="tight")
    plt.close(figure)
