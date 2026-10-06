"""Individual time panels and MW2-paper-style interpolation strips."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TIMES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def time_tag(t):
    tag = f"{t:.12g}"
    return tag if "." in tag else tag + ".0"


def validate_times(times):
    values = np.asarray(times, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1) or np.any(np.diff(values) <= 0)
            or len({time_tag(t) for t in values}) != len(values)):
        raise ValueError("times must be finite, distinct, increasing values in [0, 1]")
    return values


def _panel(axis, case, x, values, t, vmax, *, axis_labels=True):
    if case == "1d":
        color = "#24952a" if t == 0 else "#e33333" if t == 1 else "#3489cd"
        axis.plot(x, values, color=color, linewidth=1.6)
        axis.set_ylim(0, vmax * 1.08)
    else:
        # The same levels apply to every method and t in this case.
        axis.contour(x, x, values.reshape(len(x), len(x)),
                     levels=np.linspace(0.08 * vmax, 0.92 * vmax, 8),
                     cmap="rainbow", linewidths=0.9)
        axis.set_ylim(x[0], x[-1])
        axis.set_aspect("equal")
    axis.set_xlim(x[0], x[-1])
    axis.tick_params(labelsize=7)
    if axis_labels:
        axis.set_xlabel("x", fontsize=8)
        if case == "2d":
            axis.set_ylabel("y", fontsize=8)


def _strip(case, x, times, rows, labels, path, vmax):
    figure, axes = plt.subplots(len(rows), len(times),
                               figsize=(2.7 * len(times) + 0.6, 2.8 * len(rows)),
                               squeeze=False)
    for r, (sequence, label) in enumerate(zip(rows, labels)):
        for column, t in enumerate(times):
            _panel(axes[r, column], case, x, sequence[column], t, vmax)
            if r == 0:
                axes[r, column].set_title(rf"$t={time_tag(t)}$", fontsize=16, pad=10)
        axes[r, 0].annotate(label, xy=(-0.38, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontsize=10)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.12,
                           top=0.83 if len(rows) == 1 else 0.90,
                           wspace=0.36, hspace=0.48)
    figure.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def export_sequences(case, x, points, times, sequences, output, *, grid_w2=None):
    """Save one folder per method/case; every density is a normalized grid mass.

    ``sequences`` maps method names to arrays (n_times, n_grid_points).
    ``grid_w2`` is computed once per case and shared by all comparison strips.
    A common y-limit/contour scale makes panels comparable across methods/t.
    """
    times = validate_times(times)
    all_sequences = dict(sequences)
    if grid_w2 is not None:
        all_sequences["grid-W2"] = grid_w2
    shape = (len(times), len(points))
    for name, values in all_sequences.items():
        if (values.shape != shape or not np.isfinite(values).all()
                or np.min(values) < 0 or not np.allclose(values.sum(axis=1), 1)):
            raise ValueError(f"invalid interpolation grid probabilities for {name}")
    vmax = max(float(values.max()) for values in all_sequences.values())
    for name, values in all_sequences.items():
        folder = Path(output) / "methods" / name / case
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "sequence.npz", times=times, x=x, points=points,
                            probabilities=values)
        for index, t in enumerate(times):
            figure, axis = plt.subplots(figsize=(4.2, 3.5))
            _panel(axis, case, x, values[index], t, vmax, axis_labels=False)
            figure.tight_layout()
            figure.savefig(folder / f"t_{time_tag(t)}.png", dpi=170)
            plt.close(figure)
        label = "Grid W2 (regularized)" if name == "grid-W2" else name
        _strip(case, x, times, [values], [label], folder / "interpolation_strip", vmax)
        if grid_w2 is not None and name != "grid-W2":
            _strip(case, x, times, [grid_w2, values],
                   ["Grid W2 (regularized)", name], folder / "w2_vs_method", vmax)
        print(f"Saved {case} {name}: {len(times)} time images + strips in {folder}", flush=True)
