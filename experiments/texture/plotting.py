"""Texture comparison panels and K sweeps."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_comparison(folder, exemplar, adsn_image, outputs, rows):
    panels = [("Exemplar", exemplar), ("ADSN", adsn_image)]
    for method, image in outputs.items():
        panels.append((f"{method}\npatch SW2: {rows[method]['patch_sw2']:.4f}", image))
    cols = min(4, len(panels))
    fig, axes = plt.subplots(int(np.ceil(len(panels)/cols)), cols,
                             figsize=(3.15*cols, 3.35*np.ceil(len(panels)/cols)),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title, fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(folder/f"comparison.{extension}", dpi=160)
    plt.close(fig)


def save_curves(folder, summary, adsn_rows):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    specs = (
        ("patch_sw2", "Patch SW2", 1), ("patch_w2", "Patch W2", 1),
        ("autocov_rmse", "RGB autocovariance RMSE", 1), ("plan_ms", "Plan time (ms)", 1),
        ("method_ms", "Plan + map + aggregation (ms)", 1), ("total_ms", "Total pipeline time (s)", 1000),
    )
    for ax, (key, label, factor) in zip(axes.flat, specs):
        for method, frame in summary.groupby("method", sort=False):
            frame = frame.sort_values("K")
            ax.plot(frame.K, frame[f"{key}_mean"]/factor, marker="o", label=method)
        if key in ("patch_sw2", "patch_w2", "autocov_rmse"):
            mean = np.mean([row[key] for row in adsn_rows])
            ax.axhline(mean, color="gray", linestyle="--", linewidth=1, label="ADSN")
        else:
            ax.set_yscale("log")
        ax.set(xlabel="Gaussian components per mixture (K)", ylabel=label,
               xticks=sorted(summary.K.unique()))
        ax.grid(alpha=.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=min(5, len(labels)), fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(folder/f"sweep.{extension}", dpi=170)
    plt.close(fig)
