"""Notebook-style contour grids and uncaptained individual panels."""
from __future__ import annotations

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal


def density_grid(gmm, bounds, count):
    x = np.linspace(*bounds, count)
    xx, yy = np.meshgrid(x, x)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    density = np.zeros(len(points))
    for weight, mean, covariance in zip(gmm.weights.detach().cpu().numpy(),
                                        gmm.means.detach().cpu().numpy(),
                                        gmm.covariances.detach().cpu().numpy()):
        density += weight*multivariate_normal.pdf(points, mean=mean, cov=covariance)
    return x, density.reshape(count, count)


def draw_density(ax, x, values, *, show_axes=False, cmap="RdGy"):
    ax.contour(x, x, values, 8, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(x[0], x[-1])
    if not show_axes:
        ax.axis("off")


def save_panel(path, x, density, *, show_axes=False):
    fig, ax = plt.subplots(figsize=(3, 3))
    draw_density(ax, x, density, show_axes=show_axes)
    if show_axes:
        fig.tight_layout(pad=.1)
    else:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_inputs(folder, case):
    fig, axes = plt.subplots(1, len(case.inputs), figsize=(4*len(case.inputs), 4), squeeze=False)
    for index, (ax, gmm) in enumerate(zip(axes.flat, case.inputs)):
        x, density = density_grid(gmm, case.bounds, case.density_grid)
        draw_density(ax, x, density, cmap="viridis")
        save_panel(folder / f"input_{index}.png", x, density)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=.02)
    fig.savefig(folder / "inputs.png", dpi=160)
    plt.close(fig)


def save_gaussian_overlay(folder, case, barycenter):
    fig, ax = plt.subplots(figsize=(5, 5))
    for source in case.inputs:
        x, density = density_grid(source, case.bounds, case.density_grid)
        draw_density(ax, x, density, show_axes=True, cmap="viridis")
    x, density = density_grid(barycenter, case.bounds, case.density_grid)
    draw_density(ax, x, density, show_axes=True, cmap="plasma")
    fig.tight_layout()
    fig.savefig(folder / "gaussian_overlay.png", dpi=170)
    fig.savefig(folder / "gaussian_overlay.pdf")
    plt.close(fig)


def save_method_grid(folder, records, size):
    fig, axes = plt.subplots(size, size, figsize=(2.6*size, 2.6*size), squeeze=False)
    for record in records:
        path = folder / "barycenters" / f"r{record['row']:02d}_c{record['col']:02d}.npz"
        with np.load(path) as data:
            draw_density(axes[record["row"], record["col"]], data["x"], data["density"])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.02, wspace=.02)
    fig.savefig(folder / "barycenters.png", dpi=150)
    fig.savefig(folder / "barycenters.pdf")
    plt.close(fig)


def save_comparison(folder, methods, center):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        file = folder / method / "barycenters" / f"r{center:02d}_c{center:02d}.npz"
        if not file.exists():
            ax.axis("off")
            continue
        with np.load(file) as data:
            draw_density(ax, data["x"], data["density"])
        ax.set_title(method, fontsize=10)
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.tight_layout(pad=.5)
    fig.savefig(folder / "center_comparison.png", dpi=170)
    plt.close(fig)


def save_convergence(folder, methods, center):
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for method in methods:
        path = folder / method / "history" / f"r{center:02d}_c{center:02d}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        initial = float(rows[0]["best_objective_squared"])
        if initial <= 0:
            continue
        ax.plot([int(r["iteration"]) for r in rows],
                [float(r["best_objective_squared"])/initial for r in rows], label=method)
        plotted = True
    if plotted:
        ax.set(xlabel="L-BFGS outer iteration", ylabel="Objective / initial objective")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(folder / "center_convergence.png", dpi=170)
    plt.close(fig)


def save_perimeter_animation(folder, size, *, mp4=False):
    if size < 2:
        return
    perimeter = ([(0, col) for col in range(size)]
                 + [(row, size-1) for row in range(1, size)]
                 + [(size-1, col) for col in range(size-2, -1, -1)]
                 + [(row, 0) for row in range(size-2, 0, -1)])
    frames = []
    for row, col in perimeter:
        with Image.open(folder / "images" / f"r{row:02d}_c{col:02d}.png") as image:
            frames.append(image.convert("RGB"))
    frames[0].save(folder / "perimeter.gif", save_all=True, append_images=frames[1:],
                   duration=200, loop=0)
    if mp4:
        from matplotlib.animation import FFMpegWriter
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("--mp4 needs ffmpeg; GIFs are already available")
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("off")
        display = ax.imshow(np.asarray(frames[0]))
        writer = FFMpegWriter(fps=5)
        with writer.saving(fig, str(folder / "perimeter.mp4"), dpi=120):
            for frame in frames:
                display.set_data(np.asarray(frame))
                writer.grab_frame()
        plt.close(fig)
