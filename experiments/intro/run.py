"""Reproduce the GMM-OT introduction examples with MW2 and lifted LSOT plans.

Run ``python -m experiments.intro.run --output-dir results/intro`` from the
repository root. The discrete-grid W2 barycenter follows the notebook's
regularized POT comparison; it is not the exact continuous-GMM W2 geodesic.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal, norm
from threadpoolctl import threadpool_limits

from lsot import (GMM, BarycentricMap, average_lsot, minimum_lsot,
                  optimized_minimum_lsot, solve_mw2)
from lsot.gaussians import covariance_roots, symmetric_sqrt
from lsot.projections import PROJECTION_KINDS, sample_projection_bank


TIMES_1D = (0.2, 0.5, 0.8)
FAMILIES = tuple(PROJECTION_KINDS)
MODES = ("avg", "min")


def examples():
    """Exact weights, means, and covariances of GMM_OT_introduction.ipynb."""
    source_1d = GMM.from_numpy(
        [0.3, 0.7], [[0.2], [0.4]], [[[0.0009]], [[0.0016]]])
    target_1d = GMM.from_numpy(
        [0.6, 0.4], [[0.6], [0.8]], [[[0.0036]], [[0.0049]]])
    eye = np.eye(2)[None] * 0.01
    source_2d = GMM.from_numpy(
        [0.5, 0.5], [[0.3, 0.3], [0.7, 0.4]], np.repeat(eye, 2, axis=0))
    target_2d = GMM.from_numpy(
        [0.45, 0.55], [[0.5, 0.6], [0.4, 0.25]], np.repeat(eye, 2, axis=0))
    return {"1d": (source_1d, target_1d), "2d": (source_2d, target_2d)}


def _density(means, covariances, weights, points):
    values = np.zeros(len(points), dtype=np.float64)
    d = points.shape[1]
    for weight, mean, covariance in zip(weights, means, covariances):
        if weight <= 0:
            continue
        if d == 1:
            values += weight * norm.pdf(points[:, 0], mean[0], np.sqrt(covariance[0, 0]))
        else:
            values += weight * multivariate_normal.pdf(points, mean=mean, cov=covariance)
    return values


def _mixture_density(gmm, points):
    return _density(gmm.means.numpy(), gmm.covariances.numpy(),
                    gmm.weights.numpy(), points)


def _interpolated_density(source, target, plan, points, t):
    """Gaussian displacement interpolation for every positive plan edge."""
    rows, cols = plan.rows, plan.cols
    roots, inverse_roots = covariance_roots(source.covariances)
    middle = roots[rows] @ target.covariances[cols] @ roots[rows]
    maps = inverse_roots[rows] @ symmetric_sqrt(middle) @ inverse_roots[rows]
    maps = 0.5 * (maps + maps.transpose(-1, -2))
    transition = (1 - t) * torch.eye(source.dimension, dtype=maps.dtype)[None] + t * maps
    covariances = transition @ source.covariances[rows] @ transition.transpose(-1, -2)
    means = (1 - t) * source.means[rows] + t * target.means[cols]
    return _density(means.numpy(), covariances.numpy(), plan.mass.numpy(), points)


def _save_plan(path, plan):
    np.savez_compressed(path, rows=plan.rows.numpy(), cols=plan.cols.numpy(),
                        mass=plan.mass.numpy(), shape=np.array(plan.shape))


def _components_figure(source, target, case, x, points, output):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if case == "1d":
        for gmm, label, color in ((source, "source", "#27659b"),
                                  (target, "target", "#bf4f3c")):
            density = _mixture_density(gmm, points)
            ax.plot(x, density / density.sum(), label=label, color=color, lw=2)
        ax.set_ylabel("Discrete density (sum = 1)")
        ax.set_xlabel("x")
        ax.legend()
    else:
        for gmm, color in ((source, "#27659b"), (target, "#bf4f3c")):
            ax.contour(x, x, _mixture_density(gmm, points).reshape(len(x), len(x)),
                       levels=8, colors=color, linewidths=1.3)
        ax.set(xlabel="x", ylabel="y", aspect="equal")
        ax.set_title("Source (blue), target (red)")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _comparison_figures(case, source, target, plans, x, points, out):
    names = list(plans)
    fig, axes = plt.subplots(4, 4, figsize=(12, 11), constrained_layout=True)
    vmax = max(float(plan.dense().max()) for plan in plans.values())
    for ax, name in zip(axes.flat, names):
        ax.imshow(plans[name].dense().numpy(), vmin=0, vmax=vmax,
                  cmap="Blues", interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="target", ylabel="source")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{float(plans[name].dense()[i, j]):.2f}",
                        ha="center", va="center", fontsize=9)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle(f"{case.upper()}: component couplings (shared color scale)")
    fig.savefig(out / f"{case}_couplings.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(15, 11), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        plan = plans[name]
        if case == "1d":
            first = _mixture_density(source, points)
            last = _mixture_density(target, points)
            ax.plot(x, first / first.sum(), ":", color="#27659b", lw=1)
            ax.plot(x, last / last.sum(), ":", color="#bf4f3c", lw=1)
            for t, color in zip(TIMES_1D, ("#43a084", "#222d42", "#c17437")):
                density = _interpolated_density(source, target, plan, points, t)
                ax.plot(x, density / density.sum(), color=color, lw=1.7, label=f"t={t:g}")
        else:
            density = _interpolated_density(source, target, plan, points, 0.5)
            ax.contour(x, x, density.reshape(len(x), len(x)), levels=8,
                       cmap="viridis", linewidths=1.3)
            ax.set_aspect("equal")
        ax.set_title(name, fontsize=9)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    if case == "1d":
        axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{case.upper()}: Gaussian geodesics" +
                 (" (t=0.2, 0.5, 0.8)" if case == "1d" else " (t=0.5)"))
    fig.savefig(out / f"{case}_interpolations.png", dpi=170)
    plt.close(fig)


def _map_1d(source, target, plan, x, output):
    """Show component maps (the multivalued map) and their conditional mean."""
    mapper = BarycentricMap.from_plan(source, target, plan)
    tmean = mapper.transform(torch.as_tensor(x[:, None], dtype=torch.float64)).numpy()[:, 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, j, mass in zip(plan.rows.tolist(), plan.cols.tolist(), plan.mass.tolist()):
        matrix = np.sqrt(float(target.covariances[j, 0, 0] / source.covariances[i, 0, 0]))
        branch = float(target.means[j, 0]) + matrix * (x - float(source.means[i, 0]))
        ax.plot(x, branch, alpha=min(1, 0.25 + mass), lw=1.6,
                label=f"G{i} to G{j} (mass {mass:.2f})")
    ax.plot(x, tmean, "k", lw=2.2, label="conditional mean")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="source x", ylabel="target y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _grid_w2_figures(case, source, target, x, points, out):
    """Regularized W2 barycenter comparison from the original notebook."""
    import ot

    a = _mixture_density(source, points)
    b = _mixture_density(target, points)
    # A POT barycenter takes probability histograms. The notebook normalizes
    # 1D only; we also normalize 2D, recording this correction in the README.
    a, b = a / a.sum(), b / b.sum()
    cost = ot.dist(points, points, metric="sqeuclidean")
    if case == "1d":
        cost /= cost.max()
        coupling = ot.emd(a, b, cost)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(coupling, origin="lower", cmap="Blues", aspect="auto")
        axes[0].set(title="Discrete W2 coupling", xlabel="target grid index",
                    ylabel="source grid index")
        for t in TIMES_1D:
            bary = ot.bregman.barycenter(np.column_stack((a, b)), cost, 1e-3,
                                         weights=np.array([1-t, t]))
            axes[1].plot(x, bary / bary.sum(), label=f"t={t:g}")
        axes[1].plot(x, a, ":", color="#27659b")
        axes[1].plot(x, b, ":", color="#bf4f3c")
        axes[1].set(title="Regularized grid W2 barycenter", xlabel="x")
        axes[1].legend()
    else:
        bary = ot.bregman.barycenter(np.column_stack((a, b)), cost, 1e-3,
                                     weights=np.array([0.5, 0.5]))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
        for ax, z, title in zip(axes, (a, bary, b), ("Source", "W2, t=0.5", "Target")):
            ax.contour(x, x, z.reshape(len(x), len(x)), levels=8, cmap="viridis")
            ax.set(title=title, aspect="equal")
    if case == "1d":
        fig.tight_layout()
    fig.savefig(out / f"{case}_grid_w2.png", dpi=170)
    plt.close(fig)


def _run_case(case, source, target, args, output):
    x = np.linspace(0, 1, args.grid_1d if case == "1d" else args.grid_2d)
    if case == "1d":
        points = x[:, None]
    else:
        X, Y = np.meshgrid(x, x)
        points = np.column_stack((X.ravel(), Y.ravel()))
    _components_figure(source, target, case, x, points, output / f"{case}_components.png")
    if not args.skip_grid_w2:
        _grid_w2_figures(case, source, target, x, points, output)

    t0 = time.perf_counter()
    reference, mw2_cost = solve_mw2(source, target)
    mw2_ms = 1000 * (time.perf_counter() - t0)
    plans = {"MW2": reference}
    records = []

    def record(name, kind, mode, plan, elapsed, initial_cost=None, best_iteration=None):
        cost = float(plan.cost(source, target))
        if name == "MW2":
            if not np.isclose(cost, mw2_cost, rtol=1e-8, atol=1e-12):
                raise RuntimeError("MW2 solver cost disagrees with plan cost")
        if plan.marginal_error(source, target) > 1e-8:
            raise RuntimeError(f"{name} has invalid component marginals")
        records.append({"case": case, "d": source.dimension, "method": name,
                        "aggregation": mode, "projection": kind, "L": 0 if name == "MW2" else args.L,
                        "cost_squared": cost, "cost_gap_pct": 100 * (cost / mw2_cost - 1),
                        "plan_l1_to_mw2": float((plan.dense() - reference.dense()).abs().sum()),
                        "transport_ms": elapsed, "plan_nnz": plan.nnz,
                        "initial_cost_squared": initial_cost, "best_iteration": best_iteration})
        plans[name] = plan
        _save_plan(output / "plans" / f"{case}_{name}.npz", plan)
        if case == "1d":
            _map_1d(source, target, plan, x, output / "maps_1d" / f"{name}.png")
        print(f"{case} {name:26s} cost²={cost:.8f} gap={100*(cost/mw2_cost-1):6.2f}%", flush=True)

    record("MW2", "", "MW2", reference, mw2_ms)
    for kind in FAMILIES:
        bank = sample_projection_bank(source.dimension, args.L, kind=kind, seed=args.seed)
        for mode in MODES:
            t0 = time.perf_counter()
            if mode == "avg":
                plan = average_lsot(source, target, bank, kind)
                initial_cost, best_iteration = None, None
            elif mode == "min":
                result = minimum_lsot(source, target, bank, kind)
                plan, initial_cost, best_iteration = result.plan, None, None
            else:
                result = optimized_minimum_lsot(
                    source, target, bank, kind, steps=args.opt_steps,
                    samples=args.opt_samples, epsilon=args.opt_epsilon,
                    learning_rate=args.opt_learning_rate, seed=args.seed)
                plan = result.plan
                initial_cost = float(result.initial_cost_squared)
                best_iteration = result.best_iteration
            elapsed = 1000 * (time.perf_counter() - t0)
            record(f"{mode}-LSOT-{kind}_L{args.L}", kind, mode, plan, elapsed,
                   initial_cost, best_iteration)
    _comparison_figures(case, source, target, plans, x, points, output)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/intro"))
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opt-steps", type=int, default=20)
    parser.add_argument("--opt-samples", type=int, default=8)
    parser.add_argument("--opt-epsilon", type=float, default=0.05)
    parser.add_argument("--opt-learning-rate", type=float, default=0.05)
    parser.add_argument("--grid-1d", type=int, default=100)
    parser.add_argument("--grid-2d", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-grid-w2", action="store_true",
                        help="Skip the notebook's regularized grid W2 barycenter figures")
    args = parser.parse_args(argv)
    if (args.L < 1 or args.seed < 0 or args.opt_steps < 0 or args.opt_samples < 1
            or args.opt_epsilon <= 0 or args.opt_learning_rate <= 0
            or args.grid_1d < 2 or args.grid_2d < 2 or args.threads < 1):
        parser.error("counts and optimizer scales must be valid and positive")
    output = args.output_dir
    (output / "plans").mkdir(parents=True, exist_ok=True)
    (output / "maps_1d").mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        cases = examples()
        rows = [row for case, (source, target) in cases.items()
                for row in _run_case(case, source, target, args, output)]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    print(f"Saved {len(rows)} method/case results to {output}")
    return rows


if __name__ == "__main__":
    main()
