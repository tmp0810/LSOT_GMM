"""Figure-1-style 1D atom-level transport plans, with source/target marginals.

Run from the repository root::

    python -m experiments.intro.plot_transport_plans --L 100 \
        --output-dir results/intro/transport_plans

Plots W2, MW2, and all eight avg/min LSOT variants. The lifted law is
gamma(dx,dy) = sum_ij P_ij N_i(dx) delta_{T_ij(x)}(dy), using the existing
Gaussian pair maps. It is NOT the conditional-mean map Tmean. We integrate
this law over grid cells analytically, rather than drawing samples or
rounding T_ij at cell centers. Each displayed matrix has the same marginals.
Tail probabilities outside [0,1] are folded into the endpoint cells.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from lsot import SparsePlan, average_lsot, minimum_lsot, solve_mw2
from lsot.maps import gaussian_pair_maps
from lsot.projections import PROJECTION_KINDS, sample_projection_bank
from .run import examples


METHODS = ("W2", "MW2") + tuple(
    f"{mode}-LSOT-{kind}" for mode in ("min", "avg") for kind in PROJECTION_KINDS
)


def _array(tensor):
    return tensor.detach().cpu().numpy()


def grid_cells(count):
    """Uniform display cells; the first/last also collect Gaussian tails."""
    if count < 2:
        raise ValueError("grid size must be at least 2")
    edges = np.linspace(0., 1., count+1)
    centers = .5*(edges[:-1]+edges[1:])
    integration_edges = edges.copy()
    integration_edges[0], integration_edges[-1] = -np.inf, np.inf
    return centers, edges, integration_edges


def mixture_histogram(gmm, integration_edges):
    """Exact 1D Gaussian-mixture probabilities in the specified cells."""
    if gmm.dimension != 1:
        raise ValueError("Figure 1 requires one-dimensional Gaussian mixtures")
    edges = np.asarray(integration_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 3 or not np.all(np.diff(edges) > 0):
        raise ValueError("cell edges must be a strictly increasing vector")
    means = _array(gmm.means)[:, 0]
    stds = np.sqrt(_array(gmm.covariances)[:, 0, 0])
    cdf = ndtr((edges[None, :]-means[:, None])/stds[:, None])
    return _array(gmm.weights) @ np.diff(cdf, axis=1)


def rasterize_lifted_plan(source, target, plan, integration_edges):
    """P(cell_r x cell_q) for the full Gaussian-map lifted coupling.

    For T_ij(x)=a*x+b, a>0, integrate N_i over the intersection of
    source cell r and T_ij^{-1}(target cell q), then multiply by P_ij.
    Rows are source cells; columns are target cells. No Tmean, clipping
    renormalization, per-panel scaling, or smoothing is applied.
    """
    if source.dimension != 1 or target.dimension != 1:
        raise ValueError("Figure 1 requires one-dimensional Gaussian mixtures")
    if plan.shape != (source.count, target.count):
        raise ValueError("component plan shape does not match the input mixtures")
    if plan.marginal_error(source, target) > 1e-8:
        raise ValueError("component plan has incorrect marginals")
    # Also validate the integration grid through the histogram helper.
    mixture_histogram(source, integration_edges)
    edges = np.asarray(integration_edges, dtype=float)
    matrix = np.zeros((len(edges)-1, len(edges)-1), dtype=np.float64)
    maps, offsets = gaussian_pair_maps(source, target, plan.rows, plan.cols)
    means = _array(source.means)[:, 0]
    stds = np.sqrt(_array(source.covariances)[:, 0, 0])
    for i, weight, slope, shift in zip(
        _array(plan.rows), _array(plan.mass), _array(maps)[:, 0, 0], _array(offsets)[:, 0]
    ):
        if not np.isfinite(slope) or slope <= 0 or not np.isfinite(shift):
            raise ValueError("Gaussian pair map must have a finite positive slope")
        inverse = (edges-shift)/slope
        left = np.maximum(edges[:-1, None], inverse[None, :-1])
        right = np.minimum(edges[1:, None], inverse[None, 1:])
        probability = ndtr((right-means[i])/stds[i])-ndtr((left-means[i])/stds[i])
        matrix += weight*np.where(right > left, np.maximum(probability, 0.), 0.)
    return matrix


def component_plans(source, target, *, count, seed, plans_dir=None):
    """Reuse intro .npz plans if requested; otherwise run the same core solvers."""
    if plans_dir is not None:
        plans_dir = Path(plans_dir)
        config_file = plans_dir.parent / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            for key, requested in (("L", count), ("seed", seed)):
                if key in config and config[key] != requested:
                    raise ValueError(f"Saved intro {key}={config[key]} differs from requested {requested}")
        plans = {}
        for method in METHODS[1:]:
            tag = method if method == "MW2" else f"{method}_L{count}"
            path = plans_dir / f"1d_{tag}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing saved plan: {path}; use matching --L/--seed or omit --plans-dir")
            with np.load(path) as data:
                shape = tuple(int(v) for v in data["shape"])
                plans[method] = SparsePlan.from_entries(
                    torch.as_tensor(data["rows"], dtype=torch.long),
                    torch.as_tensor(data["cols"], dtype=torch.long),
                    torch.as_tensor(data["mass"], dtype=source.weights.dtype), shape)
        return plans
    plans = {"MW2": solve_mw2(source, target)[0]}
    for kind in PROJECTION_KINDS:
        bank = sample_projection_bank(1, count, kind=kind, seed=seed)
        plans[f"min-LSOT-{kind}"] = minimum_lsot(source, target, bank, kind).plan
        plans[f"avg-LSOT-{kind}"] = average_lsot(source, target, bank, kind)
    return {name: plans[name] for name in METHODS[1:]}


def _draw_panel(figure, slot, matrix, source_histogram, target_histogram, name, vmax):
    layout = slot.subgridspec(2, 2, width_ratios=(1., 3.3),
                             height_ratios=(1., 3.3), wspace=.03, hspace=.03)
    top = figure.add_subplot(layout[0, 1])
    left = figure.add_subplot(layout[1, 0])
    main = figure.add_subplot(layout[1, 1])
    grid = (np.arange(len(source_histogram))+.5)/len(source_histogram)
    top.plot(grid, target_histogram, color="red", lw=1.1)
    top.set(xlim=(0, 1), ylim=(0, 1.08*target_histogram.max()), xticks=[], yticks=[])
    top.set_title(r"$W_2$" if name == "W2" else
                  r"$\mathrm{MW}_2$" if name == "MW2" else name, fontsize=11, pad=5)
    left.plot(-source_histogram, grid, color="blue", lw=1.1)
    left.set(xlim=(-1.08*source_histogram.max(), 0), ylim=(1, 0),
             xticks=[], yticks=[0, .5, 1])
    image = main.imshow(matrix, origin="upper", extent=(0, 1, 1, 0),
                        cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest",
                        aspect="auto")
    main.set(xticks=[0, .5, 1], yticks=[])
    for axis in (top, left, main):
        axis.tick_params(labelsize=8, length=2)
        for spine in axis.spines.values():
            spine.set_color("0.55")
            spine.set_linewidth(.6)
    return image


def save_figure(path, names, matrices, a, b, *, columns, dpi, vmax):
    rows = (len(names)+columns-1)//columns
    figure = plt.figure(figsize=(4.2*columns, 4.0*rows))
    layout = figure.add_gridspec(rows, columns, left=.035, right=.99, bottom=.05,
                                top=.92, wspace=.16, hspace=.3)
    for index, name in enumerate(names):
        _draw_panel(figure, layout[index//columns, index % columns],
                    matrices[name], a, b, name, vmax)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(path.with_suffix("."+suffix), dpi=dpi, facecolor="white")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/intro/transport_plans"))
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=100, help="Cells per axis in the 1D joint plan")
    parser.add_argument("--columns", type=int, default=5, help="Columns in the all-methods figure")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--plans-dir", type=Path,
                        help="Optional existing results/intro/plans directory; original 1D example only")
    args = parser.parse_args(argv)
    if min(args.L, args.columns, args.dpi, args.threads) < 1 or args.grid_size < 2 or args.seed < 0:
        parser.error("L, columns, dpi, threads must be positive; grid-size >= 2; seed >= 0")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads), torch.inference_mode():
        source, target = examples()["1d"]
        grid, display_edges, edges = grid_cells(args.grid_size)
        a, b = mixture_histogram(source, edges), mixture_histogram(target, edges)
        cost = (grid[:, None]-grid[None, :])**2
        # Scaling the ground cost matches the notebook and leaves the LP minimizer unchanged.
        normalized_cost = cost/cost.max()
        matrices = {"W2": ot.emd(a, b, normalized_cost, numThreads=1)}
        plans = component_plans(source, target, count=args.L, seed=args.seed, plans_dir=args.plans_dir)
        for name, plan in plans.items():
            matrices[name] = rasterize_lifted_plan(source, target, plan, edges)
        records = []
        for name in METHODS:
            matrix = matrices[name]
            error_a = float(np.abs(matrix.sum(1)-a).sum())
            error_b = float(np.abs(matrix.sum(0)-b).sum())
            if not np.isfinite(matrix).all() or matrix.min() < -1e-12 or max(error_a, error_b) > 1e-8:
                raise RuntimeError(f"Invalid grid plan for {name}: marginal errors {error_a}, {error_b}")
            record = {"method": name, "L": args.L if name not in {"W2", "MW2"} else 0,
                      "grid_cost_squared": float(np.sum(matrix*cost)),
                      "component_cost_squared": float(plans[name].cost(source, target)) if name in plans else None,
                      "source_marginal_l1": error_a, "target_marginal_l1": error_b,
                      "total_mass": float(matrix.sum())}
            records.append(record)
            print(f"{name}: mass={matrix.sum():.12g}, marginal L1=({error_a:.2g}, {error_b:.2g})", flush=True)
    # One shared *linear* color scale, including all individual and two-panel figures.
    vmax = max(float(matrix.max()) for matrix in matrices.values())
    save_figure(output / "transport_plans_all", METHODS, matrices, a, b,
                columns=min(args.columns, len(METHODS)), dpi=args.dpi, vmax=vmax)
    save_figure(output / "transport_plans_w2_mw2", METHODS[:2], matrices, a, b,
                columns=2, dpi=args.dpi, vmax=vmax)
    for name in METHODS:
        save_figure(output / "methods" / name / "transport_plan", (name,), matrices, a, b,
                    columns=1, dpi=args.dpi, vmax=vmax)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / "transport_plans.npz", grid=grid, display_edges=display_edges,
                        integration_edges=edges, source_histogram=a, target_histogram=b,
                        method_names=np.array(METHODS), matrices=np.stack([matrices[n] for n in METHODS]),
                        component_method_names=np.array(METHODS[1:]),
                        component_matrices=np.stack([_array(plans[n].dense()) for n in METHODS[1:]]))
    configuration = {**vars(args), "method_order": list(METHODS), "color_vmin": 0., "color_vmax": vmax,
                     "dtype": "float64", "device": "cpu", "source_dimension": 1,
                     "law": "sum_ij P_ij N_i(dx) delta_Tij(x)(dy); not Tmean",
                     "discretization": "Exact cell probabilities; tails outside [0,1] folded into endpoint cells",
                     "grid_w2": "Unregularized finite OT with squared Euclidean cell-center cost",
                     "component_plan_origin": "saved intro plans" if args.plans_dir else "core solvers",
                     "inputs": {side: {field: _array(getattr(gmm, field)).tolist()
                                       for field in ("weights", "means", "covariances")}
                                for side, gmm in (("source", source), ("target", target))}}
    (output / "config.json").write_text(json.dumps(configuration, indent=2, default=str)+"\n")
    print(f"Saved all 10 methods to {output.resolve()}", flush=True)
    return records


if __name__ == "__main__":
    main()
