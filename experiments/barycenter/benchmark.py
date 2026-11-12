"""Reproducible plan scaling, solver ablations, and barycenter scaling.

Examples are in README.md. Timings include sampling, initialization screening,
optimizer setup and validation selection, but not post-hoc metrics/plots.
MW2 is the multi-marginal LP; MW2-variational is an additional local baseline.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lsot.gaussians import GMM
from lsot.projections import PROJECTION_KINDS, sample_projection_bank
from .data import synthetic_example, gaussian_example, image_example, save_gmm
from .reference import screened_initialization, shared_initialization, mw2_barycenter
from .solver import on_device
from .fast_geometry import pair_costs, project_components
from .fast_solver import (FastObjective, MW2Objective, optimize_fast, checkpoint_gmm,
                          plan_entries, edge_costs, synchronize)
from .run import write_csv, write_json, empirical_sw2, solver_source_hash


VARIANTS = ("reference-fixed", "fast-historical", "fast-fixed", "fast-fresh", "fast-partial")


def random_gmm(count, seed, dimension=2):
    rng = np.random.default_rng(seed)
    roots = rng.normal(size=(count, dimension, dimension))
    return GMM.from_numpy(rng.uniform(.2, 1., count), rng.uniform(-1, 1, (count, dimension)),
                          .02*(roots @ roots.transpose(0, 2, 1))+.005*np.eye(dimension))


def elapsed_call(function, device):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter()-start)*1000


def plan_once(source, target, method, L, seed, cached_cost=None):
    """Return cost and marginal error; the latter is checked outside timers."""
    import ot
    with torch.no_grad():
        if method == "MW2":
            costs = pair_costs(source, target) if cached_cost is None else cached_cost
            plan = ot.emd(source.weights, target.weights, costs)
            return (float((plan*costs).sum()), plan.sum(1), plan.sum(0))
        mode, _, kind = method.split("-")
        bank = sample_projection_bank(source.dimension, L, kind=kind, seed=seed,
                                       device=source.weights.device, dtype=source.weights.dtype)
        x, y = project_components(source, bank, kind), project_components(target, bank, kind)
        direction, rows, cols, masses = plan_entries(x, y, source.weights, target.weights)
        costs = edge_costs(source, target, rows, cols) if cached_cost is None else cached_cost[rows, cols]
        scores = masses.new_zeros(L).scatter_add_(0, direction, masses*costs)
        if mode == "min":
            keep = direction == scores.argmin()
            rows, cols, masses = rows[keep], cols[keep], masses[keep]
            value = scores.min()
        else:
            value, masses = scores.mean(), masses/L
        left = masses.new_zeros(source.count).scatter_add_(0, rows, masses)
        right = masses.new_zeros(target.count).scatter_add_(0, cols, masses)
        return float(value), left, right


def benchmark_plans(args):
    rows = []
    for count in args.component_counts:
        for seed in args.seeds:
            source, target = [on_device(random_gmm(count, seed*2+i), args.device) for i in (0, 1)]
            cached = pair_costs(source, target)
            methods = ["MW2"]+[f"{mode}-LSOT-{kind}" for mode in args.aggregations for kind in args.kinds]
            for scope in ("total", "cached_ground_cost"):
                reference = None
                for method in methods:
                    function = lambda: plan_once(source, target, method, args.L, seed,
                                                  cached if scope == "cached_ground_cost" else None)
                    for _ in range(args.warmups):
                        function()
                    samples = [elapsed_call(function, args.device) for _ in range(args.repeats)]
                    cost, left, right = samples[-1][0]
                    error = float(torch.maximum((left-source.weights).abs().max(),
                                                (right-target.weights).abs().max()))
                    if error > 1e-8:
                        raise RuntimeError(f"Invalid coupling marginals: {method}, {error}")
                    if method == "MW2":
                        reference = cost
                    durations = np.array([s[1] for s in samples])
                    row = {"K": count, "seed": seed, "method": method, "scope": scope,
                           "L": 0 if method == "MW2" else args.L, "median_ms": float(np.median(durations)),
                           "q25_ms": float(np.quantile(durations, .25)), "q75_ms": float(np.quantile(durations, .75)),
                           "cost_squared": cost, "cost_ratio_to_mw2": cost/max(reference, 1e-15),
                           "marginal_error": error, "device": args.device, "repeats": args.repeats}
                    rows.append(row)
                    print(f"plans K={count} {method} {scope}: {row['median_ms']:.3f} ms", flush=True)
                    write_csv(args.output_dir / "plans.csv", rows)
    for scope in ("total", "cached_ground_cost"):
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = defaultdict(list)
        for row in rows:
            if row["scope"] == scope:
                groups[row["method"]].append(row)
        for method, values in groups.items():
            ks = sorted({r["K"] for r in values})
            ax.plot(ks, [np.median([r["median_ms"] for r in values if r["K"] == k]) for k in ks], "o-", label=method)
        ax.set(xscale="log", yscale="log", xlabel="Components in each GMM", ylabel="Median wall time (ms)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"plans_{scope}.png", dpi=180)
        plt.close(fig)
    return rows


def warmup(inputs, weights, initial, args):
    # Fresh throwaway states; never warm-start a measured solve.
    for kind in args.kinds:
        optimize_fast(inputs, weights, initial, kind=kind, steps=1, L=2,
                      validation_L=2, patience=0)
    optimize_fast(inputs, weights, initial, mode="mw2", steps=1, L=2,
                  validation_L=2, patience=0)


def solver_case(name, cpu_inputs, bounds, args, rows, trajectory):
    weights = np.ones(len(cpu_inputs))/len(cpu_inputs)
    inputs = [on_device(g, args.device) for g in cpu_inputs]
    budget = args.barycenter_components or sum(g.count for g in inputs)-len(inputs)+1
    scale = max(1., bounds[1]-bounds[0])
    case_dir = args.output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    initial, _ = screened_initialization(cpu_inputs, weights, budget, seed=0)
    warmup(inputs, weights, on_device(initial, args.device), args)
    common = MW2Objective(inputs, weights)
    reference, reference_cost = None, None
    tuple_count = int(np.prod([g.count for g in inputs]))
    if tuple_count <= args.max_tuples:
        function = lambda: mw2_barycenter(cpu_inputs, weights, iterations=args.gaussian_iterations,
                                           fast=True, tolerance=args.gaussian_tolerance)
        for _ in range(args.warmups):
            function()
        measurements = [elapsed_call(function, "cpu") for _ in range(args.repeats)]
        reference = on_device(measurements[-1][0].gmm, args.device)
        with torch.no_grad():
            reference_cost = float(common(reference))
        rows.append({"case": name, "method": "MW2", "seed": -1, "K": inputs[0].count,
                     "Kb": reference.count, "total_ms": float(np.median([t for _, t in measurements])),
                     "mw2_objective": reference_cost, "relative_mw2_gap": 0., "status": "multimarginal_lp",
                     "tuple_count": tuple_count, "iterations": measurements[-1][0].covariance_iterations,
                     "covariance_relative_change": measurements[-1][0].covariance_relative_change,
                     "covariance_tolerance_met": measurements[-1][0].covariance_tolerance_met,
                     "gaussian_iteration_budget": args.gaussian_iterations,
                     "runtime_device": "cpu", "repeats": args.repeats})
        save_gmm(case_dir / "MW2.npz", reference)
    else:
        rows.append({"case": name, "method": "MW2", "seed": -1, "K": inputs[0].count,
                     "Kb": budget, "status": "skipped_tuple_budget", "tuple_count": tuple_count,
                     "total_ms": None, "mw2_objective": None, "relative_mw2_gap": None})
    for seed in args.seeds:
        for variant in list(args.variants)+["MW2-variational"]:
            historical = variant in {"reference-fixed", "fast-historical"}
            def initialize():
                if historical:
                    candidate = shared_initialization(cpu_inputs, weights, budget, seed=seed,
                                                       iterations=10, fast=True)
                    info = {"strategy": "historical", "optimizer_starts": 1}
                else:
                    candidate, info = screened_initialization(cpu_inputs, weights, budget, seed=seed)
                return on_device(candidate, args.device), info
            (initial, info), initialization_ms = elapsed_call(initialize, args.device)
            pairs = [("mw2", "")] if variant == "MW2-variational" else [
                (mode, kind) for mode in args.aggregations for kind in args.kinds]
            for mode, kind in pairs:
                method = variant if mode == "mw2" else f"{variant}/{mode}-LSOT-{kind}"
                sampling = "fresh" if variant == "fast-fresh" else "partial" if variant == "fast-partial" else "fixed"
                print(f"{name}: {method}, seed={seed}, Kb={budget}", flush=True)
                function = lambda: optimize_fast(inputs, weights, initial, kind=kind, mode=mode,
                    steps=args.steps, learning_rate=args.learning_rate, L=args.L,
                    validation_L=args.validation_L, seed=seed, sampling=sampling,
                    validate_every=args.validate_every, patience=args.patience,
                    refresh_every=args.refresh_every, refresh_fraction=args.refresh_fraction,
                    relative_tolerance=args.relative_tolerance, coordinate_scale=scale,
                    checkpoints=tuple(args.checkpoints),
                    backend="reference" if variant == "reference-fixed" else "fast")
                solved, solve_ms = elapsed_call(function, args.device)
                folder = case_dir / method.replace("/", "_") / f"seed{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                save_gmm(folder / "barycenter.npz", solved.gmm)
                write_json(folder / "initialization.json", info)
                write_csv(folder / "history.csv", solved.history)
                with torch.no_grad():
                    value = float(common(solved.gmm))
                    test_value = value
                    if mode != "mw2":
                        test_bank = sample_projection_bank(initial.dimension, args.test_L, kind=kind,
                            seed=seed+2000003, device=args.device)
                        test_value = float(FastObjective(inputs, weights, test_bank, kind, mode)(solved.gmm))
                    sw = empirical_sw2(reference, solved.gmm, count=args.eval_samples,
                                       directions=100, seed=9000001+seed) if reference is not None else None
                row = {"case": name, "method": method, "seed": seed, "K": inputs[0].count,
                       "Kb": budget, "L": 0 if mode == "mw2" else args.L,
                       "total_ms": initialization_ms+solve_ms, "initialization_ms": initialization_ms,
                       "solve_ms": solve_ms, "validation_ms": solved.validation_ms,
                       "mw2_objective": value, "relative_mw2_gap":
                       (value-reference_cost)/max(reference_cost, 1e-15) if reference_cost is not None else None,
                       "sw2_to_reference": sw, "test_objective": test_value,
                       "validation_objective": solved.objective, "iterations": solved.iterations,
                       "selected_step": solved.selected_step, "status": solved.status,
                       "sampling": sampling, "runtime_device": args.device, "tuple_count": tuple_count}
                rows.append(row)
                seen = set()
                for checkpoint in reversed(solved.checkpoints):
                    if checkpoint["iteration"] in seen:
                        continue
                    seen.add(checkpoint["iteration"])
                    point = checkpoint_gmm(initial, checkpoint, scale, device=args.device)
                    with torch.no_grad():
                        cost = float(common(point))
                    trajectory.append({"case": name, "method": method, "seed": seed,
                        "iteration": checkpoint["iteration"], "selected_step": checkpoint["selected_step"],
                        "elapsed_ms": initialization_ms+checkpoint["elapsed_ms"],
                        "mw2_objective": cost, "relative_mw2_gap":
                        (cost-reference_cost)/max(reference_cost, 1e-15) if reference_cost is not None else None,
                        "validation_objective": checkpoint["validation_objective"]})
                    save_gmm(folder / f"step{checkpoint['iteration']:05d}.npz", point)
                write_csv(args.output_dir / "solvers.csv", rows)
                write_csv(args.output_dir / "trajectory.csv", trajectory)


def plot_solvers(folder, rows, trajectory):
    for case in sorted({r["case"] for r in rows}):
        available = sorted({r["method"] for r in trajectory if r["case"] == case})
        selections = {"": available}
        for kind in PROJECTION_KINDS:
            matched = [m for m in available if m.endswith("-"+kind)]
            if matched:
                selections["_"+kind] = matched+[m for m in available if m == "MW2-variational"]
        for suffix, methods in selections.items():
            fig, ax = plt.subplots(figsize=(11, 5))
            for method in methods:
                values = [r for r in trajectory if r["case"] == case and r["method"] == method]
                color = None
                for index, seed in enumerate(sorted({r["seed"] for r in values})):
                    points = sorted([r for r in values if r["seed"] == seed], key=lambda r: r["elapsed_ms"])
                    line, = ax.plot([r["elapsed_ms"]/1000 for r in points],
                                   [r["mw2_objective"] for r in points], "o-", color=color,
                                   alpha=.7, label=method if index == 0 else None)
                    color = line.get_color()
            refs = [r for r in rows if r["case"] == case and r["method"] == "MW2" and r["total_ms"] is not None]
            if refs:
                ax.scatter([refs[0]["total_ms"]/1000], [refs[0]["mw2_objective"]], marker="*", s=130,
                           color="black", label="MW2 multi-marginal")
            ax.set(xlabel="Wall time including initialization and selection (s)",
                   ylabel="Common MW2 objective", title=case, xscale="log")
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.))
            fig.tight_layout()
            fig.savefig(folder / f"{case}_time_quality{suffix}.png", dpi=180)
            plt.close(fig)
    summary = []
    for case, method in sorted({(r["case"], r["method"]) for r in rows}):
        selected = [r for r in rows if r["case"] == case and r["method"] == method]
        item = {"case": case, "method": method, "runs": len(selected),
                "numerical_stops": sum(r["status"].startswith("numerical_stop") for r in selected)}
        for metric in ("total_ms", "mw2_objective", "relative_mw2_gap", "sw2_to_reference", "iterations"):
            vals = [r[metric] for r in selected if r.get(metric) is not None]
            if vals:
                item[metric+"_mean"] = float(np.mean(vals))
                item[metric+"_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.
        summary.append(item)
    write_csv(folder / "summary.csv", summary)
    targets = []
    for case, method, seed in sorted({(r["case"], r["method"], r["seed"]) for r in trajectory}):
        points = [r for r in trajectory if (r["case"], r["method"], r["seed"]) == (case, method, seed)]
        for target in (.01, .05, .10):
            reached = [r for r in points if r["relative_mw2_gap"] is not None and r["relative_mw2_gap"] <= target]
            targets.append({"case": case, "method": method, "seed": seed, "relative_gap_target": target,
                            "reached": bool(reached),
                            "first_observed_ms": min(r["elapsed_ms"] for r in reached) if reached else None})
    for row in rows:
        if row["method"] == "MW2":
            for target in (.01, .05, .10):
                targets.append({"case": row["case"], "method": "MW2", "seed": -1,
                                "relative_gap_target": target, "reached": row["total_ms"] is not None,
                                "first_observed_ms": row["total_ms"]})
    write_csv(folder / "time_to_quality.csv", targets)
    scaling = [r for r in rows if r["case"].startswith("random_K") and r["total_ms"] is not None]
    if scaling:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted({r["method"] for r in scaling}):
            data = [r for r in scaling if r["method"] == method]
            ks = sorted({r["K"] for r in data})
            ax.plot(ks, [np.median([r["total_ms"] for r in data if r["K"] == k]) for k in ks], "o-", label=method)
        ax.set(xscale="log", yscale="log", xlabel="Input components K (four input GMMs)",
               ylabel="Wall time including initialization and selection (ms)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / "barycenter_runtime_scaling.png", dpi=180)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plans", "ablation", "scaling"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/barycenter_benchmark"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cases", nargs="+", choices=("gaussian", "synthetic", "images"), default=["synthetic"])
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--kinds", nargs="+", choices=PROJECTION_KINDS, default=["B1D"])
    parser.add_argument("--aggregations", nargs="+", choices=("avg", "min"), default=["min"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--component-counts", nargs="+", type=int)
    parser.add_argument("--barycenter-components", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--validation-L", type=int, default=512)
    parser.add_argument("--test-L", type=int, default=2048)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--refresh-every", type=int, default=10)
    parser.add_argument("--refresh-fraction", type=float, default=.25)
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-tuples", type=int, default=200000)
    parser.add_argument("--gaussian-iterations", type=int, default=100)
    parser.add_argument("--gaussian-tolerance", type=float, default=1e-10)
    parser.add_argument("--eval-samples", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.component_counts is None:
        args.component_counts = [10, 20, 50, 100, 200, 500] if args.task == "plans" else [3, 5, 8, 12, 20]
    if (min(args.L, args.validation_L, args.test_L, args.repeats, args.threads, args.max_tuples,
            args.validate_every, args.refresh_every, args.gaussian_iterations, args.eval_samples,
            *args.component_counts) < 1 or args.steps < 0 or args.warmups < 0 or args.patience < 0
            or min(args.seeds) < 0 or args.learning_rate <= 0 or not 0 < args.refresh_fraction <= 1
            or args.barycenter_components < 0 or args.relative_tolerance < 0 or args.gaussian_tolerance < 0
            or any(s < 0 for s in args.checkpoints)):
        parser.error("Invalid benchmark settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**vars(args), "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                "solver_sources_sha256": solver_source_hash(),
                "timing": "resident GMM inputs; includes initialization/sampling/validation; excludes fit, plots and post-hoc metrics",
                "selection_bank_seed_offset": 1000003, "test_bank_seed_offset": 2000003}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        parser.error("Use a new output directory for an independent timing run")
    write_json(manifest_path, manifest)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        if args.task == "plans":
            return benchmark_plans(args)
        rows, trajectory = [], []
        if args.task == "scaling":
            for count in args.component_counts:
                solver_case(f"random_K{count}", [random_gmm(count, 100+i) for i in range(4)],
                            (-1., 1.), args, rows, trajectory)
        else:
            for case in args.cases:
                data = (gaussian_example() if case == "gaussian" else synthetic_example() if case == "synthetic"
                        else image_example(args.output_dir / "image_data", image_dir=args.image_dir, components=12, seed=0))
                solver_case(case, data.inputs, data.bounds, args, rows, trajectory)
        plot_solvers(args.output_dir, rows, trajectory)
    return rows


if __name__ == "__main__":
    main()
