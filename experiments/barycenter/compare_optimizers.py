"""Paired Adam-only versus Adam + L-BFGS checks at equal-weight centers.

python -m experiments.barycenter.compare_optimizers --cases synthetic \
    --seeds 0 1 2 --output-dir results/barycenter_optimizer_comparison

Input GMMs and the common initialization stay fixed across projection seeds.
Each optimizer sees the same input, start and bank within a paired comparison.
Outer iterations are not equal compute budgets: count gradient evaluations
and measure total solver time. Reference evaluations/plots are not timed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lsot.projections import sample_projection_bank, PROJECTION_KINDS
from .data import gaussian_example, synthetic_example, image_example, save_gmm
from .reference import shared_initialization, mw2_barycenter
from .solver import optimize_barycenter, mw2_objective, mw2_distance_squared
from .plotting import density_grid, save_comparison, save_convergence
from .run import METHODS, write_csv, write_json, empirical_sw2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=("gaussian", "synthetic", "images"),
                        default=["synthetic"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--adam-steps", type=int, nargs="+", default=[200, 280])
    parser.add_argument("--adam-learning-rate", type=float, default=.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lbfgs-steps", type=int, default=80)
    parser.add_argument("--image-components", type=int, default=12)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/barycenter_optimizer_comparison"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-projections", type=int, default=100)
    args = parser.parse_args(argv)
    if (min(args.L, args.threads, args.image_components, args.eval_samples,
            args.eval_projections, *args.adam_steps) < 1 or min(args.seeds) < 0
            or args.warmup_steps < 0 or args.lbfgs_steps < 0 or args.adam_learning_rate <= 0
            or len(set(args.adam_steps)) != len(args.adam_steps)
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Invalid comparison settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    write_json(args.output_dir / "config.json", vars(args))
    write_json(args.output_dir / "environment.json",
               {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "device": "cpu", "dtype": "float64"})
    # Remove one-time torch optimizer import/construction overhead from the
    # timing comparison; initialization and slice setup are shared/excluded.
    dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    torch.optim.Adam([dummy], lr=args.adam_learning_rate)
    torch.optim.LBFGS([dummy])
    strategies = [(f"Adam{n}", {"optimizer_kind": "adam", "steps": n})
                  for n in args.adam_steps]
    hybrid = f"Adam{args.warmup_steps}_LBFGS{args.lbfgs_steps}"
    strategies.append((hybrid, {"optimizer_kind": "lbfgs", "steps": args.lbfgs_steps,
                                "warmup_steps": args.warmup_steps}))
    rows = []
    with threadpool_limits(limits=args.threads):
        cases = []
        for name in args.cases:
            case = (gaussian_example() if name == "gaussian" else synthetic_example()
                    if name == "synthetic" else image_example(args.output_dir / "image_data",
                        image_dir=args.image_dir, components=args.image_components, seed=0))
            cases.append(case)
        for case in cases:
            weights = np.full(len(case.inputs), 1/len(case.inputs))
            budget = sum(g.count for g in case.inputs)-len(case.inputs)+1
            initial = shared_initialization(case.inputs, weights, budget, seed=0)
            reference = mw2_barycenter(case.inputs, weights).gmm
            x, reference_density = density_grid(reference, case.bounds, case.density_grid)
            base = args.output_dir / case.name
            (base / "inputs").mkdir(parents=True, exist_ok=True)
            for index, gmm in enumerate(case.inputs):
                save_gmm(base / "inputs" / f"input_{index}.npz", gmm)
            save_gmm(base / "initialization.npz", initial)
            write_json(base / "data_manifest.json", case.metadata)
            for seed in args.seeds:
                banks = {kind: sample_projection_bank(2, args.L, kind=kind, seed=seed)
                         for kind in PROJECTION_KINDS}
                for tag, _ in strategies:
                    folder = base / tag / f"seed_{seed}"
                    (folder / "MW2" / "barycenters").mkdir(parents=True, exist_ok=True)
                    save_gmm(folder / "MW2" / "barycenters" / "r00_c00.npz",
                             reference, x=x, density=reference_density)
                    write_json(folder / "config.json", {"tolerance_grad": 1e-7})
                for method in METHODS[1:]:
                    mode, _, kind = method.split("-")
                    for tag, options in strategies:
                        begin = time.perf_counter()
                        result = optimize_barycenter(case.inputs, weights, initial, banks[kind],
                            kind, mode, coordinate_scale=max(1., case.bounds[1]-case.bounds[0]),
                            adam_learning_rate=args.adam_learning_rate, **options)
                        solve_ms = 1000*(time.perf_counter()-begin)
                        row = {"case": case.name, "projection_seed": seed, "method": method,
                               "strategy": tag, "L": args.L, "components": budget,
                               "objective_squared": result.objective,
                               "initial_objective_squared": result.initial_objective,
                               "solve_ms": solve_ms, "iterations": result.iterations,
                               "evaluations": result.evaluations, "gradient_inf": result.gradient_inf,
                               "gradient_evaluations": result.gradient_evaluations,
                               "gradient_tolerance_met": result.gradient_tolerance_met,
                               "status": result.status,
                               "mw2_objective_squared": mw2_objective(case.inputs, weights, result.gmm),
                               "mw2_to_reference": np.sqrt(mw2_distance_squared(result.gmm, reference)),
                               "sw2_to_reference": empirical_sw2(result.gmm, reference,
                                   count=args.eval_samples, directions=args.eval_projections, seed=0)}
                        rows.append(row)
                        folder = base / tag / f"seed_{seed}" / method
                        (folder / "barycenters").mkdir(parents=True, exist_ok=True)
                        (folder / "history").mkdir(exist_ok=True)
                        x, density = density_grid(result.gmm, case.bounds, case.density_grid)
                        save_gmm(folder / "barycenters" / "r00_c00.npz", result.gmm, x=x, density=density)
                        write_csv(folder / "history" / "r00_c00.csv", result.history)
                        write_json(folder / "record.json", row)
                        write_csv(args.output_dir / "comparison.csv", rows)
                        print(f"{case.name} seed={seed} {method} {tag}: "
                              f"F={result.objective:.8g}, {solve_ms:.1f} ms, "
                              f"evals={result.evaluations}, {result.status}", flush=True)
                for tag, _ in strategies:
                    folder = base / tag / f"seed_{seed}"
                    save_comparison(folder, METHODS, 0)
                    save_convergence(folder, METHODS, 0)
    paired = []
    for row in rows:
        if row["strategy"] == hybrid:
            continue
        other = next(r for r in rows if r["case"] == row["case"]
                     and r["projection_seed"] == row["projection_seed"]
                     and r["method"] == row["method"] and r["strategy"] == hybrid)
        paired.append({**row, "objective_gap_to_hybrid_pct":
                       100*(row["objective_squared"]-other["objective_squared"])/other["objective_squared"],
                       "hybrid_over_adam_time": other["solve_ms"]/row["solve_ms"]})
    write_csv(args.output_dir / "paired.csv", paired)
    summary = []
    for case in args.cases:
        for method in METHODS[1:]:
            for tag, _ in strategies:
                selected = [r for r in rows if r["case"] == case and r["method"] == method
                            and r["strategy"] == tag]
                summary.append({"case": case, "method": method, "strategy": tag,
                                "seeds": len(selected), **{key+"_mean": float(np.mean([r[key] for r in selected]))
                                for key in ("objective_squared", "solve_ms", "evaluations",
                                            "gradient_evaluations", "sw2_to_reference")}})
    write_csv(args.output_dir / "summary.csv", summary)
    print(f"Saved {len(rows)} paired center solves to {args.output_dir}", flush=True)
    return rows


if __name__ == "__main__":
    main()
