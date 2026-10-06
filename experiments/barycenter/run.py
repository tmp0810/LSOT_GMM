"""Reproduce MW2 barycenter notebook settings and add eight LSOT methods.

python -m experiments.barycenter.run --output-dir results/barycenter
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lsot.projections import PROJECTION_KINDS, sample_projection_bank
from .data import (gaussian_example, synthetic_example, image_example,
                   grid_nodes, save_gmm, load_gmm)
from .plotting import (density_grid, save_panel, save_inputs, save_method_grid,
                       save_comparison, save_convergence, save_perimeter_animation,
                       save_gaussian_overlay)
from .reference import mw2_barycenter, shared_initialization
from .solver import (optimize_barycenter, on_device, mw2_objective,
                     mw2_distance_squared, BarycenterObjective)

METHODS = ("MW2",) + tuple(f"{mode}-LSOT-{kind}"
                           for mode in ("min", "avg") for kind in PROJECTION_KINDS)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_record(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str)+"\n")
    temporary.replace(path)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def sample_gmm(gmm, count, seed):
    rng = np.random.default_rng(seed)
    weights = gmm.weights.detach().cpu().numpy()
    means = gmm.means.detach().cpu().numpy()
    covs = gmm.covariances.detach().cpu().numpy()
    indices = np.searchsorted(weights.cumsum(), rng.random(count)).clip(max=gmm.count-1)
    normals = rng.standard_normal((count, gmm.dimension))
    return means[indices] + np.einsum("nij,nj->ni", np.linalg.cholesky(covs)[indices], normals)


def empirical_sw2(source, target, *, count, directions, seed):
    first, second = sample_gmm(source, count, seed), sample_gmm(target, count, seed)
    rng = np.random.default_rng(seed+1)
    bank = rng.normal(size=(source.dimension, directions))
    bank /= np.linalg.norm(bank, axis=0, keepdims=True)
    difference = np.sort(first @ bank, axis=0)-np.sort(second @ bank, axis=0)
    return float(np.sqrt(np.mean(difference**2)))


def fingerprint(settings, case):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
    for gmm in case.inputs:
        for tensor in (gmm.weights, gmm.means, gmm.covariances):
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_case(case, args, root):
    folder = root / case.name
    folder.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items()
                if k not in {"output_dir", "no_resume", "image_dir"}}
    settings["schema_version"] = 1
    settings["input_components"] = [source.count for source in case.inputs]
    budget = args.barycenter_components or sum(source.count for source in case.inputs)-len(case.inputs)+1
    settings["effective_barycenter_components"] = budget
    digest = fingerprint(settings, case)
    manifest_path = folder / "run_manifest.json"
    previous_manifest = read_record(manifest_path)
    if previous_manifest and not args.no_resume and previous_manifest["fingerprint"] != digest:
        raise ValueError(f"Settings changed in {folder}; use a new output directory or --no-resume")
    write_json(manifest_path, {"fingerprint": digest, "settings": settings})
    input_folder = folder / "inputs"
    input_folder.mkdir(exist_ok=True)
    for index, source in enumerate(case.inputs):
        save_gmm(input_folder / f"input_{index}.npz", source)
    save_inputs(input_folder, case)
    write_json(folder / "data_manifest.json", case.metadata)
    nodes = [(0, 0, None, None, np.ones(3)/3)] if case.name == "gaussian" else grid_nodes(args.grid_size)
    size = 1 if case.name == "gaussian" else args.grid_size
    (folder / "config.json").write_text(json.dumps(settings, indent=2, default=str)+"\n")
    device_inputs = [on_device(source, args.device) for source in case.inputs]
    kinds = [kind for kind in PROJECTION_KINDS if any(name.endswith("-"+kind) for name in args.methods)]
    banks = {kind: sample_projection_bank(case.inputs[0].dimension, args.L, kind=kind,
                                         seed=args.seed, device=args.device) for kind in kinds}
    for kind, bank in banks.items():
        np.savez_compressed(folder / f"bank_{kind}.npz", **{
            field.name: getattr(bank, field.name).detach().cpu().numpy() for field in fields(bank)})
    rows, grouped = [], {method: [] for method in args.methods}
    for row, col, tx, ty, weights in nodes:
        tag = f"r{row:02d}_c{col:02d}"
        reference_folder = folder / "MW2"
        for child in ("barycenters", "records", "images", "history"):
            (reference_folder / child).mkdir(parents=True, exist_ok=True)
        reference_file = reference_folder / "barycenters" / f"{tag}.npz"
        reference_record = reference_folder / "records" / f"{tag}.json"
        reference = None
        if not args.no_resume and reference_file.exists() and reference_record.exists():
            previous = read_record(reference_record)
            if previous and previous["fingerprint"] != digest:
                raise ValueError(f"Settings changed in {folder}; use a new output directory or --no-resume")
            if previous:
                reference = load_gmm(reference_file, device=args.device)
                ref_record = previous
        if reference is None:
            print(f"{case.name} {tag}: MW2 weights={np.round(weights, 4).tolist()}", flush=True)
            start = time.perf_counter()
            solved = mw2_barycenter(case.inputs, weights, iterations=args.gaussian_iterations)
            elapsed = 1000*(time.perf_counter()-start)
            reference = on_device(solved.gmm, args.device)
            ref_record = {"fingerprint": digest, "case": case.name, "row": row, "col": col,
                          "tx": tx, "ty": ty, "method": "MW2", "projection": "", "aggregation": "MW2",
                          "input_weights": json.dumps(weights.tolist()), "components": reference.count,
                          "L": 0, "barycenter_component_budget": 0,
                          "initial_objective_squared": None, "objective_squared": solved.cost_squared,
                          "mw2_objective_squared": mw2_objective(device_inputs, weights, reference),
                          "mw2_to_reference": 0., "sw2_to_reference": 0., "density_l1_to_reference": 0.,
                          "initialization_ms": 0., "solve_ms": elapsed, "total_ms": elapsed,
                          "iterations": args.gaussian_iterations, "evaluations": 0,
                          "status": "anchor" if np.count_nonzero(weights) == 1 else "multimarginal_lp",
                          "marginal_error": solved.marginal_error}
            x, density = density_grid(reference, case.bounds, case.density_grid)
            save_gmm(reference_file, reference, x=x, density=density, input_weights=weights)
            write_json(reference_record, ref_record)
        with np.load(reference_file) as data:
            x, reference_density = data["x"], data["density"]
        save_panel(reference_folder / "images" / f"{tag}.png", x, reference_density,
                   show_axes=args.show_axes)
        if "MW2" in args.methods:
            rows.append(ref_record)
            grouped["MW2"].append(ref_record)
        initialization, initialization_ms = None, 0.
        for method in args.methods:
            if method == "MW2":
                continue
            method_folder = folder / method
            for child in ("barycenters", "records", "images", "history"):
                (method_folder / child).mkdir(parents=True, exist_ok=True)
            record_file = method_folder / "records" / f"{tag}.json"
            gmm_file = method_folder / "barycenters" / f"{tag}.npz"
            if not args.no_resume and record_file.exists() and gmm_file.exists():
                record = read_record(record_file)
                if record and record["fingerprint"] != digest:
                    raise ValueError(f"Settings changed in {method_folder}; use a new output directory or --no-resume")
                if record:
                    rows.append(record)
                    grouped[method].append(record)
                    with np.load(gmm_file) as data:
                        save_panel(method_folder / "images" / f"{tag}.png", data["x"], data["density"],
                                   show_axes=args.show_axes)
                    continue
            mode, _, kind = method.split("-")
            anchor = np.count_nonzero(weights) == 1
            if anchor:
                candidate, elapsed, init_value, value = reference, 0., 0., 0.
                iterations, evaluations, status, history = 0, 0, "anchor", []
            else:
                if initialization is None:
                    start = time.perf_counter()
                    initial = shared_initialization(case.inputs, weights, budget,
                                                    iterations=args.gaussian_iterations, seed=args.seed)
                    initialization = on_device(initial, args.device)
                    synchronize(args.device)
                    initialization_ms = 1000*(time.perf_counter()-start)
                    init_dir = folder / "initializations"
                    init_dir.mkdir(exist_ok=True)
                    save_gmm(init_dir / f"{tag}.npz", initialization, input_weights=weights)
                print(f"{case.name} {tag}: {method}, Kb={budget}, L={args.L}", flush=True)
                synchronize(args.device)
                start = time.perf_counter()
                result = optimize_barycenter(
                    device_inputs, weights, initialization, banks[kind], kind, mode,
                    steps=args.steps, learning_rate=args.learning_rate, history_size=args.history_size,
                    tolerance_grad=args.tolerance_grad, tolerance_change=args.tolerance_change,
                    coordinate_scale=max(1., case.bounds[1]-case.bounds[0]),
                    learn_weights=not args.freeze_weights)
                synchronize(args.device)
                elapsed = 1000*(time.perf_counter()-start)
                candidate, init_value, value = result.gmm, result.initial_objective, result.objective
                iterations, evaluations, status, history = result.iterations, result.evaluations, result.status, result.history
            x, density = density_grid(candidate, case.bounds, case.density_grid)
            d1 = density/density.sum()
            d2 = reference_density/reference_density.sum()
            record = {"fingerprint": digest, "case": case.name, "row": row, "col": col,
                      "tx": tx, "ty": ty, "method": method, "projection": kind, "aggregation": mode,
                      "input_weights": json.dumps(weights.tolist()), "components": candidate.count,
                      "L": args.L, "barycenter_component_budget": budget,
                      "initial_objective_squared": init_value, "objective_squared": value,
                      "mw2_objective_squared": mw2_objective(device_inputs, weights, candidate),
                      "mw2_to_reference": np.sqrt(max(0., mw2_distance_squared(reference, candidate))),
                      "sw2_to_reference": empirical_sw2(reference, candidate, count=args.eval_samples,
                                                         directions=args.eval_projections, seed=args.seed+100000),
                      "density_l1_to_reference": float(np.abs(d1-d2).sum()),
                      "initialization_ms": 0. if anchor else initialization_ms, "solve_ms": elapsed,
                      "total_ms": elapsed+(0. if anchor else initialization_ms),
                      "iterations": iterations, "evaluations": evaluations, "status": status,
                      "marginal_error": None}
            save_gmm(gmm_file, candidate, x=x, density=density, input_weights=weights)
            save_panel(method_folder / "images" / f"{tag}.png", x, density, show_axes=args.show_axes)
            write_csv(method_folder / "history" / f"{tag}.csv", history)
            write_json(record_file, record)
            rows.append(record)
            grouped[method].append(record)
            print(f"  loss {init_value:.6g} -> {value:.6g}; {status}; {elapsed:.1f} ms", flush=True)
        write_csv(folder / "metrics.csv", rows)
    for method, records in grouped.items():
        save_method_grid(folder / method, records, size)
        save_perimeter_animation(folder / method, size, mp4=args.mp4)
        if case.name == "gaussian":
            barycenter = load_gmm(folder / method / "barycenters" / "r00_c00.npz")
            save_gaussian_overlay(folder / method, case, barycenter)
    save_comparison(folder, args.methods, size//2)
    save_convergence(folder, args.methods, size//2)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/barycenter"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cases", nargs="+", choices=("gaussian", "synthetic", "images"),
                        default=["gaussian", "synthetic", "images"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--image-dir", type=Path, help="Directory containing original redcross/duck/star/batman PNGs")
    parser.add_argument("--image-components", type=int, default=12)
    parser.add_argument("--barycenter-components", type=int, default=0,
                        help="0: sum(Kj)-J+1 (MW2 LP basic-support bound); ignored at exact corners")
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--gaussian-iterations", type=int, default=10)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1.)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--tolerance-grad", type=float, default=1e-7)
    parser.add_argument("--tolerance-change", type=float, default=1e-10)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-projections", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--freeze-weights", action="store_true")
    parser.add_argument("--show-axes", action="store_true", help="Numeric ticks on individual panels, without labels/titles")
    parser.add_argument("--mp4", action="store_true", help="Also export MP4 perimeter movies (requires ffmpeg); GIF is automatic")
    parser.add_argument("--no-resume", action="store_true")
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        supplied = json.loads(preliminary.config.read_text())
        unknown = set(supplied)-{action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**supplied)
    args = parser.parse_args(argv)
    args.output_dir, args.config = Path(args.output_dir), str(args.config) if args.config else None
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; select a GPU runtime or use --device cpu")
    if (min(args.L, args.gaussian_iterations, args.eval_samples, args.eval_projections,
            args.image_components, args.threads, args.history_size) < 1
            or args.grid_size < 2 or args.barycenter_components < 0 or args.steps < 0
            or args.seed < 0 or args.learning_rate <= 0 or args.tolerance_grad < 0 or args.tolerance_change < 0
            or len(set(args.methods)) != len(args.methods)):
        parser.error("Invalid counts, optimizer settings, or duplicate methods")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "device": args.device}
    (args.output_dir / "environment.json").write_text(json.dumps(versions, indent=2)+"\n")
    with threadpool_limits(limits=args.threads):
        cases = []
        if "gaussian" in args.cases:
            cases.append(gaussian_example())
        if "synthetic" in args.cases:
            cases.append(synthetic_example())
        if "images" in args.cases:
            cases.append(image_example(args.output_dir / "image_data", image_dir=args.image_dir,
                                       components=args.image_components, seed=args.seed))
        rows = []
        for case in cases:
            rows.extend(run_case(case, args, args.output_dir))
            write_csv(args.output_dir / "metrics.csv", rows)
    summaries = []
    for case in cases:
        for method in args.methods:
            values = [r for r in rows if r["case"] == case.name and r["method"] == method and r["status"] != "anchor"]
            if not values:
                continue
            summaries.append({"case": case.name, "method": method, "nodes": len(values),
                              **{key+"_mean": float(np.mean([r[key] for r in values]))
                                 for key in ("total_ms", "objective_squared", "mw2_objective_squared",
                                             "mw2_to_reference", "sw2_to_reference", "density_l1_to_reference")},
                              "numerical_stops": sum(r["status"].startswith("numerical_stop") for r in values)})
    write_csv(args.output_dir / "summary.csv", summaries)
    print(f"Saved {len(rows)} method/node results to {args.output_dir}", flush=True)
    return rows


if __name__ == "__main__":
    main()
