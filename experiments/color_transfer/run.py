"""Compare MW2 with averaged, minimum, top-k, and optimized minimum LSOT.

python -m experiments.color_transfer.run --config experiments/color_transfer/configs/smoke.json
"""
import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lsot import (GMM, BarycentricMap, average_lsot, minimum_lsot, topk_lsot,
                  optimized_minimum_lsot, solve_mw2)
from lsot.projections import PROJECTION_KINDS, sample_projection_bank
from .data import download_reference_images, fit_gmm, guided_output, read_rgb, save_rgb
from .evaluation import ColorEvaluator, benchmark, save_comparison, synchronize, time_summary


@dataclass
class Config:
    source: str = "data/color_transfer/renoir.jpg"
    target: str = "data/color_transfer/gauguin.jpg"
    download_reference: bool = True
    output_dir: str = "results/color_transfer/mw2_reference"
    components_source: int = 10
    components_target: int = 10
    projection_counts: list = field(default_factory=lambda: [10, 50, 100])
    projection_kinds: list = field(default_factory=lambda: list(PROJECTION_KINDS))
    aggregations: list = field(default_factory=lambda: ["avg", "min", "min-opt"])
    opt_steps: int = 20
    opt_samples: int = 8
    opt_epsilon: float = 0.05
    opt_learning_rate: float = 0.05
    opt_max_gradient_norm: float = 10.0
    seeds: list = field(default_factory=lambda: [0])
    projection_seed: int | list[int] = 20260909
    device: str = "auto"
    dtype: str = "float64"
    max_side: int | None = None
    fit_pixels: int | None = None
    em_max_iter: int = 100
    em_tol: float = 1e-3
    reg_covar: float = 1e-6
    repeats: int = 5
    warmups: int = 1
    map_repeats: int = 3
    batch_size: int = 16384
    threads: int = 1
    guided_filter: bool = True
    guided_radius: int = 10
    guided_epsilon: float = 1e-4
    eval_samples: int = 4096
    eval_projections: int = 128
    eval_w2_samples: int = 512
    eval_seed: int = 42
    save_raw: bool = False

    def projection_seed_values(self):
        """Scalar keeps the legacy layout; a list requests a projection sweep."""
        values = self.projection_seed if isinstance(self.projection_seed, list) else [self.projection_seed]
        if (not values or any(type(value) is not int or value < 0 for value in values)
                or len(set(values)) != len(values)):
            raise ValueError("projection_seed must be a nonnegative integer or a nonempty list of distinct nonnegative integers")
        return values

    def validate(self):
        self.projection_seed_values()
        if not self.projection_counts or any(n < 1 for n in self.projection_counts):
            raise ValueError("projection_counts must contain positive integers")
        if (not self.projection_kinds or len(set(self.projection_kinds)) != len(self.projection_kinds)
                or not set(self.projection_kinds) <= set(PROJECTION_KINDS)):
            raise ValueError(f"projection_kinds must contain distinct values from {PROJECTION_KINDS}")
        if (not self.aggregations or len(set(self.aggregations)) != len(self.aggregations)
                or not set(self.aggregations) <= {"avg", "min", "min-opt", "top2", "top3", "top4"}):
            raise ValueError("aggregations must contain distinct values from 'avg', 'min', 'min-opt', 'top2', 'top3', 'top4'")
        if any(int(mode[-1]) > count for mode in self.aggregations if mode.startswith("top")
               for count in self.projection_counts):
            raise ValueError("each projection count must be at least the requested top-k")
        if (type(self.opt_steps) is not int or self.opt_steps < 0
                or type(self.opt_samples) is not int or self.opt_samples < 1
                or not np.isfinite([self.opt_epsilon, self.opt_learning_rate,
                                    self.opt_max_gradient_norm]).all()
                or min(self.opt_epsilon, self.opt_learning_rate, self.opt_max_gradient_norm) <= 0):
            raise ValueError("invalid min-opt projection optimization settings")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and distinct")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if min(self.components_source, self.components_target, self.repeats,
               self.map_repeats, self.batch_size, self.threads, self.em_max_iter) < 1:
            raise ValueError("component counts, repetitions and batch/thread sizes must be positive")
        if self.warmups < 0 or self.reg_covar <= 0 or self.em_tol <= 0:
            raise ValueError("invalid warmups or EM tolerances")
        if self.guided_radius < 0 or self.guided_epsilon <= 0:
            raise ValueError("invalid guided-filter parameters")
        if any(type(value) is not int or value < 1 for value in
               (self.eval_samples, self.eval_projections, self.eval_w2_samples)):
            raise ValueError("eval_samples, eval_projections, and eval_w2_samples must be positive integers")


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_rows(path, rows, delimiter=","):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _save_plan(path, plan):
    np.savez_compressed(path, rows=_numpy(plan.rows), cols=_numpy(plan.cols),
                        mass=_numpy(plan.mass), shape=np.asarray(plan.shape))


def _plan_rmse(plan, reference):
    indices = torch.stack((torch.cat((plan.rows, reference.rows)), torch.cat((plan.cols, reference.cols))))
    values = torch.cat((plan.mass, -reference.mass))
    difference = torch.sparse_coo_tensor(indices, values, plan.shape).coalesce().values()
    return float(torch.sqrt(difference.square().sum() / (plan.shape[0] * plan.shape[1])).item())


def _prepare_banks(config, seed, device, dtype, folder, *, projection_seed=None):
    """Sample once per family; preserve the existing shared Mix/SMix bank.

    Separate local generators keep old parameter banks unchanged when new
    families are enabled. Each family reuses its bank for all aggregations
    and all prefix budgets. Only bank preparation is timed here.
    """
    projection_seed = config.projection_seed if projection_seed is None else projection_seed
    banks, times, files = {}, {}, {}
    groups = {"Mix": "parameter", "SMix": "parameter", "B": "B", "B1D": "B1D"}
    prepared = {}
    for kind in config.projection_kinds:
        group = groups[kind]
        if group not in prepared:
            synchronize(device)
            start = time.perf_counter()
            bank = sample_projection_bank(3, max(config.projection_counts), kind=kind,
                                          seed=projection_seed + seed, device=device, dtype=dtype)
            synchronize(device)
            elapsed = 1000 * (time.perf_counter() - start)
            filename = "projection_bank.npz" if group == "parameter" else f"projection_bank_{group}.npz"
            np.savez_compressed(folder / filename, **{k: _numpy(v) for k, v in vars(bank).items()})
            prepared[group] = bank, elapsed, filename
        banks[kind], times[kind], files[kind] = prepared[group]
    return banks, times, files


def _summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["K0"], row["K1"], row["method"], row["L"]), []).append(row)
    summary = []
    metrics = ["transport_ms_mean", "map_setup_ms_mean", "map_apply_ms_mean", "pipeline_ms",
               "cost_squared", "relative_cost_gap", "plan_rmse", "color_sw2", "guided_color_sw2",
               "color_w2", "guided_color_w2", "identity_color_sw2", "identity_color_w2"]
    for (_, _, method, count), group in groups.items():
        first = group[0]
        entry = {"method": method, "L": count, "K0": first["K0"], "K1": first["K1"],
                 "projection_family": first["projection_family"],
                 "aggregation": first["aggregation"],
                 "device": first["device"],
                 "n_seeds": len({r["seed"] for r in group}),
                 "n_projection_seeds": len({r.get("projection_seed") for r in group
                                            if r.get("projection_seed") is not None}),
                 "n_runs": len(group)}
        for name in metrics:
            values = [r[name] for r in group if r[name] is not None]
            entry[name + "_across_seeds_mean"] = float(np.mean(values)) if values else None
            entry[name + "_across_seeds_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(entry)
    return summary


def _projection_seed_summary(rows):
    """Separate bank seeds; MW2 is listed once with an empty projection seed.

    Within each bank seed, statistics are across data seeds only. The main
    summary instead pools (data seed, projection seed) runs for LSOT; its
    standard deviations are descriptive, not independent-replicate errors.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row.get("projection_seed"), []).append(row)
    return [{"projection_seed": seed, **entry}
            for seed, group in groups.items() for entry in _summarize(group)]


def run_experiment(config):
    """Write images, sparse plans, saved GMMs/banks, timings and evaluation data."""
    config.validate()
    projection_seeds = config.projection_seed_values()
    seed_sweep = isinstance(config.projection_seed, list)
    device_name = ("cuda:0" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("supported devices are cpu and cuda")
    dtype = getattr(torch, config.dtype)
    torch.set_num_threads(config.threads)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in (Path(config.source), Path(config.target)):
        if not input_path.exists() and config.download_reference and input_path.name in {"renoir.jpg", "gauguin.jpg"}:
            download_reference_images(input_path.parent)
        if not input_path.is_file():
            raise FileNotFoundError(f"Image not found: {input_path}")
    source_image = read_rgb(config.source, config.max_side)
    target_image = read_rgb(config.target, config.max_side)
    x, y = source_image.reshape(-1, 3), target_image.reshape(-1, 3)
    save_rgb(output_dir / "source.png", source_image)
    save_rgb(output_dir / "target.png", target_image)
    _write_json(output_dir / "config.json", asdict(config))
    packages = ["numpy", "scipy", "scikit-learn", "POT", "torch", "Pillow"]
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "device": str(device), "dtype": config.dtype,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_shape": list(source_image.shape), "target_shape": list(target_image.shape),
        "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in (config.source, config.target)},
        "timing": "Wall clock with CUDA synchronization. Transport includes cost evaluation. "
                  "Minimum LSOT includes constructing and scoring all candidate lifts. "
                  "Optimized minimum additionally includes every Stein-gradient perturbation and update. "
                  "Banks are sampled before timing and bank sampling is reported separately. "
                  "Pipeline excludes file I/O, plotting and evaluation; includes GMM fitting, "
                  "input transfers, transport, map setup/application and optional guided filtering.",
        "mw2_backend": "Original gmmot.GW2: SciPy and POT CPU; transfers included.",
        "plan_reference": "MW2 component LP, not full distribution-space W2",
        "minimum_selection": "One lift for the entire GMM pair, minimizing true Gaussian W2-squared "
                             "cost over the shared finite bank. Zero-based index; first index on exact ties.",
        "topk_selection": "Select the 2, 3, or 4 lifts with lowest true Gaussian squared cost "
                          "from the same finite bank, then average their plans with equal weights. "
                          "Exact cost ties use the earliest bank indices.",
        "optimized_selection": "Initialize from the minimum of the same L bank; use Gaussian-smoothed "
                               "Stein-gradient steps, retain the best actually evaluated lifted plan. "
                               "Optimization is heuristic and need not reach the global infimum.",
        "projection_families": config.projection_kinds,
        "projection_seed_policy": "projection_seed accepts an integer or list. Effective bank seed "
                                  "equals projection_seed + data seed (legacy convention). GMMs, "
                                  "evaluation samples and MW2 are shared across bank seeds. "
                                  "paper_results.tsv pools LSOT runs; projection_seed_results.tsv "
                                  "separates bank seeds. Pooled standard deviations are descriptive.",
        "distribution_projections": "Bonet et al. Gaussian ray laws: B at base N(0,I_3), "
                                    "B1D at base N(0,1), with full theta.T Sigma theta. "
                                    "Upstream revision: 5bb8a254f9c340a7a37af14218b7d6b06130e3d0.",
        "color_evaluation": "Root SW2 and exact empirical W2 on clipped float RGB before 8-bit "
                            "PNG quantization. W2 uses POT emd2 and an equal-size prefix of the "
                            "shared sampled source/target pixels; it is not the GMM MW2 cost. "
                            "Evaluation is excluded from timed pipeline metrics.",
    }
    _write_json(output_dir / "metadata.json", metadata)
    all_rows = []
    with threadpool_limits(limits=config.threads), torch.inference_mode():
        for seed in config.seeds:
            folder = output_dir / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Fitting shared RGB GMMs: K0={config.components_source}, K1={config.components_target}, seed={seed}", flush=True)
            fit_options = dict(fit_pixels=config.fit_pixels, max_iter=config.em_max_iter,
                               tol=config.em_tol, reg_covar=config.reg_covar)
            sx, fit_x, count_x = fit_gmm(x, config.components_source, seed, **fit_options)
            sy, fit_y, count_y = fit_gmm(y, config.components_target, seed + 1, **fit_options)
            np.savez_compressed(folder / "gmms.npz", alpha=sx.weights_, means_source=sx.means_,
                                covariances_source=sx.covariances_, beta=sy.weights_,
                                means_target=sy.means_, covariances_target=sy.covariances_)
            start = time.perf_counter()
            source = GMM.from_numpy(sx.weights_, sx.means_, sx.covariances_, device=device, dtype=dtype)
            target = GMM.from_numpy(sy.weights_, sy.means_, sy.covariances_, device=device, dtype=dtype)
            points = torch.as_tensor(x, dtype=dtype, device=device)
            synchronize(device)
            input_ms = 1000 * (time.perf_counter() - start)
            evaluator = ColorEvaluator.create(x, y, samples=config.eval_samples,
                                              projections=config.eval_projections,
                                              w2_samples=config.eval_w2_samples,
                                              seed=config.eval_seed + seed)
            identity_color_sw2 = evaluator.sw2(source_image)
            identity_color_w2 = evaluator.w2(source_image)
            np.savez_compressed(folder / "evaluation_bank.npz", source_indices=evaluator.source_indices,
                                target_indices=evaluator.target_indices, directions=evaluator.directions,
                                w2_source_indices=evaluator.w2_source_indices,
                                w2_target_indices=evaluator.w2_target_indices)
            outputs, guided_outputs, timing_rows = [], [], {}
            reference_plan, reference_cost, reference_output = None, None, None
            active_projection_seed = None
            jobs = [("MW2", 0, None, None)] + [
                (kind, count, aggregation, projection_seed) for projection_seed in projection_seeds
                for count in sorted(set(config.projection_counts))
                for aggregation in config.aggregations for kind in config.projection_kinds]
            for kind, count, aggregation, projection_seed in jobs:
                prefix = {"avg": "", "min": "min-", "min-opt": "min-opt-",
                          "top2": "top2-", "top3": "top3-", "top4": "top4-"}
                method = "MW2" if kind == "MW2" else f"{prefix[aggregation]}LSOT-{kind}"
                name = method if count == 0 else f"{method}_L{count}"
                method_folder = folder
                display_name = name
                if kind != "MW2":
                    if seed_sweep:
                        method_folder = folder / f"projection_seed_{projection_seed}"
                        display_name = f"{name}_P{projection_seed}"
                    method_folder.mkdir(parents=True, exist_ok=True)
                    if active_projection_seed != projection_seed:
                        banks, bank_times, bank_files = _prepare_banks(
                            config, seed, device, dtype, method_folder, projection_seed=projection_seed)
                        active_projection_seed = projection_seed
                if kind == "MW2":
                    def solver():
                        plan, cost = solve_mw2(source, target)
                        return plan, cost, None
                else:
                    selected_bank = banks[kind].prefix(count)

                    def solver():
                        if aggregation == "min":
                            result = minimum_lsot(source, target, selected_bank, kind)
                            return result.plan, float(result.cost_squared.item()), result
                        if aggregation == "min-opt":
                            result = optimized_minimum_lsot(
                                source, target, selected_bank, kind,
                                steps=config.opt_steps, samples=config.opt_samples,
                                epsilon=config.opt_epsilon, learning_rate=config.opt_learning_rate,
                                max_gradient_norm=config.opt_max_gradient_norm,
                                seed=projection_seed + seed)
                            return result.plan, float(result.cost_squared.item()), result
                        if aggregation.startswith("top"):
                            result = topk_lsot(source, target, selected_bank, kind, int(aggregation[-1]))
                            return result.plan, float(result.cost_squared.item()), result
                        plan = average_lsot(source, target, selected_bank, kind)
                        return plan, float(plan.cost(source, target).item()), None

                (plan, cost, selection), transport_times = benchmark(
                    solver, device=device, repeats=config.repeats, warmups=config.warmups)
                marginal_error = plan.marginal_error(source, target)
                feasibility_tol = 1e-8 if dtype == torch.float64 else 2e-5
                if marginal_error > feasibility_tol or not np.isfinite(cost):
                    raise RuntimeError(f"Invalid {name} result: marginal error={marginal_error}, cost={cost}")
                if kind == "MW2":
                    reference_plan, reference_cost = plan, cost
                if cost < reference_cost - 1e-7 * max(1.0, abs(reference_cost)):
                    raise RuntimeError("Lifted cost is below the MW2 optimum beyond numerical tolerance")
                mapper, setup_times = benchmark(lambda: BarycentricMap.from_plan(source, target, plan),
                                                device=device, repeats=config.repeats, warmups=config.warmups)
                mapped, apply_times = benchmark(
                    lambda: _numpy(mapper.transform(points, batch_size=config.batch_size)).reshape(source_image.shape),
                    device=device, repeats=config.map_repeats, warmups=config.warmups)
                if not np.isfinite(mapped).all():
                    raise RuntimeError(f"Nonfinite output from {name}")
                if kind == "MW2":
                    reference_output = mapped.copy()
                post_ms, filtered = 0.0, None
                if config.guided_filter:
                    start = time.perf_counter()
                    filtered = guided_output(source_image, mapped, config.guided_radius, config.guided_epsilon)
                    post_ms = 1000 * (time.perf_counter() - start)
                    save_rgb(method_folder / f"{name}_guided.png", filtered)
                    guided_outputs.append((display_name, filtered))
                save_rgb(method_folder / f"{name}.png", mapped)
                if config.save_raw:
                    np.save(method_folder / f"{name}_raw.npy", mapped)
                _save_plan(method_folder / f"{name}_plan.npz", plan)
                if aggregation == "min":
                    np.savez_compressed(method_folder / f"{name}_selection.npz",
                                        projection_index=selection.projection_index,
                                        projection_costs=_numpy(selection.projection_costs))
                elif aggregation is not None and aggregation.startswith("top"):
                    np.savez_compressed(method_folder / f"{name}_selection.npz",
                                        projection_indices=_numpy(selection.projection_indices),
                                        projection_costs=_numpy(selection.projection_costs))
                elif aggregation == "min-opt":
                    np.savez_compressed(
                        method_folder / f"{name}_selection.npz",
                        initial_projection_index=selection.initial_projection_index,
                        initial_projection_costs=_numpy(selection.initial_projection_costs),
                        initial_cost_squared=float(selection.initial_cost_squared.item()),
                        best_iteration=selection.best_iteration,
                        cost_history=_numpy(selection.cost_history),
                        **{f"best_{key}": _numpy(value) for key, value in vars(selection.best_bank).items()})
                outputs.append((display_name, mapped))
                transport_mean, transport_std = time_summary(transport_times)
                setup_mean, setup_std = time_summary(setup_times)
                apply_mean, apply_std = time_summary(apply_times)
                row = {
                    "seed": seed, "projection_seed": projection_seed,
                    "effective_projection_seed": projection_seed + seed if projection_seed is not None else None,
                    "method": method, "K0": source.count, "K1": target.count,
                    "d": 3, "L": count, "device": str(device), "dtype": config.dtype,
                    "aggregation": aggregation,
                    "projection_family": kind if kind != "MW2" else None,
                    "projection_bank": str((method_folder / bank_files[kind]).relative_to(folder)) if kind != "MW2" else None,
                    "selected_projection": selection.projection_index if aggregation == "min" else None,
                    "selected_projections": (_numpy(selection.projection_indices).tolist()
                                             if aggregation is not None and aggregation.startswith("top") else None),
                    "initial_projection": (selection.initial_projection_index if aggregation == "min-opt" else None),
                    "best_iteration": selection.best_iteration if aggregation == "min-opt" else None,
                    "solver_backend": "scipy+POT(cpu)" if kind == "MW2" else "torch",
                    "source_pixels": len(x), "target_pixels": len(y),
                    "fit_source_pixels": count_x, "fit_target_pixels": count_y,
                    "em_source_converged": bool(sx.converged_), "em_target_converged": bool(sy.converged_),
                    "fit_ms": fit_x + fit_y, "input_setup_ms": input_ms,
                    "bank_sampling_ms": bank_times[kind] if kind != "MW2" else 0.0,
                    "transport_ms_mean": transport_mean, "transport_ms_std": transport_std,
                    "map_setup_ms_mean": setup_mean, "map_setup_ms_std": setup_std,
                    "map_apply_ms_mean": apply_mean, "map_apply_ms_std": apply_std,
                    "guided_filter_ms": post_ms,
                    "pipeline_ms": fit_x + fit_y + input_ms + transport_mean + setup_mean + apply_mean + post_ms,
                    "cost_squared": cost, "mw2_cost_squared": reference_cost,
                    "relative_cost_gap": (cost - reference_cost) / reference_cost if reference_cost > 1e-15 else None,
                    "plan_rmse": _plan_rmse(plan, reference_plan), "nnz": plan.nnz,
                    "marginal_l1_error": marginal_error,
                    "map_rmse_to_mw2": float(np.sqrt(np.mean((mapped - reference_output) ** 2))),
                    "color_sw2": evaluator.sw2(mapped),
                    "identity_color_sw2": identity_color_sw2,
                    "identity_color_w2": identity_color_w2,
                    "guided_color_sw2": evaluator.sw2(filtered) if filtered is not None else None,
                    "color_w2": evaluator.w2(mapped),
                    "guided_color_w2": evaluator.w2(filtered) if filtered is not None else None,
                    "clipped_channel_fraction": float(np.mean((mapped < 0) | (mapped > 1))),
                }
                all_rows.append(row)
                timing_rows[display_name] = {"transport_ms": transport_times, "map_setup_ms": setup_times,
                                     "map_apply_ms": apply_times}
                # Save after each method so a long run retains completed results.
                _write_rows(output_dir / "metrics.csv", all_rows)
                _write_json(folder / "timings.json", timing_rows)
                print(f"  {display_name}: cost={cost:.6g}, transport={transport_mean:.2f} ms, "
                      f"color SW2={row['color_sw2']:.6g}, color W2={row['color_w2']:.6g}", flush=True)
            save_comparison(folder / "comparison.png", source_image, target_image, outputs)
            if guided_outputs:
                save_comparison(folder / "comparison_guided.png", source_image, target_image, guided_outputs)
    _write_rows(output_dir / "paper_results.tsv", _summarize(all_rows), delimiter="\t")
    _write_rows(output_dir / "projection_seed_results.tsv", _projection_seed_summary(all_rows), delimiter="\t")
    print(f"Saved results to {output_dir.resolve()}", flush=True)
    return all_rows


def build_parser(description=__doc__):
    """Common CLI options for a single K setting and a component-count sweep."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "mw2_reference.json")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--components", type=int, nargs="+", help="K, or K0 K1")
    parser.add_argument("--projections", type=int, nargs="+")
    parser.add_argument("--projection-kinds", choices=PROJECTION_KINDS, nargs="+",
                        help="projection families; defaults to Mix SMix B B1D")
    parser.add_argument("--aggregations", choices=["avg", "min", "min-opt", "top2", "top3", "top4"], nargs="+",
                        help="LSOT aggregation(s), overriding the config")
    parser.add_argument("--opt-steps", type=int, help="Stein-gradient updates per min-opt call")
    parser.add_argument("--opt-samples", type=int, help="Gaussian perturbations per update")
    parser.add_argument("--opt-epsilon", type=float, help="Gaussian perturbation scale")
    parser.add_argument("--opt-learning-rate", type=float, help="Stein-gradient step size")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--projection-seeds", type=int, nargs="+",
                        help="sweep projection seeds on each shared GMM; overrides projection_seed")
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--fit-pixels", type=int)
    parser.add_argument("--eval-w2-samples", type=int,
                        help="number of sampled RGB pixels for exact empirical W2 evaluation")
    return parser


def config_from_arguments(args):
    """Load the JSON setting, then apply explicitly provided CLI overrides."""
    values = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("source", "target", "output_dir", "device", "seeds", "max_side", "fit_pixels",
                "aggregations", "projection_kinds", "opt_steps", "opt_samples",
                "opt_epsilon", "opt_learning_rate", "eval_w2_samples"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.components:
        if len(args.components) not in {1, 2}:
            raise ValueError("--components expects K or K0 K1")
        values["components_source"] = args.components[0]
        values["components_target"] = args.components[-1]
    if args.projections:
        values["projection_counts"] = args.projections
    if args.projection_seeds is not None:
        values["projection_seed"] = args.projection_seeds
    return Config(**values)


def main():
    run_experiment(config_from_arguments(build_parser().parse_args()))


if __name__ == "__main__":
    main()
