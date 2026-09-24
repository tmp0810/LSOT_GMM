"""Compare preserved MW2 with averaged and minimum LSOT-Mix/SMix color transfer.

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

from lsot import GMM, BarycentricMap, average_lsot, minimum_lsot, solve_mw2
from param_proj import sample_projection_bank
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
    aggregations: list = field(default_factory=lambda: ["avg", "min"])
    seeds: list = field(default_factory=lambda: [0])
    projection_seed: int = 20260909
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
    eval_seed: int = 42
    save_raw: bool = False

    def validate(self):
        if not self.projection_counts or any(n < 1 for n in self.projection_counts):
            raise ValueError("projection_counts must contain positive integers")
        if (not self.aggregations or len(set(self.aggregations)) != len(self.aggregations)
                or not set(self.aggregations) <= {"avg", "min"}):
            raise ValueError("aggregations must contain distinct values from 'avg', 'min'")
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


def _summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["L"]), []).append(row)
    summary = []
    metrics = ["transport_ms_mean", "map_setup_ms_mean", "map_apply_ms_mean", "pipeline_ms",
               "cost_squared", "relative_cost_gap", "plan_rmse", "color_sw2", "guided_color_sw2"]
    for (method, count), group in groups.items():
        first = group[0]
        entry = {"method": method, "L": count, "K0": first["K0"], "K1": first["K1"],
                 "aggregation": first["aggregation"],
                 "device": first["device"], "n_seeds": len(group)}
        for name in metrics:
            values = [r[name] for r in group if r[name] is not None]
            entry[name + "_across_seeds_mean"] = float(np.mean(values)) if values else None
            entry[name + "_across_seeds_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(entry)
    return summary


def run_experiment(config):
    """Write images, sparse plans, saved GMMs/banks, timings and evaluation data."""
    config.validate()
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
                  "Minimum LSOT includes constructing and scoring all candidate lifts and selecting the best. "
                  "Banks are sampled before timing and bank sampling is reported separately. "
                  "Pipeline excludes file I/O, plotting and evaluation; includes GMM fitting, "
                  "input transfers, transport, map setup/application and optional guided filtering.",
        "mw2_backend": "Original gmmot.GW2: SciPy and POT CPU; transfers included.",
        "plan_reference": "MW2 component LP, not full distribution-space W2",
        "minimum_selection": "One lift for the entire GMM pair, minimizing true Gaussian W2-squared "
                             "cost over the shared finite bank. Zero-based index; first index on exact ties.",
        "color_evaluation": "Root SW2 on clipped float RGB before 8-bit PNG quantization",
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
            start = time.perf_counter()
            bank = sample_projection_bank(3, max(config.projection_counts), seed=config.projection_seed + seed,
                                          device=device, dtype=dtype)
            synchronize(device)
            bank_ms = 1000 * (time.perf_counter() - start)
            np.savez_compressed(folder / "projection_bank.npz", theta=_numpy(bank.theta),
                                psi=_numpy(bank.psi), matrices=_numpy(bank.matrices))
            evaluator = ColorEvaluator.create(x, y, samples=config.eval_samples,
                                              projections=config.eval_projections, seed=config.eval_seed + seed)
            np.savez_compressed(folder / "evaluation_bank.npz", source_indices=evaluator.source_indices,
                                target_indices=evaluator.target_indices, directions=evaluator.directions)
            outputs, guided_outputs, timing_rows = [], [], {}
            reference_plan, reference_cost, reference_output = None, None, None
            jobs = [("MW2", 0, None)] + [
                (kind, count, aggregation) for count in sorted(set(config.projection_counts))
                for aggregation in config.aggregations for kind in ("Mix", "SMix")]
            for kind, count, aggregation in jobs:
                method = "MW2" if kind == "MW2" else f"{'min-' if aggregation == 'min' else ''}LSOT-{kind}"
                name = method if count == 0 else f"{method}_L{count}"
                if kind == "MW2":
                    def solver():
                        plan, cost = solve_mw2(source, target)
                        return plan, cost, None
                else:
                    selected_bank = bank.prefix(count)

                    def solver():
                        if aggregation == "min":
                            result = minimum_lsot(source, target, selected_bank, kind)
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
                    save_rgb(folder / f"{name}_guided.png", filtered)
                    guided_outputs.append((name, filtered))
                save_rgb(folder / f"{name}.png", mapped)
                if config.save_raw:
                    np.save(folder / f"{name}_raw.npy", mapped)
                _save_plan(folder / f"{name}_plan.npz", plan)
                if selection is not None:
                    np.savez_compressed(folder / f"{name}_selection.npz",
                                        projection_index=selection.projection_index,
                                        projection_costs=_numpy(selection.projection_costs))
                outputs.append((name, mapped))
                transport_mean, transport_std = time_summary(transport_times)
                setup_mean, setup_std = time_summary(setup_times)
                apply_mean, apply_std = time_summary(apply_times)
                row = {
                    "seed": seed, "method": method, "K0": source.count, "K1": target.count,
                    "d": 3, "L": count, "device": str(device), "dtype": config.dtype,
                    "aggregation": aggregation,
                    "selected_projection": selection.projection_index if selection is not None else None,
                    "solver_backend": "scipy+POT(cpu)" if kind == "MW2" else "torch",
                    "source_pixels": len(x), "target_pixels": len(y),
                    "fit_source_pixels": count_x, "fit_target_pixels": count_y,
                    "em_source_converged": bool(sx.converged_), "em_target_converged": bool(sy.converged_),
                    "fit_ms": fit_x + fit_y, "input_setup_ms": input_ms,
                    "bank_sampling_ms": bank_ms if kind != "MW2" else 0.0,
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
                    "guided_color_sw2": evaluator.sw2(filtered) if filtered is not None else None,
                    "clipped_channel_fraction": float(np.mean((mapped < 0) | (mapped > 1))),
                }
                all_rows.append(row)
                timing_rows[name] = {"transport_ms": transport_times, "map_setup_ms": setup_times,
                                     "map_apply_ms": apply_times}
                # Save after each method so a long run retains completed results.
                _write_rows(output_dir / "metrics.csv", all_rows)
                _write_json(folder / "timings.json", timing_rows)
                print(f"  {name}: cost={cost:.6g}, transport={transport_mean:.2f} ms, color SW2={row['color_sw2']:.6g}", flush=True)
            save_comparison(folder / "comparison.png", source_image, target_image, outputs)
            if guided_outputs:
                save_comparison(folder / "comparison_guided.png", source_image, target_image, guided_outputs)
    _write_rows(output_dir / "paper_results.tsv", _summarize(all_rows), delimiter="\t")
    print(f"Saved results to {output_dir.resolve()}", flush=True)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "mw2_reference.json")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--components", type=int, nargs="+", help="K, or K0 K1")
    parser.add_argument("--projections", type=int, nargs="+")
    parser.add_argument("--aggregations", choices=["avg", "min"], nargs="+",
                        help="LSOT aggregation(s); defaults to both avg and min")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--fit-pixels", type=int)
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("source", "target", "output_dir", "device", "seeds", "max_side", "fit_pixels", "aggregations"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.components:
        if len(args.components) not in {1, 2}:
            parser.error("--components expects K or K0 K1")
        values["components_source"] = args.components[0]
        values["components_target"] = args.components[-1]
    if args.projections:
        values["projection_counts"] = args.projections
    run_experiment(Config(**values))


if __name__ == "__main__":
    main()
