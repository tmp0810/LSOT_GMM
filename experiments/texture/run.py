"""Run the one-scale MW2 / min-LSOT / avg-LSOT texture benchmark."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from lsot.projections import sample_projection_bank
from .data import adsn, extract_patches, load_rgb, save_rgb, texture_path
from .evaluation import TextureMetrics
from .synthesis import METHODS, fit_gmm, synthesize


DEFAULT = Path(__file__).with_name("configs") / "paper.json"


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--images", nargs="+", help="StoRI filenames or local paths")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--component-counts", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--L", type=int)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--fit-patches", type=int, help="Limit per GMM; 0 uses all patches")
    parser.add_argument("--timing-repeats", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-download", action="store_false", dest="download", default=None)
    parser.add_argument("--no-plots", action="store_false", dest="plots", default=None)
    args = parser.parse_args(argv)
    try:
        defaults = json.loads(DEFAULT.read_text())
        supplied = json.loads(args.config.read_text())
        if not isinstance(supplied, dict) or supplied.keys() - defaults.keys():
            raise ValueError("Config must be an object with only the supported keys")
        cfg = defaults | supplied
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                cfg[key] = str(value) if isinstance(value, Path) else value
        for key, typ in (("images", str), ("methods", str), ("component_counts", int), ("seeds", int)):
            values = cfg[key]
            if (not isinstance(values, list) or not values or any(type(v) is not typ for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{key} must be a nonempty list of distinct {typ.__name__} values")
        if any(not Path(name).stem for name in cfg["images"]):
            raise ValueError("Image paths must have a nonempty filename")
        if len({Path(name).stem.casefold() for name in cfg["images"]}) != len(cfg["images"]):
            raise ValueError("Use images with distinct filename stems to avoid output collisions")
        if set(cfg["methods"]) - set(METHODS):
            raise ValueError("Unknown method")
        if any(k < 1 for k in cfg["component_counts"]) or any(not 0 <= s < 2**32 for s in cfg["seeds"]):
            raise ValueError("K must be positive and seeds must be 32-bit unsigned integers")
        for key in ("L", "patch_size", "em_iterations", "sw_samples", "w_samples",
                    "eval_directions", "batch_size", "timing_repeats", "threads"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        side = cfg["max_side"]
        if side is not None and (type(side) is not int or side < cfg["patch_size"]):
            raise ValueError("max_side must be null or at least patch_size")
        if type(cfg["fit_patches"]) is not int or cfg["fit_patches"] < 0:
            raise ValueError("fit_patches must be nonnegative (0 means all patches)")
        if cfg["fit_patches"] and cfg["fit_patches"] < max(cfg["component_counts"]):
            raise ValueError("fit_patches must be at least the largest K")
        for key in ("em_tolerance", "em_regularization"):
            if type(cfg[key]) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("download", "plots"):
            if type(cfg[key]) is not bool:
                raise ValueError(f"{key} must be a boolean")
        for key in ("image_dir", "output_dir"):
            if not isinstance(cfg[key], str) or not cfg[key]:
                raise ValueError(f"{key} must be a nonempty path")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return cfg


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sample_patches(patches, limit, seed, stream):
    if not limit or len(patches) <= limit:
        return patches
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    return patches[rng.choice(len(patches), size=limit, replace=False)]


def fit_shared(source, target, k, seed, cfg):
    models, metadata = [], []
    for i, patches in enumerate((source, target)):
        state = int(np.random.SeedSequence([seed, 10+i]).generate_state(1)[0])
        start = perf_counter()
        model, info = fit_gmm(patches, k, state, iterations=cfg["em_iterations"],
                              tolerance=cfg["em_tolerance"], regularization=cfg["em_regularization"])
        info["fit_ms"] = 1000*(perf_counter()-start)
        info["random_state"] = state
        models.append(model)
        metadata.append(info)
    return models, metadata


def save_mixtures(path, source, target):
    np.savez_compressed(path, **{
        f"{side}_{attr}": getattr(model, attr).numpy()
        for side, model in (("source", source), ("target", target))
        for attr in ("weights", "means", "covariances")
    })


def summarize(root, records, timings, adsn_records):
    frame = pd.DataFrame(records)
    frame.to_csv(root/"metrics.csv", index=False)
    pd.DataFrame(timings).to_csv(root/"timings.csv", index=False)
    pd.DataFrame(adsn_records).to_csv(root/"adsn_metrics.csv", index=False)
    fields = ["patch_sw2", "patch_w2", "autocov_rmse", "patch_sw2_delta_vs_adsn",
              "patch_w2_delta_vs_adsn", "autocov_rmse_delta_vs_adsn", "plan_cost_squared", "plan_nnz",
              "prepare_ms", "fit_ms", "bank_ms", "plan_ms", "map_setup_ms", "map_apply_ms",
              "aggregate_ms", "method_ms", "total_ms", "plan_speedup_vs_mw2",
              "method_speedup_vs_mw2", "total_speedup_vs_mw2"]
    groups = frame.groupby(["K", "method"], sort=False)
    summary = groups[fields].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary["runs"] = groups.size()
    summary["em_converged_runs"] = groups["em_converged"].sum()
    summary.reset_index(inplace=True)
    summary.to_csv(root/"summary.csv", index=False)
    return summary


def check_plan_costs(rows):
    """Every plan is feasible; exact MW2 and a common min/avg bank give bounds."""
    costs = {name: row["plan_cost_squared"] for name, row in rows.items()}
    if any(not np.isfinite(v) for v in costs.values()):
        raise ValueError("Nonfinite component plan cost")
    if "MW2" in costs and any(v + 1e-8 < costs["MW2"] for v in costs.values()):
        raise ValueError("A component plan scored below the MW2 optimum")
    for method, cost in costs.items():
        average = method.replace("min-", "avg-", 1)
        if method.startswith("min-") and average in costs and costs[average] + 1e-8 < cost:
            raise ValueError(f"{average} scored below {method} on their shared bank")


def run_texture(root, path, origin, cfg, records, timing_rows, adsn_rows):
    start = perf_counter()
    exemplar, original_shape = load_rgb(path, cfg["max_side"])
    image_load_ms = 1000*(perf_counter()-start)
    shape = exemplar.shape[:2]
    name = Path(path).stem
    patch_size = cfg["patch_size"]
    for seed in cfg["seeds"]:
        out = root/name/f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        tick = perf_counter()
        initial = adsn(exemplar, seed)
        adsn_ms = 1000*(perf_counter()-tick)
        tick = perf_counter()
        source_patches = extract_patches(initial, patch_size)
        target_patches = extract_patches(exemplar, patch_size)
        source_fit = sample_patches(source_patches, cfg["fit_patches"], seed, 1)
        target_fit = sample_patches(target_patches, cfg["fit_patches"], seed, 2)
        points = torch.from_numpy(source_patches)
        patch_ms = 1000*(perf_counter()-tick)
        if min(len(source_fit), len(target_fit)) < max(2, max(cfg["component_counts"])):
            raise ValueError(f"{name} has too few fitting patches for the requested K")
        prepare_ms = image_load_ms+adsn_ms+patch_ms
        eval_image = TextureMetrics(exemplar, target_patches, patch_size, seed,
                                    sw_samples=cfg["sw_samples"], w_samples=cfg["w_samples"],
                                    directions=cfg["eval_directions"])
        adsn_image = np.clip(initial, 0., 1.)
        adsn_metrics = eval_image(adsn_image)
        adsn_rows.append(dict(image=name, seed=seed, adsn_ms=adsn_ms, **adsn_metrics))
        save_rgb(out/"exemplar.png", exemplar)
        save_rgb(out/"adsn.png", adsn_image)
        write_json(out/"input.json", dict(source=origin, seed=seed, original_shape=original_shape,
                   image_shape=shape, patch_size=patch_size, patch_dimension=points.shape[1],
                   source_patch_count=len(source_patches), target_patch_count=len(target_patches),
                   source_fit_patches=len(source_fit), target_fit_patches=len(target_fit),
                   sw_samples=eval_image.n_sw, w_samples=eval_image.n_w))
        banks = {}
        for method in cfg["methods"]:
            if method == "MW2":
                continue
            kind = method.rsplit("-", 1)[1]
            if kind not in banks:
                tick = perf_counter()
                bank = sample_projection_bank(points.shape[1], cfg["L"], kind=kind, seed=seed)
                banks[kind] = (bank, 1000*(perf_counter()-tick))

        for k in cfg["component_counts"]:
            folder = out/f"K{k}"
            folder.mkdir(exist_ok=True)
            (source, target), fit = fit_shared(source_fit, target_fit, k, seed, cfg)
            save_mixtures(folder/"mixtures.npz", source, target)
            write_json(folder/"fit.json", dict(source=fit[0], target=fit[1]))
            converged = fit[0]["converged"] and fit[1]["converged"]
            if not converged:
                print(f"{name} seed={seed} K={k}: EM did not converge; see fit.json", flush=True)
            fit_ms = sum(f["fit_ms"] for f in fit)
            outputs, rows = {}, {}
            times = {m: [] for m in cfg["methods"]}
            # A single untimed warmup per method on this problem. Cold EM and
            # data preparation are measured once because every method shares them.
            for m in cfg["methods"]:
                bank, _ = banks.get(m.rsplit("-", 1)[-1], (None, 0.))
                synthesize(source, target, points, shape, patch_size, m, bank, cfg["batch_size"])
            results = {}
            for repeat in range(cfg["timing_repeats"]):
                shift = repeat % len(cfg["methods"])
                for m in cfg["methods"][shift:]+cfg["methods"][:shift]:
                    bank, bank_ms = banks.get(m.rsplit("-", 1)[-1], (None, 0.))
                    result = synthesize(source, target, points, shape, patch_size, m,
                                        bank, cfg["batch_size"])
                    results[m] = result
                    times[m].append(result.timings)
                    timing_rows.append(dict(image=name, seed=seed, K=k, method=m, repeat=repeat,
                                            prepare_ms=prepare_ms, fit_ms=fit_ms, bank_ms=bank_ms,
                                            total_ms=prepare_ms+fit_ms+bank_ms+result.timings["method_ms"],
                                            **result.timings))
            for m in cfg["methods"]:
                result = results[m]
                _, bank_ms = banks.get(m.rsplit("-", 1)[-1], (None, 0.))
                error = result.plan.marginal_error(source, target)
                if not np.isfinite(error) or error > 1e-8:
                    raise ValueError(f"{name} seed={seed} K={k} {m}: invalid plan marginals ({error})")
                plan_cost = result.plan_cost_squared
                if plan_cost is None:
                    # avg-LSOT can return its plan without evaluating costs.
                    # This diagnostic is deliberately outside the plan timer.
                    plan_cost = float(result.plan.cost(source, target))
                median = {phase: float(np.median([t[phase] for t in times[m]]))
                          for phase in times[m][0]}
                quality = eval_image(result.image)
                row = dict(image=name, seed=seed, K=k, method=m,
                           L=None if m == "MW2" else cfg["L"], patch_size=patch_size,
                           patch_dimension=points.shape[1], patches=len(points),
                           source_fit_patches=len(source_fit), target_fit_patches=len(target_fit),
                           em_converged=converged, image_load_ms=image_load_ms, adsn_ms=adsn_ms,
                           patch_ms=patch_ms, prepare_ms=prepare_ms, source_fit_ms=fit[0]["fit_ms"],
                           target_fit_ms=fit[1]["fit_ms"], fit_ms=fit_ms, bank_ms=bank_ms,
                           total_ms=prepare_ms+fit_ms+bank_ms+median["method_ms"],
                           plan_nnz=result.plan.nnz, plan_marginal_l1=error,
                           plan_cost_squared=plan_cost, projection_index=result.projection_index,
                           **median, **quality,
                           **{f"{key}_delta_vs_adsn": value-adsn_metrics[key]
                              for key, value in quality.items()})
                rows[m] = row
                outputs[m] = result.image
                save_rgb(folder/f"{m}.png", result.image)
                np.savez_compressed(folder/f"{m}_plan.npz", rows=result.plan.rows.numpy(),
                                    cols=result.plan.cols.numpy(), mass=result.plan.mass.numpy(),
                                    shape=result.plan.shape)
            check_plan_costs(rows)
            for m, row in rows.items():
                for phase in ("plan", "method", "total"):
                    key = f"{phase}_ms"
                    row[f"{phase}_speedup_vs_mw2"] = rows["MW2"][key]/row[key] if "MW2" in rows else np.nan
                print(f"{name} seed={seed} K={k:3} {m:15}  SW2={row['patch_sw2']:.5f}"
                      f"  W2={row['patch_w2']:.5f}  plan={row['plan_ms']:.2f} ms"
                      f"  total={row['total_ms']:.1f} ms", flush=True)
            records.extend(rows.values())
            summarize(root, records, timing_rows, adsn_rows)
            if cfg["plots"]:
                from .plotting import save_comparison
                save_comparison(folder, exemplar, adsn_image, outputs, rows)


def main(argv=None):
    cfg = parse_config(argv)
    torch.set_num_threads(cfg["threads"])
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"settings.json", cfg)
    records, times, adsn_records = [], [], []
    with threadpool_limits(cfg["threads"]):
        for name in cfg["images"]:
            path, origin = texture_path(name, cfg["image_dir"], download=cfg["download"])
            run_texture(root, path, origin, cfg, records, times, adsn_records)
    if cfg["plots"]:
        from .plotting import save_curves
        save_curves(root, pd.read_csv(root/"summary.csv"), adsn_records)


if __name__ == "__main__":
    main()
