"""Sweep K on fixed WikiArt pairs, with avg/min x four families plus MW2.

python -m experiments.color_transfer.wikiart_sweep --device cpu
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import stdev

from .wikiart import METRICS, _number, _write_csv, _write_json, run_pairs, summarize_pairs
from .wikiart_data import load_manifest, select_pairs


DEFAULT_COMPONENT_COUNTS = (5, 10, 20, 50, 100, 200)
DEFAULT_PAIR_INDICES = (2, 5, 8, 13, 14, 16)


def summarize_sweep(rows):
    """Keep K groups separate; compare each run to its same-K/seed MW2.

    Reuse the paired WikiArt summaries independently at each K. Descriptive
    run SD is not a confidence interval, especially with projection seeds.
    """
    by_k = {}
    for row in rows:
        k = int(row["K0"])
        if k != int(row["K1"]):
            raise ValueError("WikiArt K sweep requires K0=K1")
        by_k.setdefault(k, []).append(row)
    pairs, methods = [], []
    for k, group in sorted(by_k.items()):
        pair_summary, method_summary = summarize_pairs(group)
        runs = {}
        for row in group:
            key = int(row["pair_index"]), row["method"], int(row["L"])
            runs.setdefault(key, []).append(row)
        for entry in pair_summary:
            entry["K"] = k
            selected = runs[int(entry["pair_index"]), entry["method"], int(entry["L"])]
            entry["n_data_seeds"] = len({int(row["seed"]) for row in selected})
            for metric in METRICS:
                values = [_number(row.get(metric)) for row in selected]
                values = [value for value in values if value is not None]
                entry[metric + "_std"] = stdev(values) if len(values) > 1 else None
            pairs.append(entry)
        methods.extend({**entry, "K": k} for entry in method_summary)
    return pairs, methods


def _save_results(folder, rows):
    pairs, methods = summarize_sweep(rows)
    _write_csv(folder / "metrics.csv", rows)
    _write_csv(folder / "pair_k_summary.csv", pairs)
    _write_csv(folder / "method_k_summary.csv", methods)


def run_wikiart_sweep(config, output_dir, image_dir, *,
                      component_counts=DEFAULT_COMPONENT_COUNTS,
                      pair_indices=DEFAULT_PAIR_INDICES, download=True,
                      resume=True, make_plots=True, manifest_dir=None):
    """Refit shared GMMs per K, resume at pair/K granularity, never mix K.

    All K inherit the same preprocessing, fit-pixel sampling, data seeds,
    projection seeds/budgets and evaluation pixels/directions. Each fitted
    pair of GMMs is shared by all methods. Completed K/pair signatures are
    checked by run_pairs, not inferred merely from the presence of a CSV.
    """
    counts = list(component_counts)
    if (not counts or any(type(k) is not int or k < 1 for k in counts)
            or len(set(counts)) != len(counts)):
        raise ValueError("component_counts must contain distinct positive integers")
    config.validate()
    if config.fit_pixels is not None and config.fit_pixels < max(counts):
        raise ValueError("fit_pixels must be at least the largest K")
    pairs, _ = load_manifest(manifest_dir) if manifest_dir else load_manifest()
    indices = tuple(pair_indices)
    selected = select_pairs(pairs, indices)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "component_counts": counts, "pair_indices": list(indices),
        "pair_names": [pair.name for pair in selected],
        "base_config": asdict(config), "completed_component_counts": [],
        "design": "K0=K1=K. Refit both full-covariance RGB GMMs at every K; "
                  "share each fit across methods. Same image preprocessing, "
                  "fit-pixel subsamples, projection banks and evaluation banks across K. "
                  "MW2 runs once per pair/K/data seed, independent of projection seeds.",
        "interpretation": "color_sw2 is root empirical SW2 between clipped float "
                          "transferred RGB pixels and target pixels, before PNG quantization. "
                          "Lower is better. Increasing K need not lower this pixel metric. "
                          "Guided metrics are separate. Seed SD is descriptive, not a CI.",
    }
    _write_json(folder / "sweep_config.json", manifest)
    all_rows = []
    for k in counts:
        print(f"\nWikiArt K sweep: K={k}, pairs={list(indices)}", flush=True)
        setting = replace(config, components_source=k, components_target=k)
        # Never mix several K in summarize_pairs: its references are pair/seed.
        rows = run_pairs(setting, folder / f"K_{k}", image_dir,
                         pair_indices=indices, download=download, resume=resume,
                         manifest_dir=manifest_dir)
        all_rows.extend(rows)
        _save_results(folder, all_rows)
        manifest["completed_component_counts"].append(k)
        _write_json(folder / "sweep_config.json", manifest)
    if make_plots:
        from .plot_wikiart_sweep import plot_sweep
        with (folder / "pair_k_summary.csv").open(newline="", encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream))
        plot_sweep(summary, folder / "plots", metric="color_sw2", expected_counts=counts)
        if config.guided_filter:
            plot_sweep(summary, folder / "plots", metric="guided_color_sw2", expected_counts=counts)
    print(f"K-sweep results saved to {folder.resolve()}", flush=True)
    return all_rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configs" / "wikiart_pairs.json")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/color_transfer/wikiart_k_sweep"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/color_transfer/wikiart"))
    parser.add_argument("--component-counts", type=int, nargs="+", default=list(DEFAULT_COMPONENT_COUNTS))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pair-indices", type=int, nargs="+")
    group.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--aggregations", nargs="+", choices=("avg", "min", "min-opt"),
                        default=["avg", "min"], help="Default: 8 LSOT methods, no min-opt")
    parser.add_argument("--projections", type=int, default=100, help="Fixed solver L, not evaluation directions")
    parser.add_argument("--seeds", type=int, nargs="+", help="Optional data/GMM seeds override")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main():
    from .run import Config
    args = build_parser().parse_args()
    config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    overrides = {"aggregations": args.aggregations, "projection_counts": [args.projections]}
    if args.device is not None:
        overrides["device"] = args.device
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    config = replace(config, **overrides)
    pairs, _ = load_manifest()
    indices = (list(range(len(pairs))) if args.all_pairs else
               args.pair_indices if args.pair_indices is not None else DEFAULT_PAIR_INDICES)
    run_wikiart_sweep(config, args.output_dir, args.image_dir,
                      component_counts=args.component_counts, pair_indices=indices,
                      download=not args.no_download, resume=not args.no_resume,
                      make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
