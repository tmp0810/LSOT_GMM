"""Plot per-pair and per-method SW2 versus K from pair_k_summary.csv.

python -m experiments.color_transfer.plot_wikiart_sweep \
    --input results/color_transfer/wikiart_k_sweep/pair_k_summary.csv
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from .wikiart import METRICS, _number, _write_csv


def sweep_curves(summary, metric="color_sw2"):
    """Group without averaging across K/pairs; reject ambiguous duplicates."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    curves = {}
    seen = set()
    for row in summary:
        pair, k, length = int(row["pair_index"]), int(row["K"]), int(row["L"])
        key = pair, row["method"], length
        if (*key, k) in seen:
            raise ValueError(f"duplicate pair/method/L/K summary: {(*key, k)}")
        seen.add((*key, k))
        value = _number(row.get(metric))
        if value is None:
            continue
        sd = _number(row.get(metric + "_std"))
        if not math.isfinite(value) or value < 0 or (sd is not None and (not math.isfinite(sd) or sd < 0)):
            raise ValueError("distance and SD must be finite and nonnegative")
        curves.setdefault(key, []).append({"K": k, "value": value, "std": sd,
                                            "pair_name": row["pair_name"],
                                            "n_runs": int(row["n_runs"])})
    if not curves:
        raise ValueError(f"no measured values for {metric}")
    return {key: sorted(values, key=lambda row: row["K"]) for key, values in curves.items()}


def trend_rows(curves, metric):
    """Describe endpoint and adjacent changes; do not assert monotonic gains."""
    output = []
    for (pair, method, length), values in sorted(curves.items()):
        first, last = values[0], values[-1]
        differences = [b["value"] - a["value"] for a, b in zip(values, values[1:])]
        change = last["value"] - first["value"] if differences else None
        output.append({
            "pair_index": pair, "pair_name": first["pair_name"], "method": method, "L": length,
            "metric": metric, "K_first": first["K"], "K_last": last["K"],
            "n_K_observed": len(values), "first_value": first["value"], "last_value": last["value"],
            "last_minus_first": change,
            "endpoint_reduction_pct": (-100 * change / first["value"]
                                       if change is not None and first["value"] > 0 else None),
            "n_adjacent_decreases": sum(delta < -1e-12 for delta in differences),
            "n_adjacent_increases": sum(delta > 1e-12 for delta in differences),
            "nonincreasing_on_observed_K": (all(delta <= 1e-12 for delta in differences)
                                           if differences else None),
            "best_observed_K": min(values, key=lambda row: row["value"])["K"],
        })
    return output


def plot_sweep(summary, output_dir, *, metric="color_sw2", expected_counts=None):
    """Plot fixed-K ticks, separate raw/guided scores and optional seed SD.

    Missing expected K break a line rather than implying measurements were
    made there. Per-pair plots compare methods; per-method plots compare pairs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = sweep_curves(summary, metric)
    observed = sorted({row["K"] for values in curves.values() for row in values})
    ticks = sorted(set(expected_counts) | set(observed)) if expected_counts is not None else observed
    if not ticks or any(type(k) is not int or k <= 0 for k in ticks):
        raise ValueError("K ticks must be positive integers")
    folder = Path(output_dir) / metric
    folder.mkdir(parents=True, exist_ok=True)
    methods = sorted({(method, length) for _, method, length in curves},
                     key=lambda key: (key[0] != "MW2", key[0], key[1]))
    pairs = sorted({pair for pair, _, _ in curves})
    colors = {key: ("black" if key[0] == "MW2" else plt.get_cmap("tab10")(index % 10))
              for index, key in enumerate(methods)}
    paths = []

    def decorate(axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks, labels=[str(k) for k in ticks])
        axis.minorticks_off()
        axis.set_xlabel("K (GMM components per image)")
        axis.set_ylabel("RGB " + ("SW2" if "sw2" in metric else "W2") + " (lower is better)")
        axis.grid(alpha=0.25)

    def draw(axis, values, *, label, color, mw2=False, style="-"):
        lookup = {row["K"]: row for row in values}
        y = np.array([lookup[k]["value"] if k in lookup else np.nan for k in ticks])
        sd = np.array([(lookup[k]["std"] or 0) if k in lookup else np.nan for k in ticks])
        axis.plot(ticks, y, marker="o", linestyle=style, label=label,
                  color=color, linewidth=2.4 if mw2 else 1.5, markersize=4)
        if np.any(sd > 0):
            axis.fill_between(ticks, np.maximum(0, y - sd), y + sd, color=color, alpha=0.1)

    def pair_panel(axis, pair, detailed=False):
        name = None
        for method, length in methods:
            values = curves.get((pair, method, length))
            if not values:
                continue
            name = values[0]["pair_name"]
            label = method if method == "MW2" else f"{method} (L={length})"
            draw(axis, values, label=label, color=colors[method, length], mw2=method == "MW2",
                 style="--" if method.startswith("min-") else "-")
        decorate(axis)
        if detailed:
            axis.set_title(f"Pair {pair:02d} — {metric}\n" + name.split("_", 2)[-1].replace("__", " → "), fontsize=9)
        else:
            axis.set_title(f"Pair {pair:02d} — {metric}")

    for pair in pairs:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        pair_panel(axis, pair, detailed=True)
        axis.legend(fontsize=8, loc="best", ncol=2)
        figure.tight_layout()
        path = folder / f"pair_{pair:02d}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    columns = min(3, len(pairs))
    figure, axes = plt.subplots(math.ceil(len(pairs) / columns), columns,
                               figsize=(6 * columns, 4.5 * math.ceil(len(pairs) / columns)), squeeze=False)
    handles = {}
    for axis, pair in zip(axes.flat, pairs):
        pair_panel(axis, pair)
        h, labels = axis.get_legend_handles_labels()
        handles.update(zip(labels, h))
    for axis in list(axes.flat)[len(pairs):]:
        axis.axis("off")
    figure.legend(list(handles.values()), list(handles), loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    path = folder / "all_pairs.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    for method, length in methods:
        figure, axis = plt.subplots(figsize=(9, 5))
        for index, pair in enumerate(pairs):
            values = curves.get((pair, method, length))
            if values:
                draw(axis, values, label=f"Pair {pair:02d}", color=plt.get_cmap("tab20")(index % 20))
        decorate(axis)
        axis.set_title(f"{method}" + (f" (L={length})" if length else "") + f" — {metric}")
        axis.legend(fontsize=8, ncol=3)
        figure.tight_layout()
        path = folder / f"method_{method}_L{length}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    _write_csv(folder / "trends.csv", trend_rows(curves, metric))
    print(f"Saved {len(paths)} plots and trends.csv to {folder.resolve()}", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("results/color_transfer/wikiart_k_sweep/pair_k_summary.csv"))
    parser.add_argument("--output-dir", type=Path, help="Default: sibling plots/ directory")
    parser.add_argument("--metric", choices=METRICS, default="color_sw2")
    parser.add_argument("--component-counts", type=int, nargs="+", help="Expected K; missing values break lines")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = args.component_counts
    manifest = args.input.parent / "sweep_config.json"
    if expected is None and manifest.is_file():
        import json
        expected = json.loads(manifest.read_text(encoding="utf-8"))["component_counts"]
    plot_sweep(rows, args.output_dir or args.input.parent / "plots", metric=args.metric,
               expected_counts=expected)


if __name__ == "__main__":
    main()
