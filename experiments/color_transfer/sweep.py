"""Sweep K0=K1 and compare every method across Gaussian component counts.

python -m experiments.color_transfer.sweep --component-counts 10 20 50 100 200

Other settings are inherited from --config, including projection budgets,
families, avg/min, image preprocessing, timing repetitions and data seeds.
"""
from dataclasses import asdict, replace
from pathlib import Path

from .run import (
    build_parser, config_from_arguments, run_experiment,
    _summarize, _projection_seed_summary, _write_json, _write_rows,
)

DEFAULT_COMPONENT_COUNTS = (10, 20, 50, 100, 200)
TABLE_METRICS = {
    "runtime_ms": "transport_ms_mean",
    "cost_squared": "cost_squared",
    "color_sw2": "color_sw2",
    "guided_color_sw2": "guided_color_sw2",
    "plan_rmse": "plan_rmse",
}


def comparison_tables(summary):
    """Numeric pivot tables: rows K, columns method/L, values seed means.

    MW2 appears once per K, not once for each projection budget. Each LSOT
    column includes its L to avoid mixing budgets or aggregations. Standard
    deviations and seed counts remain in paper_results.tsv.
    """
    columns = []
    counts = sorted({row["K0"] for row in summary})
    lookup = {}
    for row in summary:
        if row["K0"] != row["K1"]:
            raise ValueError("component sweep tables require K0=K1")
        label = row["method"] if row["L"] == 0 else f"{row['method']}_L{row['L']}"
        if label not in columns:
            columns.append(label)
        key = row["K0"], label
        if key in lookup:
            raise ValueError(f"duplicate summary entry for {key}")
        lookup[key] = row
    tables = {}
    for filename, metric in TABLE_METRICS.items():
        key = metric + "_across_seeds_mean"
        tables[filename] = [
            {"K": k, **{label: lookup.get((k, label), {}).get(key) for label in columns}}
            for k in counts
        ]
    return tables


def _save_sweep_results(folder, rows):
    summary = _summarize(rows)
    _write_rows(folder / "metrics.csv", rows)
    _write_rows(folder / "paper_results.tsv", summary, delimiter="\t")
    _write_rows(folder / "projection_seed_results.tsv", _projection_seed_summary(rows), delimiter="\t")
    compact = [{
        "K": row["K0"], "method": row["method"], "L": row["L"],
        "runtime_ms": row["transport_ms_mean_across_seeds_mean"],
        "cost_squared": row["cost_squared_across_seeds_mean"],
        "color_sw2": row["color_sw2_across_seeds_mean"],
        "plan_rmse": row["plan_rmse_across_seeds_mean"],
    } for row in summary]
    _write_rows(folder / "comparison.tsv", compact, delimiter="\t")
    tables = comparison_tables(summary)
    for name, table in tables.items():
        _write_rows(folder / f"{name}.tsv", table, delimiter="\t")
    return tables


def _print_table(title, rows):
    keys = list(rows[0])
    strings = [keys]
    for row in rows:
        strings.append(["--" if row[key] is None else
                        (str(row[key]) if key == "K" else f"{row[key]:.6g}") for key in keys])
    widths = [max(len(row[i]) for row in strings) for i in range(len(keys))]
    print(f"\n{title} (mean across data/projection-seed runs; MW2 once per data seed)", flush=True)
    for row in strings:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)), flush=True)


def run_component_sweep(config, component_counts=DEFAULT_COMPONENT_COUNTS):
    """Refit shared GMMs at each K; keep projection/evaluation banks fixed.

    Results for each completed K are immediately written into the root
    summary tables. Detailed images, plans, GMMs and banks live in K_<K>/.
    This is a serial sweep so methods do not compete for GPU resources.
    """
    counts = list(component_counts)
    if (not counts or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in counts)
            or len(set(counts)) != len(counts)):
        raise ValueError("component_counts must contain distinct positive integers")
    config.validate()
    folder = Path(config.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "component_counts": counts, "base_config": asdict(config),
        "completed_component_counts": [],
        "setting": "K0=K1=K. Same images, preprocessing, seeds and maximum projection "
                   "budgets across K; GMMs are fitted once per image/seed/K and shared "
                   "by every method. Projection and evaluation banks do not depend on K.",
        "tables": "Rows K, columns method/L, LSOT means across data/projection-seed runs. "
                  "MW2 is evaluated once per data seed, not repeated for projection seeds. Runtime is solver "
                  "time in ms, including true Gaussian cost evaluation. It excludes EM "
                  "and pixel mapping. Descriptive pooled standard deviations and run counts are "
                  "in paper_results.tsv; projection_seed_results.tsv separates projection seeds.",
    }
    _write_json(folder / "sweep_config.json", manifest)
    rows = []
    for k in counts:
        print(f"\nComponent sweep: K0=K1={k}", flush=True)
        setting = replace(config, components_source=k, components_target=k,
                          output_dir=str(folder / f"K_{k}"))
        rows.extend(run_experiment(setting))
        tables = _save_sweep_results(folder, rows)
        manifest["completed_component_counts"].append(k)
        _write_json(folder / "sweep_config.json", manifest)
    for metric in ("runtime_ms", "cost_squared", "color_sw2"):
        _print_table(metric, tables[metric])
    print(f"\nSweep tables saved to {folder.resolve()}", flush=True)
    return rows


def main():
    parser = build_parser(description=__doc__)
    parser.set_defaults(output_dir="results/color_transfer/k_sweep")
    parser.add_argument("--component-counts", type=int, nargs="+",
                        default=list(DEFAULT_COMPONENT_COUNTS), help="K values; K0=K1 at every setting")
    args = parser.parse_args()
    if args.components is not None:
        parser.error("use --component-counts for sweeps; --components is for single runs")
    run_component_sweep(config_from_arguments(args), args.component_counts)


if __name__ == "__main__":
    main()
