"""Run reproducible color transfer on the WikiArt pairs from Sliced-Amortized-OT.

python -m experiments.color_transfer.wikiart --output-dir results/color_transfer/wikiart
"""

import argparse
import csv
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean, median

from .wikiart_data import (DEFAULT_PAIR_INDICES, load_manifest, prepare_images,
                           save_inventory, select_pairs)


METRICS = ("color_sw2", "color_w2", "guided_color_sw2", "guided_color_w2")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    return None if value is None or value == "" else float(value)


def summarize_pairs(rows):
    """Pair LSOT and MW2 on the same images and GMM seed; weight pairs equally."""
    references = {}
    for row in rows:
        if row["method"] == "MW2":
            key = row["pair_index"], row["seed"]
            if key in references:
                raise ValueError(f"Duplicate MW2 reference for {key}")
            references[key] = row

    groups = {}
    for row in rows:
        key = row["pair_index"], row["seed"]
        if key not in references:
            raise ValueError(f"Missing same-seed MW2 reference for {key}")
        reference = references[key]
        paired = dict(row)
        for metric in METRICS:
            value, baseline = _number(row.get(metric)), _number(reference.get(metric))
            paired[f"delta_{metric}_to_mw2"] = (
                value - baseline if value is not None and baseline is not None else None)
        identity = _number(row.get("identity_color_w2"))
        value = _number(row.get("color_w2"))
        paired["delta_color_w2_to_identity"] = (
            value - identity if value is not None and identity is not None else None)
        group_key = row["pair_index"], row["method"], int(row["L"])
        groups.setdefault(group_key, []).append(paired)

    pair_summary = []
    measures = ("cost_squared", "relative_cost_gap", "transport_ms_mean",
                "identity_color_sw2", "identity_color_w2", *METRICS,
                *(f"delta_{name}_to_mw2" for name in METRICS),
                "delta_color_w2_to_identity")
    for (_, _, _), group in groups.items():
        first = group[0]
        summary = {key: first[key] for key in
                   ("pair_index", "pair_name", "source_slug", "target_slug",
                    "method", "L", "K0", "K1")}
        summary["n_runs"] = len(group)
        for name in measures:
            values = [_number(item.get(name)) for item in group]
            values = [item for item in values if item is not None]
            summary[name] = fmean(values) if values else None
        pair_summary.append(summary)

    across = {}
    for row in pair_summary:
        across.setdefault((row["method"], int(row["L"])), []).append(row)
    method_summary = []
    for (_, _), group in across.items():
        first = group[0]
        summary = {"method": first["method"], "L": first["L"],
                   "K0": first["K0"], "K1": first["K1"], "n_pairs": len(group)}
        for name in measures:
            values = [item[name] for item in group if item[name] is not None]
            summary[f"mean_{name}"] = fmean(values) if values else None
            if name.startswith("delta_"):
                summary[f"median_{name}"] = median(values) if values else None
                summary[f"wins_{name}"] = sum(v < -1e-12 for v in values)
        method_summary.append(summary)
    return pair_summary, method_summary


def _fingerprint(config, source_hash, target_hash):
    content = {"config": asdict(config), "source_sha256": source_hash,
               "target_sha256": target_hash}
    return sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()


def run_pairs(config, output_dir, image_dir, *, pair_indices=DEFAULT_PAIR_INDICES,
              download=True, prepare_only=False, force_download=False, resume=True,
              manifest_dir=None, opener=None):
    """Prepare each unique image once and retain results after every pair.

    `opener` and `manifest_dir` allow an entirely offline preparation check.
    """
    pairs, urls = load_manifest(manifest_dir) if manifest_dir else load_manifest()
    selected = select_pairs(pairs, tuple(pair_indices))
    if not prepare_only:
        config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {"download": download, "force": force_download}
    if opener is not None:
        options["opener"] = opener
    inventory = prepare_images(selected, urls, image_dir, **options)
    save_inventory(output_dir / "dataset.json", selected, inventory)
    if prepare_only:
        print(f"Prepared {len(inventory)} images for {len(selected)} pairs in {image_dir}", flush=True)
        return []

    from .run import run_experiment

    all_rows = []
    for pair in selected:
        pair_dir = output_dir / pair.name
        pair_dir.mkdir(parents=True, exist_ok=True)
        setting = replace(config, source=inventory[pair.source]["path"],
                          target=inventory[pair.target]["path"],
                          output_dir=str(pair_dir), download_reference=False)
        signature = _fingerprint(setting, inventory[pair.source]["sha256"],
                                 inventory[pair.target]["sha256"])
        completed = pair_dir / "run_complete.json"
        metrics = pair_dir / "metrics.csv"
        saved = json.loads(completed.read_text(encoding="utf-8")) if completed.is_file() else {}
        pair_rows = None
        if resume and saved.get("fingerprint") == signature and metrics.is_file():
            with metrics.open(newline="", encoding="utf-8") as stream:
                pair_rows = list(csv.DictReader(stream))
            if len(pair_rows) != saved.get("rows"):
                print(f"Recomputing incomplete saved pair {pair.name}", flush=True)
                pair_rows = None
            else:
                print(f"Resuming completed {pair.name} ({len(pair_rows)} rows)", flush=True)
        if pair_rows is None:
            print(f"Running {pair.name}", flush=True)
            pair_rows = run_experiment(setting)
            # Keep the completion marker and on-disk CSV in agreement even
            # after a previous interrupted or otherwise partial CSV write.
            _write_csv(metrics, pair_rows)
            _write_json(completed, {"fingerprint": signature, "rows": len(pair_rows)})
        all_rows.extend({"pair_index": pair.index, "pair_name": pair.name,
                         "source_slug": pair.source, "target_slug": pair.target,
                         **row} for row in pair_rows)
        _write_csv(output_dir / "metrics.csv", all_rows)
        pair_summary, method_summary = summarize_pairs(all_rows)
        _write_csv(output_dir / "pair_summary.csv", pair_summary)
        _write_csv(output_dir / "method_summary.csv", method_summary)
    print(f"Saved {len(selected)} pairs to {output_dir.resolve()}", flush=True)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configs" / "wikiart_pairs.json")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/color_transfer/wikiart"))
    parser.add_argument("--image-dir", type=Path,
                        default=Path("data/color_transfer/wikiart"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pair-indices", nargs="+", type=int)
    group.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-download", action="store_true",
                        help="Use local slug.jpg/.png files, for offline or manual data setup")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.force_download and args.no_download:
        parser.error("--force-download and --no-download cannot be used together")
    pairs, _ = load_manifest()
    indices = (range(len(pairs)) if args.all_pairs else
               args.pair_indices if args.pair_indices is not None else DEFAULT_PAIR_INDICES)
    if args.prepare_only:
        config = None
    else:
        from .run import Config
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    run_pairs(config, args.output_dir, args.image_dir, pair_indices=indices,
              download=not args.no_download, prepare_only=args.prepare_only,
              force_download=args.force_download, resume=not args.no_resume)


if __name__ == "__main__":
    main()
