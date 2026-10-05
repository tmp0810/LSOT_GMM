"""K-isolated MW2 comparisons, fixed banks, resumption and plotting checks."""
import csv
import json
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from experiments.color_transfer.run import Config
from experiments.color_transfer.wikiart_data import load_manifest, select_pairs
from experiments.color_transfer.wikiart_sweep import (
    DEFAULT_COMPONENT_COUNTS, build_parser, run_wikiart_sweep, summarize_sweep,
)
from experiments.color_transfer.plot_wikiart_sweep import plot_sweep, sweep_curves, trend_rows


def sample_rows():
    rows = []
    for pair in [2, 5]:
        for k, baseline, result in [(5, 0.3, 0.4), (10, 0.1, 0.2), (20, 0.2, 0.25)]:
            for seed in [0, 1]:
                for method, value in [("MW2", baseline), ("min-LSOT-B", result)]:
                    rows.append({
                        "pair_index": pair, "pair_name": f"pair_{pair:02d}_source__target",
                        "source_slug": "source", "target_slug": "target",
                        "K0": k, "K1": k, "seed": seed, "method": method,
                        "L": 0 if method == "MW2" else 100,
                        "cost_squared": 0.1, "relative_cost_gap": 0, "transport_ms_mean": 1,
                        "color_sw2": value + 0.01 * seed, "color_w2": 2 * value,
                        "guided_color_sw2": None, "guided_color_w2": None,
                        "identity_color_sw2": 0.5, "identity_color_w2": 0.8,
                    })
    return rows


def test_summaries_do_not_mix_k_or_mw2_references():
    pairs, methods = summarize_sweep(sample_rows())
    assert len(pairs) == 12 and len(methods) == 6
    entry = next(row for row in pairs if row["pair_index"] == 2 and row["K"] == 10
                 and row["method"] == "min-LSOT-B")
    assert entry["n_runs"] == entry["n_data_seeds"] == 2
    assert entry["color_sw2"] == pytest.approx(0.205)
    assert entry["color_sw2_std"] == pytest.approx(0.01 / np.sqrt(2))
    assert entry["delta_color_sw2_to_mw2"] == pytest.approx(0.1)
    assert entry["guided_color_sw2"] is None
    assert {row["K"] for row in methods} == {5, 10, 20}


def test_curves_and_trends_do_not_confuse_endpoint_and_monotonic_improvement():
    pairs, _ = summarize_sweep(sample_rows())
    curves = sweep_curves(pairs)
    assert len(curves) == 4
    trend = next(row for row in trend_rows(curves, "color_sw2")
                 if row["pair_index"] == 2 and row["method"] == "min-LSOT-B")
    assert trend["last_minus_first"] == pytest.approx(-0.15)
    assert trend["endpoint_reduction_pct"] > 0
    assert trend["n_adjacent_decreases"] == trend["n_adjacent_increases"] == 1
    assert not trend["nonincreasing_on_observed_K"]
    assert trend["best_observed_K"] == 10
    one = trend_rows(sweep_curves([pairs[0]]), "color_sw2")[0]
    assert one["last_minus_first"] is None
    assert one["nonincreasing_on_observed_K"] is None
    with pytest.raises(ValueError, match="duplicate"):
        sweep_curves([pairs[0], pairs[0]])
    with pytest.raises(ValueError, match="no measured"):
        sweep_curves(pairs, "guided_color_sw2")


def test_plot_outputs_and_sparse_k(tmp_path):
    pairs, _ = summarize_sweep(sample_rows())
    paths = plot_sweep(pairs, tmp_path, expected_counts=[5, 10, 20, 50])
    assert len(paths) == 5  # 2 pair figures, 1 grid, 2 method figures
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)
    with (tmp_path / "color_sw2/trends.csv").open() as stream:
        trends = list(csv.DictReader(stream))
    assert len(trends) == 4
    assert all(row["K_last"] == "20" for row in trends)


def test_defaults_exactly_avg_min_four_families():
    args = build_parser().parse_args([])
    assert args.aggregations == ["avg", "min"]
    assert args.component_counts == [5, 10, 20, 50, 100, 200]
    assert args.projections == 100
    assert len(Config().projection_kinds) * len(args.aggregations) + 1 == 9


@pytest.mark.parametrize("counts", [[], [0], [True], [5, 5], [5.5]])
def test_invalid_counts_before_io(tmp_path, counts):
    with pytest.raises(ValueError, match="component_counts"):
        run_wikiart_sweep(Config(), tmp_path / "out", tmp_path / "images", component_counts=counts)


def test_fit_budget_checked_before_download(tmp_path):
    with pytest.raises(ValueError, match="fit_pixels"):
        run_wikiart_sweep(Config(fit_pixels=10), tmp_path / "out", tmp_path / "images",
                         component_counts=DEFAULT_COMPONENT_COUNTS)


def test_offline_nine_methods_sweep_fixed_banks_and_resume(tmp_path, monkeypatch):
    import experiments.color_transfer.run as run
    pairs, _ = load_manifest()
    chosen = select_pairs(pairs, [2, 5])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(81)
    for slug in {slug for pair in chosen for slug in (pair.source, pair.target)}:
        Image.fromarray(rng.integers(0, 256, (12, 13, 3), dtype=np.uint8)).save(image_dir / f"{slug}.png")
    config = Config(download_reference=False, aggregations=["avg", "min"],
                    projection_counts=[2], device="cpu", seeds=[0], repeats=1, warmups=0,
                    map_repeats=1, guided_filter=False, eval_samples=32,
                    eval_projections=3, eval_w2_samples=16)
    fits = Mock(wraps=run.fit_gmm)
    monkeypatch.setattr(run, "fit_gmm", fits)
    folder = tmp_path / "out"
    kwargs = dict(component_counts=[2, 3], pair_indices=[2, 5], download=False, make_plots=False)
    rows = run_wikiart_sweep(config, folder, image_dir, **kwargs)
    assert len(rows) == 36 and fits.call_count == 8  # 2 K x 2 pairs x 2 images
    assert len({row["method"] for row in rows}) == 9
    assert not any("min-opt" in row["method"] for row in rows)
    # Cached CSV values are strings. Aggregation must accept resumed/mixed runs.
    resumed = run_wikiart_sweep(config, folder, image_dir, **kwargs)
    assert len(resumed) == 36 and fits.call_count == 8
    with (folder / "pair_k_summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 36
    assert {row["K"] for row in summary} == {"2", "3"}
    assert json.loads((folder / "sweep_config.json").read_text())["completed_component_counts"] == [2, 3]
    for pair in chosen:
        for name in ["evaluation_bank.npz", "projection_bank.npz", "projection_bank_B.npz", "projection_bank_B1D.npz"]:
            a = np.load(folder / "K_2" / pair.name / "seed_0" / name)
            b = np.load(folder / "K_3" / pair.name / "seed_0" / name)
            assert a.files == b.files
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field])
    # Incomplete pair/K is recomputed, without rerunning the other 3 settings.
    partial = folder / "K_2" / chosen[0].name / "metrics.csv"
    partial.write_text(partial.read_text().splitlines()[0] + "\n")
    run_wikiart_sweep(config, folder, image_dir, **kwargs)
    assert fits.call_count == 10
