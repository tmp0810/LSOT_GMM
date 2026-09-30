"""Check sweep aggregation, bank fairness and saved comparison tables."""
import csv
import json

import numpy as np
import pytest
from PIL import Image

from experiments.color_transfer.run import Config
from experiments.color_transfer.sweep import run_component_sweep, comparison_tables


def test_offline_component_sweep_tables_and_fixed_banks(tmp_path):
    rng = np.random.default_rng(65)
    for name in ("source", "target"):
        Image.fromarray(rng.integers(0, 256, (12, 13, 3), dtype=np.uint8)).save(tmp_path / f"{name}.png")
    folder = tmp_path / "sweep"
    config = Config(source=str(tmp_path / "source.png"), target=str(tmp_path / "target.png"),
                    output_dir=str(folder), download_reference=False, device="cpu",
                    projection_counts=[2], seeds=[0, 1], repeats=1, warmups=0,
                    map_repeats=1, guided_filter=False, eval_samples=32, eval_projections=3)
    rows = run_component_sweep(config, [2, 3])
    assert len(rows) == 2 * 2 * 9
    assert {r["K0"] for r in rows} == {2, 3}
    assert config.components_source == 10  # the caller's config is not mutated
    with (folder / "paper_results.tsv").open() as stream:
        summary = list(csv.DictReader(stream, delimiter="\t"))
    assert len(summary) == 18 and all(int(r["n_seeds"]) == 2 for r in summary)
    with (folder / "comparison.tsv").open() as stream:
        compact = list(csv.DictReader(stream, delimiter="\t"))
    assert len(compact) == 18
    assert list(compact[0]) == ["K", "method", "L", "runtime_ms", "cost_squared", "color_sw2", "plan_rmse"]
    np.testing.assert_allclose(float(compact[0]["cost_squared"]),
                               float(summary[0]["cost_squared_across_seeds_mean"]))
    for filename, metric in (("runtime_ms", "transport_ms_mean"), ("cost_squared", "cost_squared"),
                             ("color_sw2", "color_sw2"), ("plan_rmse", "plan_rmse")):
        with (folder / f"{filename}.tsv").open() as stream:
            table = list(csv.DictReader(stream, delimiter="\t"))
        assert [int(r["K"]) for r in table] == [2, 3]
        assert len(table[0]) == 10  # K plus nine distinct methods
        for entry in table:
            k = int(entry["K"])
            for row in summary:
                if int(row["K0"]) != k:
                    continue
                name = row["method"] if int(row["L"]) == 0 else f"{row['method']}_L{row['L']}"
                expected = np.mean([r[metric] for r in rows if r["K0"] == k and r["method"] == row["method"]])
                np.testing.assert_allclose(float(entry[name]), expected, rtol=1e-14)
    for seed in config.seeds:
        for filename in ("projection_bank.npz", "projection_bank_B.npz",
                         "projection_bank_B1D.npz", "evaluation_bank.npz"):
            a = np.load(folder / "K_2" / f"seed_{seed}" / filename)
            b = np.load(folder / "K_3" / f"seed_{seed}" / filename)
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    manifest = json.loads((folder / "sweep_config.json").read_text())
    assert manifest["completed_component_counts"] == [2, 3]


def test_pivot_keeps_projection_budgets_separate():
    summary = [dict(K0=k, K1=k, method=method, L=l, cost_squared_across_seeds_mean=value)
               for k in [10, 20] for method, l, value in
               [("MW2", 0, 1.0), ("min-LSOT-B", 5, 2.0), ("min-LSOT-B", 20, 1.5)]]
    table = comparison_tables(summary)["cost_squared"]
    assert list(table[0]) == ["K", "MW2", "min-LSOT-B_L5", "min-LSOT-B_L20"]
    assert table[0]["min-LSOT-B_L20"] == 1.5


@pytest.mark.parametrize("counts", [[], [0], [10, 10], [1.5]])
def test_invalid_component_sweep(counts):
    with pytest.raises(ValueError, match="component_counts"):
        run_component_sweep(Config(), counts)
