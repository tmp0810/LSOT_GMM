"""Check sweep aggregation, bank fairness and saved comparison tables."""
import csv
import json

import numpy as np
import pytest
from PIL import Image

from experiments.color_transfer.run import Config
from experiments.color_transfer.sweep import run_component_sweep, comparison_tables


def test_projection_seed_sweep_shares_fit_and_reference(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import experiments.color_transfer.run as run
    rng = np.random.default_rng(23)
    for name in ("source", "target"):
        Image.fromarray(rng.integers(0, 256, (10, 11, 3), dtype=np.uint8)).save(tmp_path / f"{name}.png")
    fit = Mock(wraps=run.fit_gmm)
    solve = Mock(wraps=run.solve_mw2)
    monkeypatch.setattr(run, "fit_gmm", fit)
    monkeypatch.setattr(run, "solve_mw2", solve)
    folder = tmp_path / "out"
    config = Config(source=str(tmp_path / "source.png"), target=str(tmp_path / "target.png"),
                    output_dir=str(folder), download_reference=False, device="cpu",
                    projection_seed=[42, 43], projection_counts=[2, 4], seeds=[1],
                    projection_kinds=["Mix", "SMix"], repeats=1, warmups=0,
                    map_repeats=1, guided_filter=False, eval_samples=32, eval_projections=3)
    rows = run_component_sweep(config, [2, 3])
    assert fit.call_count == 4  # two images x two K; NOT multiplied by projection seeds
    assert solve.call_count == 2
    assert len(rows) == 2 * (1 + 2 * 2 * 2 * 2)
    for k in [2, 3]:
        root = folder / f"K_{k}" / "seed_1"
        assert (root / "gmms.npz").is_file()
        assert (root / "evaluation_bank.npz").is_file()
        selected_rows = [r for r in rows if r["K0"] == k]
        assert sum(r["method"] == "MW2" for r in selected_rows) == 1
        for row in selected_rows:
            if row["method"] == "MW2":
                assert row["projection_seed"] is None
                continue
            ps = row["projection_seed"]
            assert row["effective_projection_seed"] == ps + 1
            assert (root / row["projection_bank"]).is_file()
            output = root / f"projection_seed_{ps}" / f"{row['method']}_L{row['L']}"
            assert output.with_suffix(".png").is_file()
            assert output.with_name(output.name + "_plan.npz").is_file()
            assert row["marginal_l1_error"] < 1e-12
        first = np.load(root / "projection_seed_42" / "projection_bank.npz")
        second = np.load(root / "projection_seed_43" / "projection_bank.npz")
        assert not np.array_equal(first["theta"], second["theta"])
    for ps in [42, 43]:
        a = np.load(folder / "K_2" / "seed_1" / f"projection_seed_{ps}" / "projection_bank.npz")
        b = np.load(folder / "K_3" / "seed_1" / f"projection_seed_{ps}" / "projection_bank.npz")
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    with (folder / "projection_seed_results.tsv").open() as stream:
        split = list(csv.DictReader(stream, delimiter="\t"))
    assert len(split) == len(rows)
    with (folder / "paper_results.tsv").open() as stream:
        summary = list(csv.DictReader(stream, delimiter="\t"))
    for row in summary:
        assert int(row["n_seeds"]) == 1
        assert int(row["n_runs"]) == (1 if row["method"] == "MW2" else 2)
        assert int(row["n_projection_seeds"]) == (0 if row["method"] == "MW2" else 2)


@pytest.mark.parametrize("value", [[], [1, 1], [True], [-1], [1.5], "42", True])
def test_invalid_projection_seeds(value):
    with pytest.raises(ValueError, match="projection_seed"):
        Config(projection_seed=value).validate()


def test_projection_seed_cli_and_singleton_list(tmp_path):
    from experiments.color_transfer.run import build_parser, config_from_arguments
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"projection_seed": 42}))
    parser = build_parser()
    scalar = config_from_arguments(parser.parse_args(["--config", str(path)]))
    assert scalar.projection_seed_values() == [42]
    one = config_from_arguments(parser.parse_args(["--config", str(path), "--projection-seeds", "7"]))
    assert one.projection_seed == [7]
    many = config_from_arguments(parser.parse_args(["--config", str(path), "--projection-seeds", "0", "1", "2"]))
    many.validate()
    assert many.projection_seed == [0, 1, 2]


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
