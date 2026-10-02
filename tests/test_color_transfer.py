"""Offline end-to-end checks, including different source/target image sizes."""
import json
import numpy as np
import pytest
import torch
from PIL import Image

from experiments.color_transfer.data import read_rgb, guided_output
from experiments.color_transfer.run import Config, run_experiment
from distribution_proj import BusemannBank, Busemann1DBank
from param_proj import ProjectionBank, sample_projection_bank
from lsot import GMM, average_lsot, minimum_lsot


def test_png_normalization_and_zero_displacement_filter(tmp_path):
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(12, 9, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(image).save(path)
    loaded = read_rgb(path)
    np.testing.assert_array_equal(loaded, image / 255.0)
    np.testing.assert_allclose(guided_output(loaded, loaded, radius=2), loaded)


def test_offline_pipeline_and_reproducible_saved_inputs(tmp_path):
    rng = np.random.default_rng(4)
    source, target = tmp_path / "source.png", tmp_path / "target.png"
    Image.fromarray(rng.integers(0, 256, size=(18, 15, 3), dtype=np.uint8)).save(source)
    Image.fromarray(rng.integers(0, 256, size=(13, 17, 3), dtype=np.uint8)).save(target)
    config = Config(source=str(source), target=str(target), download_reference=False,
                    output_dir=str(tmp_path / "out"), components_source=3, components_target=2,
                    projection_counts=[3], seeds=[2], device="cpu", repeats=1, warmups=0,
                    map_repeats=1, guided_filter=True, guided_radius=2,
                    eval_samples=64, eval_projections=7, batch_size=31,
                    opt_steps=2, opt_samples=2)
    rows = run_experiment(config)
    assert len(rows) == 13
    assert {r["projection_family"] for r in rows} == {None, "Mix", "SMix", "B", "B1D"}
    for row in rows:
        assert row["marginal_l1_error"] < 1e-12
        assert row["relative_cost_gap"] >= -1e-10
        assert row["source_pixels"] == 270 and row["target_pixels"] == 221
        assert np.isfinite(row["color_sw2"]) and np.isfinite(row["guided_color_sw2"])
    assert rows[0]["plan_rmse"] == 0
    folder = tmp_path / "out"
    assert (folder / "paper_results.tsv").is_file()
    assert (folder / "seed_2" / "comparison.png").is_file()
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["device"] == "cpu"
    gmms = np.load(folder / "seed_2" / "gmms.npz")
    weights = gmms["alpha"]
    source_gmm = GMM.from_numpy(weights, gmms["means_source"], gmms["covariances_source"])
    target_gmm = GMM.from_numpy(gmms["beta"], gmms["means_target"], gmms["covariances_target"])
    # Enabling B/B1D must not perturb the previous Mix/SMix random bank.
    previous_bank = sample_projection_bank(3, 3, seed=config.projection_seed + 2)
    saved_bank = np.load(folder / "seed_2" / "projection_bank.npz")
    for key, value in vars(previous_bank).items():
        np.testing.assert_array_equal(saved_bank[key], value.numpy())
    for row in rows:
        name = row["method"] if row["L"] == 0 else f"{row['method']}_L{row['L']}"
        plan = np.load(folder / "seed_2" / f"{name}_plan.npz")
        np.testing.assert_allclose(np.bincount(plan["rows"], weights=plan["mass"], minlength=3), weights, atol=1e-12)
        if row["projection_family"] is not None:
            # Reconstruct every plan from saved bank tensors, without resampling.
            saved_bank = np.load(folder / "seed_2" / row["projection_bank"])
            bank_type = {"Mix": ProjectionBank, "SMix": ProjectionBank,
                         "B": BusemannBank, "B1D": Busemann1DBank}[row["projection_family"]]
            bank = bank_type(**{key: torch.tensor(saved_bank[key]) for key in saved_bank.files})
            if row["aggregation"] == "min":
                replay = minimum_lsot(source_gmm, target_gmm, bank, row["projection_family"]).plan
            elif row["aggregation"] == "avg":
                replay = average_lsot(source_gmm, target_gmm, bank, row["projection_family"])
            else:
                saved = np.load(folder / "seed_2" / f"{name}_selection.npz")
                best = bank_type(**{key: torch.tensor(saved[f"best_{key}"])
                                    for key in vars(bank)})
                replay = minimum_lsot(source_gmm, target_gmm, best, row["projection_family"]).plan
                assert row["initial_projection"] == int(saved["initial_projection_index"])
                assert row["best_iteration"] == int(saved["best_iteration"])
                assert row["cost_squared"] <= float(saved["initial_cost_squared"]) + 1e-12
            np.testing.assert_array_equal(replay.rows.numpy(), plan["rows"])
            np.testing.assert_array_equal(replay.cols.numpy(), plan["cols"])
            np.testing.assert_allclose(replay.mass.numpy(), plan["mass"], atol=1e-15)
        if row["aggregation"] == "min":
            selection = np.load(folder / "seed_2" / f"{name}_selection.npz")
            selected = int(selection["projection_index"])
            assert selected == row["selected_projection"] == np.argmin(selection["projection_costs"])
            assert len(selection["projection_costs"]) == row["L"]
            np.testing.assert_allclose(selection["projection_costs"][selected], row["cost_squared"])
            average = next(r for r in rows if r["method"] == row["method"].removeprefix("min-"))
            np.testing.assert_allclose(selection["projection_costs"].mean(), average["cost_squared"])
        elif row["aggregation"] != "min":
            assert row["selected_projection"] is None
        with Image.open(folder / "seed_2" / f"{name}.png") as image:
            assert image.size == (15, 18)


@pytest.mark.parametrize("kinds", [[], ["B", "B"], ["unknown"]])
def test_config_rejects_invalid_projection_families(kinds):
    with pytest.raises(ValueError, match="projection_kinds"):
        Config(projection_kinds=kinds).validate()
