"""Offline end-to-end checks, including different source/target image sizes."""
import json
import numpy as np
from PIL import Image

from experiments.color_transfer.data import read_rgb, guided_output
from experiments.color_transfer.run import Config, run_experiment


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
                    eval_samples=64, eval_projections=7, batch_size=31)
    rows = run_experiment(config)
    assert len(rows) == 3
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
    weights = np.load(folder / "seed_2" / "gmms.npz")["alpha"]
    for name in ("MW2", "LSOT-Mix_L3", "LSOT-SMix_L3"):
        plan = np.load(folder / "seed_2" / f"{name}_plan.npz")
        np.testing.assert_allclose(np.bincount(plan["rows"], weights=plan["mass"], minlength=3), weights, atol=1e-12)
        with Image.open(folder / "seed_2" / f"{name}.png") as image:
            assert image.size == (15, 18)
