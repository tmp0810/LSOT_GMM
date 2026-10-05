"""Focused integration checks for the tiny original examples."""
import csv

import numpy as np

from .run import examples, main


def test_example_parameters_match_notebook():
    cases = examples()
    source, target = cases["1d"]
    np.testing.assert_allclose(source.weights.numpy(), [0.3, 0.7])
    np.testing.assert_allclose(target.means.numpy().ravel(), [0.6, 0.8])
    np.testing.assert_allclose(source.covariances.numpy().ravel(), [0.0009, 0.0016])
    source, target = cases["2d"]
    np.testing.assert_allclose(target.weights.numpy(), [0.45, 0.55])
    np.testing.assert_allclose(source.covariances.numpy(), 0.01 * np.tile(np.eye(2), (2, 1, 1)))


def test_all_families_modes_and_mw2(tmp_path):
    main(["--output-dir", str(tmp_path), "--L", "8", "--opt-steps", "2",
          "--opt-samples", "2", "--grid-1d", "30", "--grid-2d", "12",
          "--skip-grid-w2"])
    with (tmp_path / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 26
    for case, (source, target) in examples().items():
        subset = {row["method"]: row for row in rows if row["case"] == case}
        assert len(subset) == 13
        assert (tmp_path / f"{case}_couplings.png").exists()
        assert (tmp_path / f"{case}_interpolations.png").exists()
        mw2 = float(subset["MW2"]["cost_squared"])
        for kind in ("Mix", "SMix", "B", "B1D"):
            minimum = float(subset[f"min-LSOT-{kind}_L8"]["cost_squared"])
            optimized = float(subset[f"min-opt-LSOT-{kind}_L8"]["cost_squared"])
            assert mw2 <= optimized + 1e-10
            assert optimized <= minimum + 1e-10
        for name in subset:
            saved = np.load(tmp_path / "plans" / f"{case}_{name}.npz")
            matrix = np.zeros(tuple(saved["shape"]))
            np.add.at(matrix, (saved["rows"], saved["cols"]), saved["mass"])
            np.testing.assert_allclose(matrix.sum(axis=1), source.weights.numpy(), atol=1e-9)
            np.testing.assert_allclose(matrix.sum(axis=0), target.weights.numpy(), atol=1e-9)
