"""Focused integration checks for the tiny original examples."""
import csv

import numpy as np
import pytest

from .run import _grid_w2_figures, examples, main
from .presentation import DEFAULT_TIMES, time_tag, validate_times


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
    main(["--output-dir", str(tmp_path), "--L", "8",
          "--grid-1d", "30", "--grid-2d", "12",
          "--skip-grid-w2"])
    with (tmp_path / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18  # 8 avg/min variants + MW2, in both 1D and 2D
    for case, (source, target) in examples().items():
        subset = {row["method"]: row for row in rows if row["case"] == case}
        assert len(subset) == 9
        assert not any("min-opt" in name for name in subset)
        assert (tmp_path / f"{case}_couplings.png").exists()
        assert (tmp_path / f"{case}_interpolations.png").exists()
        mw2 = float(subset["MW2"]["cost_squared"])
        for kind in ("Mix", "SMix", "B", "B1D"):
            minimum = float(subset[f"min-LSOT-{kind}_L8"]["cost_squared"])
            average = float(subset[f"avg-LSOT-{kind}_L8"]["cost_squared"])
            assert mw2 <= minimum + 1e-10
            assert minimum <= average + 1e-10
        for name in subset:
            saved = np.load(tmp_path / "plans" / f"{case}_{name}.npz")
            matrix = np.zeros(tuple(saved["shape"]))
            np.add.at(matrix, (saved["rows"], saved["cols"]), saved["mass"])
            np.testing.assert_allclose(matrix.sum(axis=1), source.weights.numpy(), atol=1e-9)
            np.testing.assert_allclose(matrix.sum(axis=0), target.weights.numpy(), atol=1e-9)
            folder = tmp_path / "methods" / name / case
            sequence = np.load(folder / "sequence.npz")
            np.testing.assert_allclose(sequence["times"], DEFAULT_TIMES)
            assert sequence["probabilities"].shape == (6, len(sequence["points"]))
            np.testing.assert_allclose(sequence["probabilities"].sum(axis=1), 1)
            assert not np.allclose(sequence["probabilities"][2], sequence["probabilities"][3])
            for t in DEFAULT_TIMES:
                assert (folder / f"t_{time_tag(t)}.png").is_file()
            assert (folder / "interpolation_strip.png").is_file()
            assert (folder / "interpolation_strip.pdf").is_file()
            assert not (folder / "w2_vs_method.png").exists()  # grid comparison was skipped
            # All admissible component plans reproduce exactly the same endpoints.
            reference_sequence = np.load(tmp_path / "methods" / "MW2" / case / "sequence.npz")
            np.testing.assert_array_equal(sequence["probabilities"][[0, -1]],
                                          reference_sequence["probabilities"][[0, -1]])


@pytest.mark.parametrize("times", [[], [0.4, 0.2], [0, 0], [-0.1, 1], [0, 1.1], [float("nan")]])
def test_invalid_time_lists(times):
    with pytest.raises(ValueError, match="times"):
        validate_times(times)


def test_grid_barycenters_at_each_requested_time(tmp_path):
    from .run import _mixture_density
    for case, (source, target) in examples().items():
        x = np.linspace(0, 1, 20 if case == "1d" else 12)
        if case == "1d":
            points = x[:, None]
        else:
            xx, yy = np.meshgrid(x, x)
            points = np.column_stack((xx.ravel(), yy.ravel()))
        sequence = _grid_w2_figures(case, source, target, x, points, tmp_path, DEFAULT_TIMES)
        assert sequence.shape == (6, len(points))
        np.testing.assert_allclose(sequence.sum(axis=1), 1)
        a, b = _mixture_density(source, points), _mixture_density(target, points)
        np.testing.assert_array_equal(sequence[0], a / a.sum())
        np.testing.assert_array_equal(sequence[-1], b / b.sum())
        assert not np.allclose(sequence[2], sequence[3])
