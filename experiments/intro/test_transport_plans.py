"""Check the atom-level law, including exact cell marginals and plan reuse."""
import json

import numpy as np
import pytest
import torch

from lsot import GMM, SparsePlan
from .plot_transport_plans import (
    METHODS, component_plans, grid_cells, main, mixture_histogram, rasterize_lifted_plan,
)
from .run import examples


def test_identity_lift_has_only_diagonal_mass():
    source, _ = examples()["1d"]
    plan = SparsePlan.from_dense(torch.diag(source.weights))
    _, _, edges = grid_cells(100)
    a = mixture_histogram(source, edges)
    matrix = rasterize_lifted_plan(source, source, plan, edges)
    np.testing.assert_allclose(matrix, np.diag(a), atol=1e-14)


def test_affine_gaussian_lift_integrates_cell_intersections():
    source = GMM.from_numpy([1.], [[0.]], [[[1.]]])
    target = GMM.from_numpy([1.], [[1.]], [[[4.]]])
    plan = SparsePlan.from_dense(torch.ones((1, 1), dtype=torch.float64))
    # T(x)=2*x+1. Source x<=0 implies target y<=1, a visibly non-diagonal joint law.
    edges = np.array([-np.inf, 0., 1., np.inf])
    from scipy.special import ndtr
    expected = np.zeros((3, 3))
    expected[0, 0] = ndtr(-.5)
    expected[0, 1] = .5-ndtr(-.5)
    expected[1, 2] = ndtr(1.)-.5
    expected[2, 2] = 1.-ndtr(1.)
    matrix = rasterize_lifted_plan(source, target, plan, edges)
    np.testing.assert_allclose(matrix, expected, atol=1e-14)
    np.testing.assert_allclose(matrix.sum(0), mixture_histogram(target, edges), atol=1e-14)


def test_mixture_lift_preserves_both_marginals_and_multiple_branches():
    source, target = examples()["1d"]
    _, _, edges = grid_cells(100)
    plans = component_plans(source, target, count=8, seed=0)
    a, b = mixture_histogram(source, edges), mixture_histogram(target, edges)
    assert tuple(plans) == METHODS[1:]
    for plan in plans.values():
        matrix = rasterize_lifted_plan(source, target, plan, edges)
        np.testing.assert_allclose(matrix.sum(1), a, atol=1e-13)
        np.testing.assert_allclose(matrix.sum(0), b, atol=1e-13)
        assert np.min(matrix) >= 0
    # A single source cell sends mass to widely separated targets: not Tmean.
    matrix = rasterize_lifted_plan(source, target, plans["MW2"], edges)
    occupied = np.flatnonzero(matrix[39] > 1e-6)
    assert occupied[-1]-occupied[0] > 10
    np.testing.assert_allclose(plans["MW2"].dense().numpy(), [[.3, 0], [.3, .4]], atol=1e-12)


def test_rejects_invalid_dimensions_and_component_marginals():
    source, target = examples()["1d"]
    _, _, edges = grid_cells(20)
    wrong = SparsePlan.from_dense(torch.eye(2, dtype=torch.float64)/2)
    with pytest.raises(ValueError, match="marginals"):
        rasterize_lifted_plan(source, target, wrong, edges)
    source, target = examples()["2d"]
    with pytest.raises(ValueError, match="one-dimensional"):
        rasterize_lifted_plan(source, target, wrong, edges)


def test_saved_intro_plans_are_reused_and_config_mismatches_rejected(tmp_path):
    source, target = examples()["1d"]
    plans = component_plans(source, target, count=8, seed=0)
    folder = tmp_path / "plans"
    folder.mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"L": 8, "seed": 0}))
    for name, plan in plans.items():
        tag = name if name == "MW2" else name+"_L8"
        np.savez(folder / f"1d_{tag}.npz", rows=plan.rows.numpy(), cols=plan.cols.numpy(),
                 mass=plan.mass.numpy(), shape=plan.shape)
    reused = component_plans(source, target, count=8, seed=0, plans_dir=folder)
    for name in plans:
        np.testing.assert_array_equal(reused[name].dense(), plans[name].dense())
    with pytest.raises(ValueError, match="differs"):
        component_plans(source, target, count=100, seed=0, plans_dir=folder)


def test_full_export_contains_all_ten_methods_with_valid_marginals(tmp_path):
    rows = main(["--output-dir", str(tmp_path), "--L", "8", "--grid-size", "24", "--dpi", "40"])
    assert [r["method"] for r in rows] == list(METHODS)
    assert not any("min-opt" in r["method"] for r in rows)
    assert max(r["source_marginal_l1"] for r in rows) < 1e-12
    assert max(r["target_marginal_l1"] for r in rows) < 1e-12
    baseline_cost = rows[0]["grid_cost_squared"]
    assert all(r["grid_cost_squared"] >= baseline_cost-1e-12 for r in rows)
    with np.load(tmp_path / "transport_plans.npz") as data:
        assert data["matrices"].shape == (10, 24, 24)
        assert data["component_matrices"].shape == (9, 2, 2)
    for suffix in ("png", "pdf"):
        assert (tmp_path / f"transport_plans_all.{suffix}").is_file()
        assert (tmp_path / f"transport_plans_w2_mw2.{suffix}").is_file()
        for name in METHODS:
            assert (tmp_path / "methods" / name / f"transport_plan.{suffix}").is_file()
