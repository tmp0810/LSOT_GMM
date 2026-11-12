"""Numerical parity and protocol tests for the fast barycenter experiment."""
import numpy as np
import pytest
import torch

from lsot.gaussians import GMM
from lsot.projections import PROJECTION_KINDS, sample_projection_bank, project_gaussians
from .data import synthetic_example, gaussian_example
from .solver import BarycenterObjective, ParameterizedGMM, pair_cost_matrix
from .reference import screened_initialization, mw2_barycenter, shared_initialization
from .fast_geometry import pair_costs, project_components
from .fast_solver import (FastParameterizedGMM, FastObjective, MW2Objective,
                          optimize_fast, checkpoint_gmm, subset_bank, join_banks)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def random_gmm(k, seed, d=2):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, d, d))
    return GMM.from_numpy(rng.uniform(.2, 1, k), rng.normal(size=(k, d)),
                          .2*a @ a.transpose(0, 2, 1)+.1*np.eye(d))


@pytest.mark.parametrize("dimension", [2, 3])
def test_cost_value_and_gradient_match_spectral(dimension):
    source, initial = random_gmm(3, 3, dimension), random_gmm(4, 4, dimension)
    first, second = ParameterizedGMM(initial), FastParameterizedGMM(initial)
    old, new = pair_cost_matrix(source, first()), pair_costs(source, second())
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    old.sum().backward()
    new.sum().backward()
    for name in ["means", "raw_cholesky"]:
        torch.testing.assert_close(getattr(first, name).grad, getattr(second, name).grad,
                                   atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("kind", PROJECTION_KINDS)
@pytest.mark.parametrize("mode", ["min", "avg"])
@pytest.mark.parametrize("dense_threshold", [0, 4096])
def test_fast_objective_value_and_all_parameter_gradients(kind, mode, dense_threshold):
    inputs, initial = [random_gmm(3, 3), random_gmm(4, 5)], random_gmm(5, 8)
    bank = sample_projection_bank(2, 17, kind=kind, seed=13)
    models = [ParameterizedGMM(initial), FastParameterizedGMM(initial)]
    values = [BarycenterObjective(inputs, [.37, .63], bank, kind, mode)(models[0]()),
              FastObjective(inputs, [.37, .63], bank, kind, mode,
                            dense_threshold=dense_threshold)(models[1]())]
    torch.testing.assert_close(values[0], values[1], atol=1e-12, rtol=1e-12)
    for value in values:
        value.backward()
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("mode", ["min", "avg"])
def test_proportional_fiber_ties_and_minimum_ties(mode):
    source = random_gmm(2, 12)
    initial = GMM.from_numpy([.2, .35, .45], [[0., 0.]]*3, [np.eye(2)]*3)
    bank = sample_projection_bank(2, 1, kind="B1D", seed=8)
    bank = join_banks(bank, bank)
    models = [ParameterizedGMM(initial), FastParameterizedGMM(initial)]
    values = [cls([source], [1.], bank, "B1D", mode)(model())
              for cls, model in zip([BarycenterObjective, FastObjective], models)]
    torch.testing.assert_close(*values, atol=1e-12, rtol=1e-12)
    for value in values:
        value.backward()
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)


def test_algebraic_b_projection_and_reference_covariance_iterations():
    g = random_gmm(5, 1)
    bank = sample_projection_bank(2, 11, kind="B", seed=4)
    torch.testing.assert_close(project_components(g, bank, "B"),
                               project_gaussians(g.means, g.covariances, bank, "B"),
                               atol=1e-11, rtol=1e-11)
    inputs = synthetic_example().inputs
    old, new = [mw2_barycenter(inputs, [.13, .27, .23, .37], fast=fast) for fast in [False, True]]
    assert old.cost_squared == pytest.approx(new.cost_squared, abs=1e-12)
    np.testing.assert_allclose(old.ground_costs, new.ground_costs, atol=1e-12)


def test_mw2_dual_gradient_reaches_learned_weights():
    source, initial = random_gmm(3, 4), random_gmm(4, 5)
    model = FastParameterizedGMM(initial)
    objective = MW2Objective([source], [1.])
    objective(model()).backward()
    analytic = model.logits.grad.clone()
    assert analytic.abs().max() > 1e-5
    for k in range(initial.count):
        h = 1e-6
        with torch.no_grad():
            model.logits[k] += h
            high = float(objective(model()))
            model.logits[k] -= 2*h
            low = float(objective(model()))
            model.logits[k] += h
        assert analytic[k] == pytest.approx((high-low)/(2*h), abs=2e-7)


def test_screening_is_shared_reproducible_and_one_start():
    inputs = synthetic_example().inputs
    a, info = screened_initialization(inputs, [.25]*4, 11, seed=7)
    b, other = screened_initialization(inputs, [.25]*4, 11, seed=7)
    assert info == other and info["optimizer_starts"] == 1
    assert info["selected"] == int(np.argmin(info["proxy_costs"]))
    assert a.count == 11
    torch.testing.assert_close(a.means, b.means)
    torch.testing.assert_close(a.weights, b.weights)


@pytest.mark.parametrize("sampling", ["fixed", "fresh", "partial"])
def test_selection_uses_fixed_validation_and_is_reproducible(sampling):
    inputs = synthetic_example().inputs
    initial, _ = screened_initialization(inputs, [.25]*4, 11)
    kwargs = dict(kind="B1D", steps=5, L=7, validation_L=13, validate_every=1,
                  sampling=sampling, refresh_every=2, checkpoints=(2, 4), patience=0, seed=33)
    result = optimize_fast(inputs, [.25]*4, initial, **kwargs)
    repeat = optimize_fast(inputs, [.25]*4, initial, **kwargs)
    bank = sample_projection_bank(2, 13, kind="B1D", seed=33+1000003)
    expected = float(FastObjective(inputs, [.25]*4, bank, "B1D", "min")(result.gmm))
    assert result.objective == pytest.approx(expected, abs=1e-12)
    assert result.objective <= result.initial_objective+1e-12
    assert result.objective == pytest.approx(min(row["objective_squared"] for row in result.history))
    torch.testing.assert_close(result.gmm.means, repeat.gmm.means)
    if sampling != "fixed":
        assert len({row["bank_epoch"] for row in result.history}) > 1
    for checkpoint in result.checkpoints:
        candidate = checkpoint_gmm(initial, checkpoint)
        value = float(FastObjective(inputs, [.25]*4, bank, "B1D", "min")(candidate))
        assert value == pytest.approx(checkpoint["validation_objective"], abs=1e-12)


def test_early_stopping_is_a_plateau_not_a_convergence_certificate():
    inputs = gaussian_example().inputs
    initial = shared_initialization(inputs, [1/3]*3, 1)
    result = optimize_fast(inputs, [1/3]*3, initial, kind="B1D", steps=20,
                           learning_rate=1e-12, L=3, validation_L=5,
                           validate_every=1, patience=2, relative_tolerance=.1)
    assert result.status == "validation_plateau"
    assert result.iterations == 2
    assert not result.gradient_tolerance_met


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", PROJECTION_KINDS)
def test_cuda_objective_gradient_and_short_run(kind):
    from .solver import on_device
    inputs = [on_device(random_gmm(3, 3), "cuda")]
    initial = on_device(random_gmm(4, 7), "cuda")
    result = optimize_fast(inputs, [1.], initial, kind=kind, steps=2, L=5, validation_L=7,
                           sampling="fresh", validate_every=1)
    assert result.gmm.means.is_cuda and np.isfinite(result.objective)


def test_standalone_benchmark_checks_plan_marginals_and_cost_order():
    from .benchmark import plan_once
    source, target = random_gmm(4, 2), random_gmm(5, 8)
    exact, _, _ = plan_once(source, target, "MW2", 11, 0)
    costs = pair_costs(source, target)
    for mode in ("avg", "min"):
        for kind in PROJECTION_KINDS:
            value, left, right = plan_once(source, target, f"{mode}-LSOT-{kind}", 11, 0)
            cached, _, _ = plan_once(source, target, f"{mode}-LSOT-{kind}", 11, 0, costs)
            assert value >= exact-1e-12
            assert value == pytest.approx(cached, abs=1e-12)
            torch.testing.assert_close(left, source.weights, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(right, target.weights, atol=1e-12, rtol=1e-12)
