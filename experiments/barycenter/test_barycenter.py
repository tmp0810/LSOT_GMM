"""Numerical checks against upstream MW2 and existing lifted LSOT solvers."""
import numpy as np
import pytest
import torch

from gmmot import create_cost_matrix_from_gmm, solveMMOT
from lsot.plans import average_lsot, minimum_lsot
from lsot.projections import PROJECTION_KINDS, sample_projection_bank
from .data import synthetic_example, gaussian_example, image_cloud, bilinear_weights, grid_nodes
from .reference import (mw2_barycenter, shared_initialization, shared_initializations,
                        gaussian_barycenters)
from .solver import BarycenterObjective, ParameterizedGMM, optimize_barycenter


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_mw2_matches_original_notebook_functions():
    inputs = synthetic_example().inputs
    weights = np.array([.13, .22, .31, .34])
    original = [[g.count, g.weights.numpy(), g.means.numpy(), g.covariances.numpy()] for g in inputs]
    cost, means, covariances = create_cost_matrix_from_gmm(original, weights, N=10)
    plan = solveMMOT([g.weights.numpy() for g in inputs], cost)
    result = mw2_barycenter(inputs, weights, iterations=10)
    assert result.cost_squared == pytest.approx(float(np.sum(plan*cost)), abs=1e-12)
    selected = tuple(result.tuples.T)
    np.testing.assert_allclose(result.ground_costs, cost[selected], atol=1e-12)
    np.testing.assert_allclose(result.gmm.means.numpy(), means[selected], atol=1e-12)
    np.testing.assert_allclose(result.gmm.covariances.numpy(), covariances[selected], atol=1e-12)
    assert result.marginal_error < 1e-10


@pytest.mark.parametrize("weights", ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]))
def test_exact_corners(weights):
    inputs = synthetic_example().inputs
    result = mw2_barycenter(inputs, weights)
    assert result.gmm is inputs[np.argmax(weights)]
    assert result.cost_squared == 0


def test_weights_and_display_orientation():
    nodes = grid_nodes(7)
    assert len(nodes) == 49
    np.testing.assert_array_equal(bilinear_weights(0, 0), [1, 0, 0, 0])
    np.testing.assert_array_equal(bilinear_weights(1, 0), [0, 1, 0, 0])
    np.testing.assert_array_equal(bilinear_weights(0, 1), [0, 0, 1, 0])
    np.testing.assert_array_equal(bilinear_weights(1, 1), [0, 0, 0, 1])
    np.testing.assert_array_equal(nodes[6][-1], [0, 1, 0, 0])
    np.testing.assert_allclose(bilinear_weights(.5, .5), [.25]*4)


def test_notebook_image_preprocessing_uses_blue_nonzero_and_vertical_flip():
    image = np.full((2, 3, 3), 255, dtype=np.uint8)
    image[0, 1, 2] = 254  # Any nonzero 1-blue, not a threshold at 0.5.
    image[1, 2, 2] = 0
    image[0, 0, 0] = 0    # Red channel must not define the foreground.
    cloud, _ = image_cloud(image)
    np.testing.assert_array_equal(cloud, [[2, 0], [1, 1]])


@pytest.mark.parametrize("kind", PROJECTION_KINDS)
@pytest.mark.parametrize("mode", ["avg", "min"])
def test_objective_and_gradients(kind, mode):
    inputs = synthetic_example().inputs
    weights = [.13, .22, .31, .34]
    initial = shared_initialization(inputs, weights, 11, seed=3)
    model = ParameterizedGMM(initial)
    # Avoid cumulative-mass coincidences, where a classical derivative does
    # not exist and centered finite differences need not match either branch.
    with torch.no_grad():
        model.logits.add_(torch.linspace(-.17, .23, initial.count))
    bank = sample_projection_bank(2, 11, kind=kind, seed=4)
    objective = BarycenterObjective(inputs, weights, bank, kind, mode)
    target = model()
    expected = sum(weight*(average_lsot(source, target, bank, kind).cost(source, target)
                           if mode == "avg" else minimum_lsot(source, target, bank, kind).cost_squared)
                   for source, weight in zip(inputs, weights))
    actual = objective(target)
    assert float(actual.detach()) == pytest.approx(float(expected.detach()), abs=1e-12)
    actual.backward()
    for parameter in model.parameters():
        for index in [0, parameter.numel()//3, parameter.numel()-1]:
            gradient = parameter.grad.reshape(-1)[index].item()
            with torch.no_grad():
                original = parameter.reshape(-1)[index].item()
                h = 1e-6
                parameter.reshape(-1)[index] = original+h
                plus = float(objective(model()))
                parameter.reshape(-1)[index] = original-h
                minus = float(objective(model()))
                parameter.reshape(-1)[index] = original
            assert gradient == pytest.approx((plus-minus)/(2*h), abs=2e-8, rel=1e-5)


def test_minimum_is_selected_separately_for_each_input():
    inputs = synthetic_example().inputs
    weights = [.13, .22, .31, .34]
    initial = shared_initialization(inputs, weights, 11, seed=3)
    bank = sample_projection_bank(2, 31, kind="Mix", seed=4)
    per_input = [minimum_lsot(source, initial, bank, "Mix").projection_costs
                 for source in inputs]
    separate = sum(weight*cost.min() for weight, cost in zip(weights, per_input))
    shared = sum(weight*cost for weight, cost in zip(weights, per_input)).min()
    loss = BarycenterObjective(inputs, weights, bank, "Mix", "min")(initial)
    assert float(loss) == pytest.approx(float(separate), abs=1e-12)
    assert float(shared-separate) > 1e-4


@pytest.mark.parametrize("kind", PROJECTION_KINDS)
@pytest.mark.parametrize("mode", ["avg", "min"])
def test_lbfgs_recovers_single_gaussian_barycenter(kind, mode):
    inputs = gaussian_example().inputs
    weights = np.ones(3)/3
    initial = shared_initialization(inputs, weights, 1, iterations=10)
    bank = sample_projection_bank(2, 8, kind=kind)
    result = optimize_barycenter(inputs, weights, initial, bank, kind, mode,
                                 steps=40, coordinate_scale=4)
    means = np.stack([g.means.numpy()[0] for g in inputs])[None]
    covariances = np.stack([g.covariances.numpy()[0] for g in inputs])[None]
    expected_mean, expected_covariance, expected_cost = gaussian_barycenters(
        means, covariances, weights, iterations=100)
    assert not result.status.startswith("numerical_stop")
    assert result.gradient_tolerance_met
    assert result.gradient_inf <= 1e-7
    assert result.objective <= result.initial_objective+1e-12
    assert result.objective == pytest.approx(expected_cost[0], abs=1e-8)
    np.testing.assert_allclose(result.gmm.means.numpy(), expected_mean, atol=1e-6)
    np.testing.assert_allclose(result.gmm.covariances.numpy(), expected_covariance, atol=1e-5)


def test_shared_initialization_respects_budget_without_mutating_inputs():
    inputs = synthetic_example().inputs
    snapshots = [g.means.clone() for g in inputs]
    for budget in [1, 7, 11]:
        initial = shared_initialization(inputs, [.25]*4, budget)
        assert initial.count == budget
        assert torch.isclose(initial.weights.sum(), torch.tensor(1., dtype=torch.float64))
        assert bool((torch.linalg.eigvalsh(initial.covariances) > 0).all())
    for source, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(source.means, snapshot)


def test_shared_starts_are_reproducible_and_preserve_the_original_first_start():
    inputs = synthetic_example().inputs
    weights = [.25]*4
    starts = shared_initializations(inputs, weights, 11, starts=4, seed=7)
    repeated = shared_initializations(inputs, weights, 11, starts=4, seed=7)
    original = shared_initialization(inputs, weights, 11, seed=7)
    assert len(starts) == 4
    torch.testing.assert_close(starts[0].means, original.means)
    for first, second in zip(starts, repeated):
        assert first.count == 11
        torch.testing.assert_close(first.weights, second.weights)
        torch.testing.assert_close(first.means, second.means)
        torch.testing.assert_close(first.covariances, second.covariances)


def test_rejected_trial_gradient_cannot_certify_convergence(monkeypatch):
    from lsot.gaussians import GMM
    from . import solver

    initial = GMM.from_numpy([1.], [[.5]], [[[.1]]])
    bank = sample_projection_bank(1, 1)
    # z=0 is a stationary maximum with loss 1; the accepted z=.5 has loss
    # .5625 and a nonzero gradient. A last-trial gradient would report false
    # convergence even though that stationary trial was rejected.
    monkeypatch.setattr(solver, "BarycenterObjective",
                        lambda *args: lambda g: (g.means[0, 0]**2-1)**2)

    class RejectedStationaryTrial:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters

        def step(self, closure):
            original = self.parameters[0].detach().clone()
            with torch.no_grad():
                self.parameters[0].zero_()
            closure()
            with torch.no_grad():
                self.parameters[0].copy_(original)

    monkeypatch.setattr(torch.optim, "LBFGS", RejectedStationaryTrial)
    result = optimize_barycenter([initial], [1.], initial, bank, "Mix", "avg",
                                 steps=2, max_restarts=0)
    assert result.gradient_inf == pytest.approx(1.5)
    assert not result.gradient_tolerance_met
    assert result.status != "gradient_tolerance"
    assert result.objective == pytest.approx(.5625)


def test_accepting_best_trial_resets_curvature_and_logs_its_actual_gradient(monkeypatch):
    from lsot.gaussians import GMM
    from . import solver

    initial = GMM.from_numpy([1.], [[.5]], [[[.1]]])
    bank = sample_projection_bank(1, 1)
    monkeypatch.setattr(solver, "BarycenterObjective",
                        lambda *args: lambda g: (g.means[0, 0]-1)**2)

    class SuboptimalAcceptedTrial:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters

        def step(self, closure):
            with torch.no_grad():
                self.parameters[0].fill_(1.)
            closure()
            with torch.no_grad():
                self.parameters[0].fill_(.75)

    monkeypatch.setattr(torch.optim, "LBFGS", SuboptimalAcceptedTrial)
    result = optimize_barycenter([initial], [1.], initial, bank, "Mix", "avg", steps=2)
    assert result.objective == 0
    assert result.gradient_inf == 0
    assert result.gradient_tolerance_met
    assert result.restarts >= 1
    retained = result.history[-1]
    assert retained["objective_squared"] == 0
    assert retained["gradient_inf"] == 0


def test_returned_gradient_is_recomputed_at_the_returned_best_candidate():
    inputs = synthetic_example().inputs
    starts = shared_initializations(inputs, [.25]*4, 11, starts=2)
    bank = sample_projection_bank(2, 12, kind="B1D")
    result = optimize_barycenter(inputs, [.25]*4, starts[0], bank, "B1D", "min",
                                 initializations=starts, steps=4, coordinate_scale=2.)
    model = ParameterizedGMM(result.gmm, coordinate_scale=2.)
    loss = BarycenterObjective(inputs, [.25]*4, bank, "B1D", "min")(model())/4.
    loss.backward()
    gradient = max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
    assert result.gradient_inf == pytest.approx(gradient, abs=1e-10)
    assert result.gradient_tolerance_met == (gradient <= 1e-7)
    best = [r["best_objective_squared"] for r in result.history]
    assert all(b <= a for a, b in zip(best, best[1:]))
    assert result.objective <= result.initial_objective
    assert result.starts == 2


def test_adam_only_never_constructs_lbfgs_and_restores_best_iterate(monkeypatch):
    from . import solver

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure Adam must not construct L-BFGS")

    monkeypatch.setattr(torch.optim, "LBFGS", forbidden)
    inputs = synthetic_example().inputs
    initial = shared_initialization(inputs, [.25]*4, 11)
    bank = sample_projection_bank(2, 10, kind="B1D")
    result = solver.optimize_barycenter(inputs, [.25]*4, initial, bank, "B1D", "min",
                                        optimizer_kind="adam", steps=6, learn_weights=False)
    assert result.iterations == 6
    assert result.restarts == 0
    assert len([r for r in result.history if r['phase'] == 'adam']) == 6
    assert not any(r['phase'] in {'lbfgs', 'curvature_restart', 'adam_warmup'}
                   for r in result.history)
    assert result.objective == min(r['best_objective_squared'] for r in result.history)
    torch.testing.assert_close(result.gmm.weights, initial.weights)
    assert result.gradient_evaluations < result.evaluations


def test_adam_only_matches_the_same_number_of_warmup_updates():
    inputs = synthetic_example().inputs
    initial = shared_initialization(inputs, [.25]*4, 11)
    bank = sample_projection_bank(2, 10, kind="Mix")
    pure = optimize_barycenter(inputs, [.25]*4, initial, bank, "Mix", "avg",
                               optimizer_kind="adam", steps=7)
    warm = optimize_barycenter(inputs, [.25]*4, initial, bank, "Mix", "avg",
                               steps=0, warmup_steps=7)
    assert pure.objective == pytest.approx(warm.objective, abs=1e-14)
    torch.testing.assert_close(pure.gmm.means, warm.gmm.means)
    torch.testing.assert_close(pure.gmm.covariances, warm.gmm.covariances)
    torch.testing.assert_close(pure.gmm.weights, warm.gmm.weights)


def test_pure_adam_rejects_an_ambiguous_extra_warmup():
    inputs = gaussian_example().inputs
    initial = shared_initialization(inputs, [1/3]*3, 1)
    bank = sample_projection_bank(2, 1)
    with pytest.raises(ValueError, match="Invalid optimizer settings"):
        optimize_barycenter(inputs, [1/3]*3, initial, bank, "Mix", "avg",
                            optimizer_kind="adam", warmup_steps=3)
