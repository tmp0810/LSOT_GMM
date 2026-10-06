"""Numerical checks against upstream MW2 and existing lifted LSOT solvers."""
import numpy as np
import pytest
import torch

from gmmot import create_cost_matrix_from_gmm, solveMMOT
from lsot.plans import average_lsot, minimum_lsot
from lsot.projections import PROJECTION_KINDS, sample_projection_bank
from .data import synthetic_example, gaussian_example, image_cloud, bilinear_weights, grid_nodes
from .reference import mw2_barycenter, shared_initialization, gaussian_barycenters
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
