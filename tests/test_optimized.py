"""Check the Stein search returns an actual feasible single-projection lift."""

import pytest
import numpy as np
import torch

from lsot import GMM, minimum_lsot, optimized_minimum_lsot
from lsot.projections import sample_projection_bank


def make_gmm(k=5, d=3, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(k, d, d))
    covariances = matrix @ matrix.transpose(0, 2, 1) + 0.4 * np.eye(d)
    return GMM.from_numpy(rng.dirichlet(np.ones(k)), rng.normal(size=(k, d)), covariances)


@pytest.mark.parametrize("kind", ["Mix", "SMix", "B", "B1D"])
def test_optimized_lift_replays_and_never_worsens_initial_bank(kind):
    source = make_gmm(k=4, seed=71)
    target = make_gmm(k=5, seed=72)
    bank = sample_projection_bank(3, 6, kind=kind, seed=73)
    start = minimum_lsot(source, target, bank, kind)
    answer = optimized_minimum_lsot(source, target, bank, kind,
                                     steps=3, samples=3, seed=74)
    again = optimized_minimum_lsot(source, target, bank, kind,
                                    steps=3, samples=3, seed=74)
    assert answer.initial_projection_index == start.projection_index
    torch.testing.assert_close(answer.initial_projection_costs, start.projection_costs)
    assert answer.cost_squared.item() <= start.cost_squared.item() + 1e-12
    assert answer.plan.marginal_error(source, target) < 1e-13
    torch.testing.assert_close(answer.plan.cost(source, target), answer.cost_squared,
                               atol=1e-11, rtol=1e-12)
    replay = minimum_lsot(source, target, answer.best_bank, kind)
    torch.testing.assert_close(replay.plan.dense(), answer.plan.dense(), atol=1e-14, rtol=0)
    torch.testing.assert_close(replay.cost_squared, answer.cost_squared, atol=1e-11, rtol=1e-12)
    torch.testing.assert_close(again.plan.dense(), answer.plan.dense(), atol=1e-14, rtol=0)
    torch.testing.assert_close(again.cost_history, answer.cost_history)
    assert len(answer.cost_history) == 5
    assert bool((answer.cost_history[1:] <= answer.cost_history[:-1] + 1e-12).all())


@pytest.mark.parametrize("kind", ["Mix", "SMix", "B", "B1D"])
def test_zero_updates_equals_finite_minimum(kind):
    source, target = make_gmm(seed=21), make_gmm(k=6, seed=22)
    bank = sample_projection_bank(3, 4, kind=kind, seed=23)
    finite = minimum_lsot(source, target, bank, kind)
    optimized = optimized_minimum_lsot(source, target, bank, kind, steps=0)
    torch.testing.assert_close(optimized.plan.dense(), finite.plan.dense(), atol=1e-14, rtol=0)
    torch.testing.assert_close(optimized.cost_squared, finite.cost_squared, atol=1e-11, rtol=1e-12)
