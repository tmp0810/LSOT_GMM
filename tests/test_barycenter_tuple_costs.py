"""Check batched tuple geometry and the CPU LP's tuple ordering."""
import numpy as np
import pytest
import torch

from experiments.barycenter.data import synthetic_example
from experiments.barycenter.reference import gaussian_barycenters, mw2_barycenter
from experiments.barycenter.tuple_costs import BatchedGaussianTuples, tuple_constraints


def test_batched_tuple_constraints_have_correct_marginal_rows():
    shape = (2, 3, 4)
    tuples = np.indices(shape).reshape(len(shape), -1).T
    expected = np.zeros((sum(shape), len(tuples)))
    for j, offset in enumerate(np.cumsum((0,) + shape[:-1])):
        expected[tuples[:, j] + offset, np.arange(len(tuples))] = 1
    np.testing.assert_array_equal(tuple_constraints(shape, 7).toarray(), expected)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('tolerance', [0., 1e-8])
def test_batched_gaussian_costs_and_lp_agree_with_numpy(device, tolerance):
    inputs = synthetic_example().inputs
    weights = np.array([.13, .27, .23, .37])
    means = [g.means.numpy() for g in inputs]
    covariances = [g.covariances.numpy() for g in inputs]
    shape = tuple(len(m) for m in means)
    tuples = np.indices(shape).reshape(len(shape), -1).T
    tm = np.stack([m[tuples[:, j]] for j,m in enumerate(means)], axis=1)
    tc = np.stack([c[tuples[:, j]] for j,c in enumerate(covariances)], axis=1)
    iterations, batch_size = (15 if not tolerance else 100), 17
    batched = BatchedGaussianTuples(means, covariances, weights, device, batch_size)
    cost, steps, info = batched.costs(iterations, tolerance)
    assert len(steps) > 1 and info['covariance_iterations'] <= iterations
    # Use each chunk's iteration count to independently check its costs and replayed centers.
    for start in (0, 23, 46, len(tuples)-1):
        indices = np.arange(start, min(start+11, len(tuples)))
        indices = indices[(indices//batch_size) == start//batch_size]
        mean, cov, expected = gaussian_barycenters(tm[indices], tc[indices], weights,
                                                     iterations=steps[start//batch_size], fast=True)
        replay_mean, replay_cov = batched.active_centers(indices, steps)
        np.testing.assert_allclose(replay_mean, mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(replay_cov, cov, atol=1e-12, rtol=0)
        np.testing.assert_allclose(cost[indices], expected, atol=1e-12, rtol=0)
    times = {}
    kwargs = dict(fast=True, iterations=iterations, tolerance=tolerance, lp_threads=1)
    reference = mw2_barycenter(inputs, weights, **kwargs)
    hybrid = mw2_barycenter(inputs, weights, **kwargs, gaussian_device=device,
                            tuple_batch_size=batch_size, timings=times)
    assert hybrid.cost_squared == pytest.approx(reference.cost_squared, abs=1e-10)
    assert hybrid.marginal_error < 1e-8
    assert hybrid.covariance_tolerance_met == reference.covariance_tolerance_met
    assert min(times.values()) >= 0.
