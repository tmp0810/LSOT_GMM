"""Independent B/B1D formula and ray checks, including full covariances."""
import numpy as np
import pytest
import scipy.linalg
import torch

from distribution_proj import (
    BusemannBank, Busemann1DBank, sample_busemann_bank, sample_busemann1d_bank,
    project_busemann, project_busemann_1d,
)
from lsot import GMM, average_lsot, BarycentricMap
from lsot.projections import sample_projection_bank, project_gaussians


def _gaussians(k=5, d=3):
    rng = np.random.default_rng(82)
    a = rng.normal(size=(k, d, d))
    return torch.tensor(rng.normal(size=(k, d))), torch.tensor(a @ a.transpose(0, 2, 1) + np.eye(d))


def _reference_b(means, covariances, bank):
    """General endpoint formula, Eq. (19), using independent SciPy sqrtm.

    Deliberately reconstruct T and C from the ray endpoint; the implementation
    instead computes eigvals of S Sigma S and never forms these matrices.
    """
    d = means.shape[-1]
    result = np.zeros((len(means), bank.count))
    for ell, (v, s) in enumerate(zip(bank.mean_directions.numpy(), bank.tangent_matrices.numpy())):
        endpoint = (np.eye(d) + s) @ (np.eye(d) + s)
        transport = scipy.linalg.sqrtm(endpoint)
        c = np.eye(d) - transport - transport + endpoint
        for i, (m, cov) in enumerate(zip(means.numpy(), covariances.numpy())):
            root = scipy.linalg.sqrtm(cov)
            result[i, ell] = -v @ m + np.trace(transport - np.eye(d)) - np.trace(scipy.linalg.sqrtm(root @ c @ root))
    return result


def test_b_full_covariance_matches_general_endpoint_formula_and_chunking():
    means, covariances = _gaussians()
    bank = sample_busemann_bank(3, 9, seed=91)
    expected = _reference_b(means, covariances, bank)
    actual = project_busemann(means, covariances, bank, pair_batch_size=7)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-11, rtol=1e-11)
    torch.testing.assert_close(actual, project_busemann(means, covariances, bank))
    torch.testing.assert_close(project_busemann(means, covariances, bank.prefix(4)), actual[:, :4])
    assert torch.linalg.eigvalsh(bank.tangent_matrices).min() > 0
    torch.testing.assert_close(bank.mean_directions.square().sum(-1), torch.full((9,), 0.5, dtype=torch.float64))
    torch.testing.assert_close(bank.tangent_matrices.square().sum((-2, -1)), torch.full((9,), 0.5, dtype=torch.float64))


def test_b_is_zero_at_base_and_minus_time_on_its_unit_speed_ray():
    bank = sample_busemann_bank(3, 6, seed=45)
    base = project_busemann(torch.zeros(1, 3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)[None], bank)
    torch.testing.assert_close(base, torch.zeros_like(base), atol=3e-14, rtol=0)
    for t in [0.25, 1.0, 3.0]:
        root = torch.eye(3) + t * bank.tangent_matrices
        covariances = root @ root
        values = project_busemann(t * bank.mean_directions, covariances, bank).diagonal()
        torch.testing.assert_close(values, torch.full((bank.count,), -t, dtype=torch.float64), atol=2e-13, rtol=1e-12)


def test_b1d_matches_mean_std_formula_and_includes_covariance_cross_terms():
    means, covariances = _gaussians()
    bank = sample_busemann1d_bank(3, 9, seed=91)
    expected = np.empty((len(means), bank.count))
    for i, (m, cov) in enumerate(zip(means.numpy(), covariances.numpy())):
        for ell, (theta, a, b) in enumerate(zip(bank.theta.numpy(), bank.mean_speeds.numpy(), bank.std_speeds.numpy())):
            expected[i, ell] = -a * (theta @ m) - b * (np.sqrt(theta @ cov @ theta) - 1)
    actual = project_busemann_1d(means, covariances, bank)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(project_busemann_1d(means, covariances, bank.prefix(4)), actual[:, :4])
    torch.testing.assert_close(bank.mean_speeds.square() + bank.std_speeds.square(), torch.ones(9, dtype=torch.float64))
    assert bool((bank.std_speeds >= 0).all())
    # An explicit regression for the upstream repeated-index einsum: full
    # variance is 4, whereas a diagonal-only contraction would give 3.
    one = Busemann1DBank(torch.tensor([[1., 1.]], dtype=torch.float64) / np.sqrt(2),
                         torch.tensor([0.6], dtype=torch.float64), torch.tensor([0.8], dtype=torch.float64))
    value = project_busemann_1d(torch.zeros(1, 2, dtype=torch.float64),
                                torch.tensor([[[4., 1.], [1., 2.]]], dtype=torch.float64), one)
    torch.testing.assert_close(value, torch.tensor([[-0.8]], dtype=torch.float64))


@pytest.mark.parametrize("kind", ["B", "B1D"])
def test_single_gaussian_single_projection_dtype_and_seed(kind):
    means, covariances = _gaussians(k=1, d=1)
    bank = sample_projection_bank(1, 1, kind=kind, seed=32)
    replay = sample_projection_bank(1, 1, kind=kind, seed=32)
    for name, value in vars(bank).items():
        torch.testing.assert_close(value, getattr(replay, name), atol=0, rtol=0)
    values = project_gaussians(means, covariances, bank, kind)
    assert values.shape == (1, 1)
    value32 = project_gaussians(means.float(), covariances.float(), bank.to(dtype=torch.float32), kind)
    assert value32.dtype == torch.float32
    torch.testing.assert_close(value32.double(), values, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("kind", ["B", "B1D"])
def test_busemann_average_self_plan_and_identity_map(kind):
    means, covariances = _gaussians()
    weights = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3], dtype=torch.float64)
    source = GMM(weights, means, covariances)
    order = torch.tensor([4, 2, 0, 1, 3])
    target = GMM(weights[order], means[order], covariances[order])
    bank = sample_projection_bank(3, 13, kind=kind, seed=52)
    plan = average_lsot(source, target, bank, kind)
    torch.testing.assert_close(plan.dense(), torch.diag(weights)[:, order], atol=3e-15, rtol=0)
    torch.testing.assert_close(BarycentricMap.from_plan(source, target, plan).transform(means),
                               means, atol=1e-12, rtol=1e-12)


def test_projection_family_and_bank_mismatch_are_explicit():
    means, covariances = _gaussians()
    with pytest.raises(ValueError, match="kind must be"):
        sample_projection_bank(3, 4, kind="unknown")
    with pytest.raises(TypeError, match="B requires"):
        project_gaussians(means, covariances, sample_busemann1d_bank(3, 4), "B")
    with pytest.raises(TypeError, match="B1D requires"):
        project_gaussians(means, covariances, sample_busemann_bank(3, 4), "B1D")
