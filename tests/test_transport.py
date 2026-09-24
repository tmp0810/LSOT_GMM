"""Independent mathematical checks against SciPy/POT and dense fiber lifting."""
import numpy as np
import pytest
import scipy.linalg
import torch

import gmmot
from lsot import GMM, BarycentricMap, average_lsot, solve_mw2
from lsot.gaussians import gaussian_pair_costs
from lsot.maps import gaussian_pair_maps
from lsot.plans import average_projected_plans, lift_projection, SparsePlan
from param_proj import sample_projection_bank, project_gaussians
from param_proj.sot_gms import MixSW, SMixW
from param_proj.sw import one_dimensional_Wasserstein

torch.set_num_threads(1)


def make_gmm(k=5, d=3, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(k, d, d))
    covariances = matrix @ matrix.transpose(0, 2, 1) + 0.4 * np.eye(d)
    return GMM.from_numpy(rng.dirichlet(np.ones(k)), rng.normal(size=(k, d)), covariances, device=device)


def dense_reference_lift(x, y, a, b):
    """Literal interval formula on aggregated fibers, independent of Torch code."""
    u, v = np.unique(x), np.unique(y)
    ua = np.array([a[x == z].sum() for z in u])
    vb = np.array([b[y == z].sum() for z in v])
    ca, cb = np.r_[0, ua.cumsum()], np.r_[0, vb.cumsum()]
    result = np.zeros((len(a), len(b)))
    for r in range(len(u)):
        for t in range(len(v)):
            mass = max(0, min(ca[r + 1], cb[t + 1]) - max(ca[r], cb[t]))
            i, j = np.where(x == u[r])[0], np.where(y == v[t])[0]
            result[np.ix_(i, j)] = mass * np.outer(a[i] / ua[r], b[j] / vb[t])
    return result


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_lifting_matches_dense_formula_and_projected_ot(ties):
    rng = np.random.default_rng(10)
    for k, m in [(1, 1), (1, 7), (6, 1), (5, 9), (17, 11)]:
        a, b = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(m))
        x, y = rng.normal(size=(k, 5)), rng.normal(size=(m, 5))
        if ties:
            x, y = np.round(x), np.round(y)
        xt, yt, at, bt = [torch.tensor(z, dtype=torch.float64) for z in (x, y, a, b)]
        expected = sum(dense_reference_lift(x[:, ell], y[:, ell], a, b) for ell in range(5)) / 5
        result = average_projected_plans(xt, yt, at, bt).dense().numpy()
        np.testing.assert_allclose(result, expected, atol=3e-15, rtol=0)
        np.testing.assert_allclose(result.sum(1), a, atol=3e-15)
        np.testing.assert_allclose(result.sum(0), b, atol=3e-15)
        projected_costs = one_dimensional_Wasserstein(xt, yt, at, bt, 2).ravel()
        for ell in range(5):
            plan = lift_projection(xt[:, ell], yt[:, ell], at, bt)
            cost = torch.sum(plan.mass * (xt[plan.rows, ell] - yt[plan.cols, ell]).square())
            torch.testing.assert_close(cost, projected_costs[ell], atol=1e-12, rtol=1e-12)


def test_constant_projection_uses_product_lift_not_identity_or_mask():
    a = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    zeros = torch.zeros(3, 1, dtype=torch.float64)
    actual = average_projected_plans(zeros, zeros, a, a).dense()
    torch.testing.assert_close(actual, torch.outer(a, a))


@pytest.mark.parametrize("kind", ["Mix", "SMix"])
def test_projection_formulas_and_legacy_distance(kind):
    source, target = make_gmm(seed=7), make_gmm(k=4, seed=8)
    bank = sample_projection_bank(3, 7, seed=11)
    torch.testing.assert_close(torch.linalg.vector_norm(bank.theta, dim=1), torch.ones(7, dtype=torch.float64))
    torch.testing.assert_close(torch.linalg.matrix_norm(bank.matrices), torch.ones(7, dtype=torch.float64))
    torch.testing.assert_close(bank.matrices, bank.matrices.transpose(-1, -2))
    expected = np.zeros((source.count, bank.count))
    for i, (mean, cov) in enumerate(zip(source.means.numpy(), source.covariances.numpy())):
        for ell in range(bank.count):
            theta, psi, matrix = [t[ell].numpy() for t in (bank.theta, bank.psi, bank.matrices)]
            covariance_part = np.sum(matrix * scipy.linalg.logm(cov)) if kind == "Mix" else np.log(np.sqrt(theta @ cov @ theta))
            expected[i, ell] = psi[0] * (theta @ mean) + psi[1] * covariance_part
    actual = project_gaussians(source.means, source.covariances, bank, kind)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
    short = project_gaussians(source.means, source.covariances, bank.prefix(3), kind)
    torch.testing.assert_close(short, actual[:, :3])
    function = MixSW if kind == "Mix" else SMixW
    distance = function(source.means, source.covariances, target.means, target.covariances,
                        source.weights, target.weights, L=7, bank=bank)
    projected_target = project_gaussians(target.means, target.covariances, bank, kind)
    expected_distance = one_dimensional_Wasserstein(actual, projected_target, source.weights, target.weights).mean().sqrt()
    torch.testing.assert_close(distance, expected_distance)


def test_gaussian_geometry_against_original_and_pushforward_covariance():
    source, target = make_gmm(seed=2), make_gmm(seed=3)
    rows = torch.arange(5)
    costs = gaussian_pair_costs(source, target, rows, rows).numpy()
    matrices, offsets = gaussian_pair_maps(source, target, rows, rows)
    points = np.random.default_rng(4).normal(size=(13, 3))
    for i in range(5):
        m, n, s, t = [z[i].numpy() for z in (source.means, target.means, source.covariances, target.covariances)]
        np.testing.assert_allclose(costs[i], gmmot.GaussianW2(m, n, s, t), rtol=1e-11)
        np.testing.assert_allclose(points @ matrices[i].numpy().T + offsets[i].numpy(),
                                   gmmot.GaussianMap(m, n, s, t, points), atol=2e-12)
        torch.testing.assert_close(matrices[i] @ source.covariances[i] @ matrices[i].T,
                                   target.covariances[i], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("kind", ["Mix", "SMix"])
def test_self_transport_permuted_components_and_cost_lower_bound(kind):
    source = make_gmm(seed=23)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    target = GMM(source.weights[permutation], source.means[permutation], source.covariances[permutation])
    bank = sample_projection_bank(3, 13, seed=3)
    plan = average_lsot(source, target, bank, kind)
    expected = torch.diag(source.weights)[:, permutation]
    torch.testing.assert_close(plan.dense(), expected, atol=2e-15, rtol=0)
    points = torch.randn(19, 3, dtype=torch.float64)
    torch.testing.assert_close(BarycentricMap.from_plan(source, target, plan).transform(points), points, atol=1e-11, rtol=1e-11)
    unrelated = make_gmm(k=7, seed=52)
    lifted = average_lsot(source, unrelated, bank, kind)
    _, optimal_cost = solve_mw2(source, unrelated)
    assert lifted.cost(source, unrelated).item() >= optimal_cost - 1e-11
    assert lifted.marginal_error(source, unrelated) < 1e-14
    individual_costs = [average_lsot(source, unrelated,
                        type(bank)(bank.theta[i:i+1], bank.psi[i:i+1], bank.matrices[i:i+1]), kind).cost(source, unrelated)
                        for i in range(bank.count)]
    torch.testing.assert_close(lifted.cost(source, unrelated), torch.stack(individual_costs).mean())


def test_barycentric_map_matches_original_pixelwise_formula():
    from scipy.stats import multivariate_normal
    source, target = make_gmm(seed=13), make_gmm(k=4, seed=14)
    plan, _ = solve_mw2(source, target)
    rng = np.random.default_rng(5)
    points = rng.normal(size=(23, 3))
    densities = np.stack([multivariate_normal.pdf(points, mean=m, cov=c)
                          for m, c in zip(source.means.numpy(), source.covariances.numpy())], axis=1)
    posterior = densities * source.weights.numpy()
    posterior /= posterior.sum(1, keepdims=True)
    expected = np.zeros_like(points)
    dense = plan.dense().numpy()
    for i in range(source.count):
        for j in range(target.count):
            mapped = gmmot.GaussianMap(source.means[i].numpy(), target.means[j].numpy(),
                                      source.covariances[i].numpy(), target.covariances[j].numpy(), points)
            expected += dense[i, j] / source.weights[i].item() * posterior[:, i, None] * mapped
    mapper = BarycentricMap.from_plan(source, target, plan, pair_batch_size=3)
    actual = mapper.transform(torch.tensor(points), batch_size=7).numpy()
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["Mix", "SMix"])
def test_cuda_matches_cpu(kind):
    cpu_s, cpu_t = make_gmm(seed=12), make_gmm(k=7, seed=13)
    gpu_s, gpu_t = make_gmm(seed=12, device="cuda"), make_gmm(k=7, seed=13, device="cuda")
    bank = sample_projection_bank(3, 9, seed=17)
    cpu_plan = average_lsot(cpu_s, cpu_t, bank, kind)
    gpu_plan = average_lsot(gpu_s, gpu_t, bank.to(device="cuda"), kind)
    torch.testing.assert_close(cpu_plan.dense(), gpu_plan.dense().cpu(), atol=1e-12, rtol=1e-12)
    points = torch.randn(10, 3, dtype=torch.float64)
    a = BarycentricMap.from_plan(cpu_s, cpu_t, cpu_plan).transform(points)
    b = BarycentricMap.from_plan(gpu_s, gpu_t, gpu_plan).transform(points.cuda()).cpu()
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
