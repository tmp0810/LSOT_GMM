"""Regression for large small-matrix batches rejected by CUDA cuSOLVER."""
from unittest.mock import patch

import pytest
import torch

from distribution_proj import sample_busemann_bank, project_busemann
from lsot import GMM, BarycentricMap
from lsot.gaussians import gaussian_pair_costs
from lsot.plans import SparsePlan


def limited_solver(original, calls):
    def solve(matrix, *args, **kwargs):
        count = matrix.numel() // (matrix.shape[-1] ** 2)
        calls.append(count)
        assert count <= 4096, "oversized batch reached eigensolver"
        return original(matrix, *args, **kwargs)
    return solve


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_b_projection_k100_l500_never_submits_oversized_batch(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(46)
    means = torch.randn(100, 3, generator=generator, dtype=torch.float64)
    factors = torch.randn(100, 3, 3, generator=generator, dtype=torch.float64)
    covariances = factors @ factors.transpose(-1, -2) + 0.2 * torch.eye(3)
    bank = sample_busemann_bank(3, 500, seed=17)
    # Reference uses a different, smaller partition on CPU.
    expected = project_busemann(means, covariances, bank, pair_batch_size=997)
    calls = []
    with patch("torch.linalg.eigvalsh", limited_solver(torch.linalg.eigvalsh, calls)):
        actual = project_busemann(means.to(device), covariances.to(device),
                                  bank.to(device=device), pair_batch_size=65536)
    assert len(calls) == 13 and sum(calls) == 50000
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-10, rtol=2e-10)


def test_gaussian_costs_and_map_setup_cap_large_pair_batches():
    n = 100
    weights = torch.full((n,), 1 / n, dtype=torch.float64)
    means = torch.arange(n, dtype=torch.float64)[:, None].expand(-1, 3) / n
    covariances = torch.eye(3, dtype=torch.float64).repeat(n, 1, 1)
    source = GMM(weights, means, covariances)
    target = GMM(weights, means + 0.2, covariances)
    plan = SparsePlan.from_dense(torch.outer(weights, weights))
    expected_costs = (means[plan.rows] - target.means[plan.cols]).square().sum(-1)
    calls = []
    with patch("torch.linalg.eigvalsh", limited_solver(torch.linalg.eigvalsh, calls)), \
            patch("torch.linalg.eigh", limited_solver(torch.linalg.eigh, calls)):
        costs = gaussian_pair_costs(source, target, plan.rows, plan.cols)
        mapper = BarycentricMap.from_plan(source, target, plan)
    assert calls and max(calls) <= 4096
    torch.testing.assert_close(costs, expected_costs, atol=2e-14, rtol=1e-12)
    # For equal identity covariances, all pair linear maps are I. The
    # product plan's source-i offset is target mean minus source mean_i.
    torch.testing.assert_close(mapper.matrices, covariances, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(mapper.offsets, target.means.mean(0) - means, atol=1e-13, rtol=1e-13)


def test_b_rejects_nonfinite_inputs_before_eigensolver():
    bank = sample_busemann_bank(3, 2)
    covariances = torch.eye(3, dtype=torch.float64)[None]
    covariances[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        project_busemann(torch.zeros(1, 3, dtype=torch.float64), covariances, bank)
