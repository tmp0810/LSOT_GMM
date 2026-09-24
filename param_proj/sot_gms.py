"""Original MixSW/SMixW distances with reusable projection banks.

These functions return PROJECTED distances, not lifted LSOT costs.
Original positional arguments and sampling distributions are preserved.
Optional bank/seed arguments make comparisons reproducible. Matrix logarithms
use torch.linalg.eigh instead of requiring geoopt solely for sym_logm.
"""
import torch

from param_proj.projections import sample_projection_bank, project_gaussians
from param_proj.sw import one_dimensional_Wasserstein


def _distance(mu1s, Sigma1s, mu2s, Sigma2s, a, b, L, p, kind, bank, seed):
    if p < 1:
        raise ValueError("p must be at least one")
    if bank is None:
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        bank = sample_projection_bank(mu1s.shape[1], L, seed=seed,
                                      device=mu1s.device, dtype=mu1s.dtype)
    else:
        bank = bank.prefix(L)
    x = project_gaussians(mu1s, Sigma1s, bank, kind)
    y = project_gaussians(mu2s, Sigma2s, bank, kind)
    return one_dimensional_Wasserstein(x, y, a, b, p).mean().pow(1.0 / p)


def SMixW(mu1s, Sigma1s, mu2s, Sigma2s, a, b, L=10, p=2, *, bank=None, seed=None):
    return _distance(mu1s, Sigma1s, mu2s, Sigma2s, a, b, L, p, "SMix", bank, seed)


def MixSW(mu1s, Sigma1s, mu2s, Sigma2s, a, b, L=10, p=2, *, bank=None, seed=None):
    return _distance(mu1s, Sigma1s, mu2s, Sigma2s, a, b, L, p, "Mix", bank, seed)
