"""Dispatch scalar component projections; lifting is shared by all families."""
import torch

from param_proj import sample_projection_bank as sample_parameter_bank
from param_proj import project_gaussians as project_parameters
from distribution_proj import (
    sample_busemann_bank, sample_busemann1d_bank,
    project_busemann, project_busemann_1d,
)

PROJECTION_KINDS = ("Mix", "SMix", "B", "B1D")


def sample_projection_bank(dimension, count, *, kind="Mix", seed=0,
                           device="cpu", dtype=torch.float64):
    """Build a reusable bank for one family. Mix/SMix accept the same bank."""
    samplers = {"Mix": sample_parameter_bank, "SMix": sample_parameter_bank,
                "B": sample_busemann_bank, "B1D": sample_busemann1d_bank}
    if kind not in samplers:
        raise ValueError(f"kind must be one of {PROJECTION_KINDS}")
    return samplers[kind](dimension, count, seed=seed, device=device, dtype=dtype)


def project_gaussians(means, covariances, bank, kind):
    """Return scalar component locations (K,L) for Mix, SMix, B or B1D."""
    if kind in {"Mix", "SMix"}:
        return project_parameters(means, covariances, bank, kind)
    if kind == "B":
        return project_busemann(means, covariances, bank)
    if kind == "B1D":
        return project_busemann_1d(means, covariances, bank)
    raise ValueError(f"kind must be one of {PROJECTION_KINDS}")
