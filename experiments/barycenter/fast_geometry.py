"""Exact Gaussian costs with an algebraic 2-D path (no change of geometry)."""
from dataclasses import dataclass

import torch

from lsot.gaussians import gaussian_pair_costs
from lsot.projections import project_gaussians


@dataclass
class TensorGMM:
    """Internal view of already parameterized tensors; validate at checkpoints.

    This is deliberately private to the optimization path. Public GMM inputs
    and returned solutions still pass the full GMM validation.
    """
    weights: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor

    @property
    def count(self):
        return len(self.weights)

    @property
    def dimension(self):
        return self.means.shape[1]


def pair_costs(source, target, rows=None, cols=None):
    """Dense matrix or requested edges. Exact for SPD inputs, float64 recommended."""
    dense = rows is None
    if dense:
        rows = torch.arange(source.count, device=source.weights.device).repeat_interleave(target.count)
        cols = torch.arange(target.count, device=target.weights.device).repeat(source.count)
    if source.dimension == 2:
        a, b = source.covariances[rows], target.covariances[cols]
        det_a = (a[:, 0, 0]*a[:, 1, 1]-a[:, 0, 1]*a[:, 1, 0]).clamp_min(0)
        det_b = (b[:, 0, 0]*b[:, 1, 1]-b[:, 0, 1]*b[:, 1, 0]).clamp_min(0)
        # tr(sqrt(sqrt(A) B sqrt(A))) = sqrt(tr(AB)+2 sqrt(det(A)det(B))).
        cross = ((a*b.transpose(-1, -2)).sum((-2, -1))
                 + 2*(det_a*det_b).sqrt()).clamp_min(0).sqrt()
        trace = torch.diagonal(a+b, dim1=-2, dim2=-1).sum(-1)
        value = ((source.means[rows]-target.means[cols]).square().sum(-1)
                 + trace-2*cross).clamp_min(0)
    else:
        value = gaussian_pair_costs(source, target, rows, cols)
    return value.reshape(source.count, target.count) if dense else value


def project_components(gmm, bank, kind):
    if kind != "B" or gmm.dimension != 2:
        return project_gaussians(gmm.means, gmm.covariances, bank, kind)
    # tr(sqrt(S Sigma S)) in dimension two, including singular PSD S.
    s, a = bank.tangent_matrices, gmm.covariances
    tr_middle = torch.einsum("lab,kba->kl", s @ s, a)
    det_a = (a[:, 0, 0]*a[:, 1, 1]-a[:, 0, 1]*a[:, 1, 0]).clamp_min(0)
    det_s = (s[:, 0, 0]*s[:, 1, 1]-s[:, 0, 1]*s[:, 1, 0]).abs()
    cross = (tr_middle+2*det_a.sqrt()[:, None]*det_s[None]).clamp_min(0).sqrt()
    return (-gmm.means @ bank.mean_directions.T
            + torch.diagonal(s, dim1=-2, dim2=-1).sum(-1)[None]-cross)
