"""Busemann scalar projections using Bonet et al.'s Gaussian ray laws.

These return component locations (K,L), not sliced distances or couplings.
Reference: https://arxiv.org/abs/2510.04579, Eq. (18)-(19), Appendix B.2.
Source formulas/sampling were reviewed at upstream commit
5bb8a254f9c340a7a37af14218b7d6b06130e3d0; see ATTRIBUTION.md.

B has base N(0,I_d); B1D has base N(0,1) after the spatial projection.
The default eps=1 setting of the upstream GMM experiments is retained.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class BusemannBank:
    """Unit-speed Gaussian rays N(t*v, (I+t*S)^2), with S positive semidefinite.

    Each row satisfies ||v||^2 + ||S||_F^2 = 1. The sampler below uses the
    upstream law with both terms equal to 1/2, not a uniform tangent sphere.
    """
    mean_directions: torch.Tensor    # (L,d), normalized mean velocities v
    tangent_matrices: torch.Tensor   # (L,d,d), normalized covariance velocities S

    @property
    def count(self):
        return self.mean_directions.shape[0]

    def prefix(self, count):
        _check_count(count, self.count)
        return BusemannBank(self.mean_directions[:count], self.tangent_matrices[:count])

    def to(self, *, device=None, dtype=None):
        return BusemannBank(self.mean_directions.to(device=device, dtype=dtype),
                            self.tangent_matrices.to(device=device, dtype=dtype))


@dataclass(frozen=True)
class Busemann1DBank:
    """Spatial directions and rays N(t*a, (1+t*b)^2), a^2+b^2=1, b>=0."""
    theta: torch.Tensor        # (L,d), unit spatial directions
    mean_speeds: torch.Tensor  # (L,), a uniform on [-1,1]
    std_speeds: torch.Tensor   # (L,), b=sqrt(1-a^2); NOT uniform semicircle angles

    @property
    def count(self):
        return self.theta.shape[0]

    def prefix(self, count):
        _check_count(count, self.count)
        return Busemann1DBank(self.theta[:count], self.mean_speeds[:count], self.std_speeds[:count])

    def to(self, *, device=None, dtype=None):
        return Busemann1DBank(*(t.to(device=device, dtype=dtype) for t in
                                (self.theta, self.mean_speeds, self.std_speeds)))


def _check_count(count, maximum):
    if not 1 <= count <= maximum:
        raise ValueError("projection count must be between 1 and bank.count")


def _generator(dimension, count, seed):
    if dimension < 1 or count < 1:
        raise ValueError("dimension and count must be positive")
    return torch.Generator(device="cpu").manual_seed(seed)


def _sphere(count, dimension, generator):
    values = torch.randn(count, dimension, generator=generator, dtype=torch.float64)
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True)


def sample_busemann_bank(dimension, count, *, seed=0, device="cpu", dtype=torch.float64):
    """Sample the B law used by upstream busemann_sliced_gaussian(eps=1).

    Draw v,z independently on S^(d-1), Haar Q, and A=Q diag(|z|) Q.T.
    Divide v,A by sqrt(||v||^2+||A||_F^2), yielding a unit-speed ray.
    Sampling is on CPU float64, then transferred, as for Mix/SMix.
    Generate the largest budget once and use prefix(L) for nested budgets.
    """
    generator = _generator(dimension, count, seed)
    means = _sphere(count, dimension, generator)
    eigenvalues = _sphere(count, dimension, generator).abs()
    z = torch.randn(count, dimension, dimension, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    signs = torch.where(torch.diagonal(r, dim1=-2, dim2=-1) < 0, -1.0, 1.0)
    q = q * signs[:, None, :]
    matrices = (q * eigenvalues[:, None, :]) @ q.transpose(-1, -2)
    norm = (means.square().sum(-1) + matrices.square().sum((-2, -1))).sqrt()
    bank = BusemannBank(means / norm[:, None], matrices / norm[:, None, None])
    return bank.to(device=device, dtype=dtype)


def sample_busemann1d_bank(dimension, count, *, seed=0, device="cpu", dtype=torch.float64):
    """Sample theta uniform on S^(d-1), a uniform on [-1,1], b=sqrt(1-a^2).

    This is the upstream sliced_1Dbusemann_GMM ray law. In particular the
    (a,b) direction is not uniform on S^1, unlike the Mix/SMix mixing angle.
    """
    generator = _generator(dimension, count, seed)
    theta = _sphere(count, dimension, generator)
    mean_speeds = 2 * torch.rand(count, generator=generator, dtype=torch.float64) - 1
    std_speeds = (1 - mean_speeds.square()).clamp_min(0).sqrt()
    return Busemann1DBank(theta, mean_speeds, std_speeds).to(device=device, dtype=dtype)


def _check_inputs(means, covariances, direction, tensors):
    if means.ndim != 2 or len(means) == 0:
        raise ValueError("expected nonempty means (K,d)")
    if covariances.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError("expected covariances (K,d,d)")
    if direction.ndim != 2 or direction.shape[1] != means.shape[1] or len(direction) == 0:
        raise ValueError("projection dimension does not match the GMM or bank is empty")
    if means.dtype not in (torch.float32, torch.float64):
        raise ValueError("projections require float32 or float64")
    if any(t.device != means.device or t.dtype != means.dtype for t in (covariances, *tensors)):
        raise ValueError("GMM and projection bank must share device and dtype")


def project_busemann(means, covariances, bank, *, pair_batch_size=65536):
    """B projection, shape (K,L), for SPD covariances and a BusemannBank.

    p(v,S)(m,Sigma) = -<v,m> + tr(S) - tr(sqrt(S Sigma S)).

    This is Eq. (19) with Sigma0=I and Sigma1=(I+S)^2: T=I+S and
    C=S^2. The matrices Sigma^(1/2) S^2 Sigma^(1/2) and S Sigma S have
    the same eigenvalues, so their square-root traces coincide. We avoid
    reconstructing ray endpoints and subtracting nearly equal matrices.
    Chunking bounds temporary matrix storage by pair_batch_size, rather
    than allocating all K*L matrices at once. No component-pair costs here.
    """
    if not isinstance(bank, BusemannBank):
        raise TypeError("B requires a BusemannBank")
    v, s = bank.mean_directions, bank.tangent_matrices
    _check_inputs(means, covariances, v, (v, s))
    if s.shape != (bank.count, means.shape[1], means.shape[1]):
        raise ValueError("expected tangent_matrices (L,d,d)")
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    cross_trace = means.new_empty(len(means) * bank.count)
    for start in range(0, len(cross_trace), pair_batch_size):
        stop = min(start + pair_batch_size, len(cross_trace))
        indices = torch.arange(start, stop, device=means.device)
        component, projection = indices // bank.count, indices % bank.count
        tangent = s[projection]
        middle = tangent @ covariances[component] @ tangent
        middle = 0.5 * (middle + middle.transpose(-1, -2))
        cross_trace[start:stop] = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum(-1)
    offset = torch.diagonal(s, dim1=-2, dim2=-1).sum(-1)
    return -(means @ v.T) + offset - cross_trace.reshape(len(means), bank.count)


def project_busemann_1d(means, covariances, bank):
    """B1D projection: -a <theta,m> - b (sqrt(theta.T Sigma theta)-1).

    The second coordinate is standard deviation, not variance or log std.
    All covariance entries participate. The upstream 'ld,...ndd,ld' einsum
    discards off-diagonal entries; using 'li,kij,lj' corrects this for the
    full-covariance GMMs fitted to RGB in our experiment.
    """
    if not isinstance(bank, Busemann1DBank):
        raise TypeError("B1D requires a Busemann1DBank")
    theta, a, b = bank.theta, bank.mean_speeds, bank.std_speeds
    _check_inputs(means, covariances, theta, (theta, a, b))
    if a.shape != (bank.count,) or b.shape != (bank.count,):
        raise ValueError("expected mean_speeds and std_speeds (L,)")
    variances = torch.einsum("li,kij,lj->kl", theta, covariances, theta)
    if bool((variances <= 0).any()):
        raise ValueError("B1D requires strictly positive projected variances")
    return -a * (means @ theta.T) - b * (variances.sqrt() - 1)
