"""Reusable Mix/SMix projections with the sampling law of sot_gms.py.

Mix uses A = Q diag(eigen_direction) Q.T, NOT a uniformly distributed
Frobenius-sphere symmetric matrix. The original spectral sampling is preserved.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ProjectionBank:
    theta: torch.Tensor       # (L, d), unit mean/spatial directions
    psi: torch.Tensor         # (L, 2), unit mixing directions
    matrices: torch.Tensor    # (L, d, d), Mix covariance directions

    @property
    def count(self):
        return self.theta.shape[0]

    def prefix(self, count):
        if not 1 <= count <= self.count:
            raise ValueError("projection count must be between 1 and bank.count")
        return ProjectionBank(self.theta[:count], self.psi[:count], self.matrices[:count])

    def to(self, *, device=None, dtype=None):
        return ProjectionBank(*(t.to(device=device, dtype=dtype) for t in
                                (self.theta, self.psi, self.matrices)))


def sample_projection_bank(dimension, count, *, seed=0, device="cpu", dtype=torch.float64):
    """Generate on CPU for identical saved banks across CPU/CUDA; then transfer.

    Generate the largest budget once and use .prefix(L) for nested budgets.
    Mix and SMix share theta and psi, while only Mix uses matrices.
    """
    if dimension < 1 or count < 1:
        raise ValueError("dimension and count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def sphere(width):
        x = torch.randn(count, width, generator=generator, dtype=torch.float64)
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    theta, psi, eigen_direction = sphere(dimension), sphere(2), sphere(dimension)
    z = torch.randn(count, dimension, dimension, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(z)
    signs = torch.where(torch.diagonal(r, dim1=-2, dim2=-1) < 0, -1.0, 1.0)
    q = q * signs[:, None, :]
    matrices = (q * eigen_direction[:, None, :]) @ q.transpose(-1, -2)
    return ProjectionBank(theta, psi, matrices).to(device=device, dtype=dtype)


def project_gaussians(means, covariances, bank, kind):
    """Return scalar locations (K,L), NOT distances or plans.

    Mix:  psi_1 <theta,m> + psi_2 <A,log(Sigma)>_F
    SMix: psi_1 <theta,m> + psi_2 log(sqrt(theta.T Sigma theta))
    """
    if kind not in {"Mix", "SMix"}:
        raise ValueError("kind must be 'Mix' or 'SMix'")
    if means.ndim != 2 or covariances.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError("expected means (K,d) and covariances (K,d,d)")
    if means.device != bank.theta.device or means.dtype != bank.theta.dtype:
        raise ValueError("GMM and projection bank must share device and dtype")
    if means.shape[1] != bank.theta.shape[1]:
        raise ValueError("projection dimension does not match the GMM")
    projected_means = means @ bank.theta.T
    if kind == "Mix":
        values, vectors = torch.linalg.eigh(covariances)
        if bool((values <= 0).any()):
            raise ValueError("Mix requires positive definite covariances")
        log_cov = (vectors * values.log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        projected_cov = torch.einsum("lij,kij->kl", bank.matrices, log_cov)
    else:
        variances = torch.einsum("li,kij,lj->kl", bank.theta, covariances, bank.theta)
        if bool((variances <= 0).any()):
            raise ValueError("SMix requires strictly positive projected variances")
        projected_cov = 0.5 * variances.log()
    return bank.psi[:, 0] * projected_means + bank.psi[:, 1] * projected_cov
