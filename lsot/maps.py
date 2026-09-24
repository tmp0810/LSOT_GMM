"""The MW2 Tmean construction, applied to any admissible component plan."""
from dataclasses import dataclass
import math
import torch

from .gaussians import covariance_roots, symmetric_sqrt


def gaussian_pair_maps(source, target, rows, cols):
    roots, inverse_roots = covariance_roots(source.covariances)
    middle = roots[rows] @ target.covariances[cols] @ roots[rows]
    matrices = inverse_roots[rows] @ symmetric_sqrt(middle) @ inverse_roots[rows]
    matrices = (matrices + matrices.transpose(-1, -2)) * 0.5
    offsets = target.means[cols] - torch.einsum("bij,bj->bi", matrices, source.means[rows])
    return matrices, offsets


@dataclass
class BarycentricMap:
    source: object
    matrices: torch.Tensor
    offsets: torch.Tensor
    precision: torch.Tensor
    log_normalizer: torch.Tensor

    @classmethod
    def from_plan(cls, source, target, plan, *, pair_batch_size=65536):
        """Collapse pair maps into one affine map per source component.

        B_i = sum_j P_ij/alpha_i M_ij;
        b_i = sum_j P_ij/alpha_i (n_j - M_ij m_i).
        No K0*K1*Npixels tensor is constructed.
        """
        if plan.shape != (source.count, target.count):
            raise ValueError("plan shape does not match the GMMs")
        matrices = torch.zeros_like(source.covariances)
        offsets = torch.zeros_like(source.means)
        for start in range(0, plan.nnz, pair_batch_size):
            sl = slice(start, start + pair_batch_size)
            i, j = plan.rows[sl], plan.cols[sl]
            pair_matrices, pair_offsets = gaussian_pair_maps(source, target, i, j)
            conditional = plan.mass[sl] / source.weights[i]
            matrices.index_add_(0, i, conditional[:, None, None] * pair_matrices)
            offsets.index_add_(0, i, conditional[:, None] * pair_offsets)
        cholesky = torch.linalg.cholesky(source.covariances)
        precision = torch.cholesky_inverse(cholesky)
        logdet = 2 * torch.diagonal(cholesky, dim1=-2, dim2=-1).log().sum(-1)
        normalizer = source.weights.log() - 0.5 * (source.dimension * math.log(2 * math.pi) + logdet)
        return cls(source, matrices, offsets, precision, normalizer)

    def responsibilities(self, points):
        diff = points[:, None, :] - self.source.means[None]
        squared = torch.einsum("nki,kij,nkj->nk", diff, self.precision, diff)
        return torch.softmax(self.log_normalizer[None] - 0.5 * squared, dim=1)

    def transform(self, points, *, batch_size=16384):
        """T_P(x)=sum_i posterior_i(x) (B_i x+b_i), returned without clipping."""
        if batch_size < 1 or points.ndim != 2 or points.shape[1] != self.source.dimension:
            raise ValueError("expected points (N,d) and a positive batch_size")
        output = torch.empty_like(points)
        for start in range(0, len(points), batch_size):
            x = points[start:start + batch_size]
            posterior = self.responsibilities(x)
            mapped = torch.einsum("kij,nj->nki", self.matrices, x) + self.offsets[None]
            output[start:start + batch_size] = (posterior[..., None] * mapped).sum(1)
        return output
