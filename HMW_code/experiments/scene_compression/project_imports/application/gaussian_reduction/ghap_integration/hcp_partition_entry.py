"""Change only GHAP's block partition; keep upstream solver/export intact.

HCP features are raw [mu, vecl(chol(Sigma))], without scaling or jitter.
GHAP's binary split size sequence is reused to match each block's budget.
"""
import hashlib
import os
import runpy
from time import perf_counter

import numpy as np
import torch
import gmm_sampler as upstream
import ghap_hot_bridge as bridge
from official_hcp_backend import OfficialHCP


def binary_sizes(n, depth):
    if depth == 0 or n <= 1:
        return [n]
    return binary_sizes(n // 2, depth - 1) + binary_sizes(n - n // 2, depth - 1)


def stable_model_factors(model):
    """Positive-diagonal Cholesky via QR of (R diag(s))^T, no jitter.

    Avoid squaring the condition number in float32 packed covariances.
    Uses original stored scale/rotation parameters in float64, for sorting only.
    """
    scales = np.exp(model._scaling.detach().cpu().numpy().astype(np.float64))
    quat = model._rotation.detach().cpu().numpy().astype(np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    output = np.empty((len(scales), 3, 3), dtype=np.float64)
    for start in range(0, len(scales), 65536):
        stop = min(start+65536, len(scales))
        w, x, y, z = quat[start:stop].T
        rot = np.empty((stop-start, 3, 3))
        rot[:, 0, 0] = 1-2*(y*y+z*z)
        rot[:, 0, 1] = 2*(x*y-w*z)
        rot[:, 0, 2] = 2*(x*z+w*y)
        rot[:, 1, 0] = 2*(x*y+w*z)
        rot[:, 1, 1] = 1-2*(x*x+z*z)
        rot[:, 1, 2] = 2*(y*z-w*x)
        rot[:, 2, 0] = 2*(x*z-w*y)
        rot[:, 2, 1] = 2*(y*z+w*x)
        rot[:, 2, 2] = 1-2*(x*x+y*y)
        a = rot * scales[start:stop, None, :]
        _, upper = np.linalg.qr(a.transpose(0, 2, 1))
        signs = np.where(np.diagonal(upper, axis1=1, axis2=2) < 0, -1., 1.)
        lower = (upper * signs[:, :, None]).transpose(0, 2, 1)
        if not np.isfinite(lower).all() or np.any(np.diagonal(lower, axis1=1, axis2=2) <= 0):
            raise ValueError('Invalid Cholesky factors from original model')
        if not np.allclose(lower @ lower.transpose(0, 2, 1), a @ a.transpose(0, 2, 1), rtol=1e-10, atol=1e-14):
            raise AssertionError('QR Cholesky reconstruction mismatch')
        output[start:stop] = lower
    return output


def partition(means, covs, depth, mode, backend, factors=None):
    tick = perf_counter()
    features = np.asarray(means, dtype=np.float64)
    if mode == 'chol_hcp':
        if factors is None:
            factors = np.linalg.cholesky(np.asarray(covs, dtype=np.float64))
        ij = np.tril_indices(means.shape[1])
        features = np.column_stack((features, factors[:, ij[0], ij[1]]))
    elif mode != 'xyz_hcp':
        raise ValueError(mode)
    representation = perf_counter() - tick
    tick = perf_counter()
    order = backend.order(features)
    ordering = perf_counter() - tick
    tick = perf_counter()
    sizes = binary_sizes(len(means), depth)
    cuts = np.cumsum([0] + sizes)
    indices = [torch.from_numpy(order[a:b].copy()) for a, b in zip(cuts[:-1], cuts[1:])]
    return indices, dict(representation_seconds=representation,
                         ordering_seconds=ordering,
                         cutting_seconds=perf_counter()-tick,
                         feature_dimension=int(features.shape[1]),
                         order_sha256=hashlib.sha256(order.tobytes()).hexdigest())


def install():
    mode = os.environ.get('GHAP_PARTITION', 'kd')
    if mode not in ('kd', 'xyz_hcp', 'chol_hcp'):
        raise ValueError(mode)
    backend = None if mode == 'kd' else OfficialHCP(os.environ['HCP_EXTENSION'])
    original_divide = upstream.KdGMR_CTD.KDdevide
    original_sampler = upstream.GMM_Sampler
    original_payload = bridge._ghap_payload
    diagnostics = {}
    blocks = []
    prepared = {}

    def divide(self):
        tick = perf_counter()
        if mode == 'kd':
            original_divide(self)
        else:
            self.index_list, stages = partition(self.all_means, self.all_covs, self.depth, mode, backend, prepared.get('factors'))
            diagnostics.update(stages)
            self.means_list = [self.all_means[x.numpy()] for x in self.index_list]
            self.covs_list = [self.all_covs[x.numpy()] for x in self.index_list]
            weights = [self.all_weights[x.numpy()] for x in self.index_list]
            self.weights_list = [w / np.sum(w) for w in weights]
            self.batch = len(self.index_list)
        diagnostics['partition_total_seconds'] = perf_counter()-tick
        sizes = [len(x) for x in self.index_list]
        if sizes != binary_sizes(len(self.all_means), self.depth):
            raise AssertionError('Block sizes differ from original KD recursion')
        diagnostics['block_count'] = len(sizes)
        diagnostics['block_sizes'] = sizes

    class RecordedSampler(original_sampler):
        def iterative(self):
            tick = perf_counter()
            iterations = super().iterative()
            if not self.converged_:
                raise RuntimeError('Upstream solver did not converge')
            if not torch.isfinite(self.reduced_means).all() or not torch.isfinite(self.reduced_covs).all():
                raise RuntimeError('Nonfinite reduced geometry')
            blocks.append(dict(source=int(self.origin_n), output=int(self.new_n),
                               iterations=int(iterations), host_seconds=perf_counter()-tick))
            return iterations

    def payload(*args, **kwargs):
        diagnostics.clear()
        blocks.clear()
        if mode == 'chol_hcp':
            tick = perf_counter()
            prepared['factors'] = stable_model_factors(args[0])
            diagnostics['qr_cholesky_seconds'] = perf_counter()-tick
        tensors, record = original_payload(*args, **kwargs)
        prepared.clear()
        record.update(partition_method=mode, partition_diagnostics=dict(diagnostics),
                      block_records=list(blocks),
                      appearance='upstream_full_block_cost_column_argmin',
                      solver='unmodified_GMM_Sampler',
                      regularization_added=False)
        record['cholesky_computation'] = 'float64_QR_from_raw_scale_rotation' if mode == 'chol_hcp' else None
        return tensors, record

    upstream.KdGMR_CTD.KDdevide = divide
    upstream.GMM_Sampler = RecordedSampler
    bridge._ghap_payload = payload


if __name__ == '__main__':
    install()
    module = ('application.gaussian_reduction.ghap_integration.time_reduction'
              if os.environ.get('GHAP_TIMING_ONLY') == '1' else
              'application.gaussian_reduction.ghap_integration.blockwise_training_entry')
    runpy.run_module(module, run_name='__main__')
