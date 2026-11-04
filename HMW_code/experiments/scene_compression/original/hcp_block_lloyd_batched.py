"""Hilbert-local weighted Lloyd clustering with GHAP geometry cost.

Start from exactly the previous XYZ fixed groups, allow assignments to move
within blocks of at most 100 representatives. Preserve the exact budget.
"""
import os
from time import perf_counter
import numpy as np
import torch
from gmm_sampler import TensorKL
from official_hcp_backend import OfficialHCP
from application.gaussian_reduction.ghap_integration.hcp_partition_entry import stable_model_factors
from ghap_hot_bridge import matrix_to_quaternion
import hashlib


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


from hcp_batched_solver import solve_batch


def payload(model, target_components, seed):
    stages = {}
    tick = perf_counter()
    mu = model._xyz.detach().cpu().numpy().astype(np.float64)
    factors = stable_model_factors(model)
    cov = factors @ factors.transpose(0, 2, 1)
    a = model.get_opacity.detach().cpu().numpy().astype(np.float64).ravel()
    stages['input_and_covariance'] = perf_counter()-tick
    tick = perf_counter()
    order = OfficialHCP(os.environ['HCP_EXTENSION']).order(mu.copy())
    stages['hcp_order'] = perf_counter()-tick
    tick = perf_counter()
    n, m = len(mu), target_components
    cuts = np.arange(m+1, dtype=np.int64)*n//m
    k = (m+99)//100
    rep_cuts = np.arange(k+1, dtype=np.int64)*m//k
    labels_sorted = np.repeat(np.arange(m), np.diff(cuts))
    stages['blocking'] = perf_counter()-tick
    tick = perf_counter()
    device = model._xyz.device
    output_mu, output_cov, donors, records = [None]*k, [None]*k, [None]*k, [None]*k
    final_labels = np.empty(n, dtype=np.int64)
    buckets = {}
    for block, (left,right) in enumerate(zip(rep_cuts[:-1],rep_cuts[1:])):
        key = (int(cuts[right]-cuts[left]),int(right-left))
        buckets.setdefault(key,[]).append((block,int(left),int(right)))
    batch_size = int(os.environ.get('HCP_BATCH_SIZE','16'))
    with torch.no_grad():
        for (block_n,block_m), entries in buckets.items():
            for start in range(0,len(entries),batch_size):
                batch = entries[start:start+batch_size]
                ids = np.stack([order[cuts[left]:cuts[right]] for _,left,right in batch])
                init = np.stack([labels_sorted[cuts[left]:cuts[right]]-left for _,left,right in batch])
                def tensor(x,dtype=torch.float64):
                    return torch.as_tensor(x,dtype=dtype,device=device)
                centers,shapes,donor,labels,recs = solve_batch(
                    tensor(mu[ids]),tensor(cov[ids]),tensor(a[ids]),tensor(init,torch.long),block_m)
                centers,shapes,donor,labels = [x.cpu().numpy() for x in (centers,shapes,donor,labels)]
                for q,(block,left,right) in enumerate(batch):
                    output_mu[block] = centers[q]
                    output_cov[block] = shapes[q]
                    donors[block] = ids[q,donor[q]]
                    final_labels[ids[q]] = labels[q]+left
                    recs[q].update(source=block_n,output=block_m)
                    records[block] = recs[q]
            print('BUCKET_COMPLETE',block_n,block_m,len(entries),flush=True)
    torch.cuda.synchronize()
    stages['block_lloyd_and_donor'] = perf_counter()-tick
    tick = perf_counter()
    means = np.concatenate(output_mu)
    covariance = np.concatenate(output_cov)
    donor = np.concatenate(donors)
    vals, vecs = np.linalg.eigh(covariance)
    assert np.isfinite(vals).all() and np.all(vals[:, -1] > 0)
    # Weighted SPD inputs are PSD in floating point; extreme anisotropy can
    # produce tiny nonpositive eigenvalues. Reject substantive indefiniteness.
    floor = np.maximum(64*np.finfo(np.float64).eps*vals[:, -1], np.finfo(np.float64).tiny)
    assert np.all(vals[:, 0] >= -floor), 'Covariance is materially indefinite'
    eigen_repaired = int(np.sum(np.any(vals <= 0, axis=1)))
    minimum_relative_eigenvalue = float(np.min(vals[:, 0]/vals[:, -1]))
    vals = np.where(vals <= 0, floor[:, None], vals)
    vecs[:, :, 0] *= np.where(np.linalg.det(vecs) < 0, -1., 1.)[:, None]
    dtype = model._xyz.dtype
    indices = torch.as_tensor(donor, device=device)
    tensors = dict(xyz=torch.as_tensor(means, device=device, dtype=dtype),
        scaling=torch.as_tensor(.5*np.log(vals), device=device, dtype=dtype),
        rotation=matrix_to_quaternion(torch.as_tensor(vecs, device=device)).to(dtype),
        f_dc=model._features_dc.detach()[indices].clone(),
        f_rest=model._features_rest.detach()[indices].clone(),
        opacity=model._opacity.detach()[indices].clone())
    torch.cuda.synchronize()
    stages['output_conversion'] = perf_counter()-tick
    initial_labels = np.empty(n, dtype=np.int64)
    initial_labels[order] = labels_sorted
    record = dict(batch_size=batch_size, solver_execution="batched_float64", stages_seconds=stages, block_records=records, target=m, source=n,
        order_sha256=digest(order), initial_labels_sha256=digest(initial_labels),
        final_labels_sha256=digest(final_labels), donors_sha256=digest(donor),
        eigen_roundoff_repaired_components=eigen_repaired,
        minimum_relative_eigenvalue_before_repair=minimum_relative_eigenvalue,
        eigen_floor='nonpositive_only:64*float64_eps*largest_eigenvalue_per_component',
        block_count=k, max_representatives_per_block=100,
        capped_blocks=sum(not r['converged'] for r in records),
        total_empty_repairs=sum(r['empty_repairs'] for r in records),
        mean_iterations=float(np.mean([r['iterations'] for r in records])),
        moved_fraction=float(np.mean(initial_labels != final_labels)),
        cluster_size_quantiles=np.quantile(np.bincount(final_labels, minlength=m), [0,.5,.9,1]).tolist(),
        appearance='own_final_cluster_full_GHAP_cost_argmin',
        initialization='exact_XYZ_fixed_groups_weighted_mean', covariance_scatter=False,
        cost='GHAP_TensorKL_float64', stopping='unchanged_labels_or_relative_objective_1e-12_or_100_updates',
        empty_policy='split_highest_weighted_residual_from_non_singleton',
        finetuning_iterations=0)
    print('BLOCK_LLOYD_COMPLETE', {k:v for k,v in record.items() if k != 'block_records'}, flush=True)
    return tensors, record
