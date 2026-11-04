"""Batch independent, equally shaped GHAP-cost Lloyd blocks in float64."""
import torch


def costs(mu, cov, centers, shapes):
    # Identical direct differences and norms to upstream TensorKL; avoids
    # cancellation from ||x||^2+||y||^2-2<x,y> on thin scene components.
    return (torch.linalg.norm(mu[:, :, None]-centers[:, None], dim=-1)**2
            + torch.linalg.matrix_norm(cov[:, :, None]-shapes[:, None], ord='fro')**2)


def update(mu, cov, a, labels, m):
    b, n = a.shape
    mass = torch.zeros((b, m), dtype=a.dtype, device=a.device)
    mass.scatter_add_(1, labels, a)
    centers = torch.zeros((b, m, 3), dtype=mu.dtype, device=mu.device)
    shapes = torch.zeros((b, m, 9), dtype=cov.dtype, device=cov.device)
    centers.scatter_add_(1, labels[..., None].expand(-1,-1,3), a[..., None]*mu)
    shapes.scatter_add_(1, labels[..., None].expand(-1,-1,9), a[..., None]*cov.flatten(2))
    return centers/mass[..., None], (shapes/mass[..., None]).reshape(b,m,3,3)


def solve_batch(mu, cov, a, initial_labels, m, max_iter=100):
    a = a/a.sum(dim=1, keepdim=True)
    labels = initial_labels.clone()
    b, n = a.shape
    device = a.device
    centers, shapes = update(mu, cov, a, labels, m)
    cost = costs(mu, cov, centers, shapes)
    previous = (a*cost.gather(2, labels[..., None]).squeeze(2)).sum(1)
    initial = previous.clone()
    active = torch.ones(b, dtype=torch.bool, device=device)
    iterations = torch.zeros(b, dtype=torch.long, device=device)
    repairs = iterations.clone()
    bids = torch.arange(b, device=device)
    ones = torch.ones_like(labels)
    for iteration in range(1, max_iter+1):
        distances, candidate = cost.min(dim=2)
        candidate = torch.where(active[:, None], candidate, labels)
        counts = torch.zeros((b,m), dtype=torch.long, device=device).scatter_add_(1,candidate,ones)
        empty = counts == 0
        # One lowest-ID empty cluster per active block per pass. This preserves
        # the serial repair ordering and the original weighted residual ranking.
        while bool(empty.any()):
            needs = empty.any(1)
            destination = empty.long().argmax(1)
            eligible = counts.gather(1,candidate) > 1
            priority = torch.where(eligible, a*distances, -torch.inf)
            chosen = priority.argmax(1)
            rows = bids[needs]
            selected = chosen[needs]
            origin = candidate[rows, selected].clone()
            counts[rows,origin] -= 1
            candidate[rows,selected] = destination[needs]
            counts[rows,destination[needs]] += 1
            repairs += needs.long()
            empty = counts == 0
        stable = (candidate == labels).all(1)
        labels = candidate
        new_centers, new_shapes = update(mu,cov,a,labels,m)
        centers = torch.where(active[:,None,None],new_centers,centers)
        shapes = torch.where(active[:,None,None,None],new_shapes,shapes)
        cost = costs(mu,cov,centers,shapes)
        obj = (a*cost.gather(2,labels[...,None]).squeeze(2)).sum(1)
        if bool((active & (obj > previous+1e-10*previous.abs().clamp_min(1))).any()):
            raise AssertionError('Batched Lloyd objective increased')
        done = stable | ((previous-obj).abs() <= 1e-12*initial.abs().clamp_min(1e-12))
        iterations += active.long()
        active &= ~done
        previous = obj
        if not bool(active.any()):
            break
    mask = labels[...,None] != torch.arange(m,device=device)[None,None,:]
    donor = cost.masked_fill(mask,torch.inf).argmin(1)
    assert torch.equal(labels.gather(1,donor),torch.arange(m,device=device)[None,:].expand(b,-1))
    counts = torch.zeros((b,m),dtype=torch.long,device=device).scatter_add_(1,labels,ones)
    summary = torch.stack([iterations.double(), (~active).double(), repairs.double(),
        initial,previous,(labels!=initial_labels).double().mean(1),
        counts.min(1).values.double(),counts.max(1).values.double()],1).cpu().numpy()
    keys = ['iterations','converged','empty_repairs','initial_objective','final_objective',
            'moved_fraction','min_cluster_size','max_cluster_size']
    records = [dict(zip(keys,map(float,row))) for row in summary]
    for rec in records:
        for key in ['iterations','empty_repairs','min_cluster_size','max_cluster_size']:
            rec[key] = int(rec[key])
        rec['converged'] = bool(rec['converged'])
    return centers,shapes,donor,labels,records
