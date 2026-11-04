"""A shared appearance rule for all controlled transport solvers."""
import torch


def ghap_pairs(source,target,i,j):
    return ((source.means[i]-target.means[j]).square().sum(-1)
            +(source.covariances[i]-target.covariances[j]).square().sum((-2,-1)))


def transfer(scene, ids, source, target, plan, *, rule="incoming", chunk=4096):
    device = source.means.device
    local = scene.subset(ids)
    sh = local.sh.to(device=device,dtype=source.means.dtype)
    opacity = local.opacity_logits.to(device=device,dtype=source.means.dtype)
    m = target.count
    if rule == "mean":
        new_sh = sh.new_zeros((m,*sh.shape[1:]))
        new_opacity = opacity.new_zeros((m,1))
        for i,j,p in plan.edges(chunk):
            new_sh.index_add_(0,j,p[:,None,None]*sh[i])
            new_opacity.index_add_(0,j,p[:,None]*opacity[i].sigmoid())
        new_sh /= target.weights[:,None,None]
        activated = (new_opacity/target.weights[:,None]).clamp(1e-12,1-1e-12)
        return new_sh,torch.logit(activated),{"rule":"coupling_mean_activated_opacity"}
    if rule not in ("incoming","full-block"):
        raise ValueError(f"Unknown appearance rule {rule}")

    def edges():
        if rule == "incoming":
            for i,j,p in plan.edges(chunk):
                yield i,j
        else:
            for k in range(0,source.count*m,chunk):
                indices = torch.arange(k,min(k+chunk,source.count*m),device=device)
                yield indices//m,indices%m

    best = source.weights.new_full((m,),float("inf"))
    for i,j in edges():
        best.scatter_reduce_(0,j,ghap_pairs(source,target,i,j),reduce="amin",include_self=True)
    original_ids = torch.as_tensor(ids,device=device)
    donor_ids = original_ids.new_full((m,),torch.iinfo(torch.int64).max)
    for i,j in edges():
        keep = ghap_pairs(source,target,i,j) == best[j]
        donor_ids.scatter_reduce_(0,j[keep],original_ids[i[keep]],reduce="amin",include_self=True)
    if bool((donor_ids == torch.iinfo(torch.int64).max).any()):
        raise RuntimeError("Target has no appearance donor")
    # Original indices are stable across different local orderings and tie cases.
    return (scene.sh[donor_ids.cpu()].to(device=device,dtype=sh.dtype),
            scene.opacity_logits[donor_ids.cpu()].to(device=device,dtype=opacity.dtype),
            {"rule":f"{rule}_GHAP_cost_donor","donor_original_ids":donor_ids.cpu().tolist()})
