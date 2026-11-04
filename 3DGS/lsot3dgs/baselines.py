"""Adapters to reviewed native baseline code; no transport-metric relabeling."""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
import torch

from .common import ROOT,check_ghap,timed
from .geometry import GaussianSet
from .model_io import SceneGaussians,geometry_to_scene
from .partition import make_partition


def import_hmw_solver():
    path=ROOT/"HMW_code/experiments/scene_compression/original/hcp_batched_solver.py"
    spec=importlib.util.spec_from_file_location("lsot3dgs_native_hmw_lloyd",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.solve_batch


@torch.no_grad()
def hmw_local(scene,retention,options):
    if options.get("block_cap",100)!=100:
        raise ValueError("Native HMW-local baseline requires block_cap=100")
    device=options.get("device","cuda")
    requested=max(1,math.floor(scene.count*retention))
    times={}
    with timed(times,"partition_seconds"):
        partition=make_partition(scene,requested,options)
    if len(partition.order)!=scene.count:
        raise ValueError("Native HMW baseline requires positive source opacity; do not silently change its input")
    solve=import_hmw_solver()
    groups={}
    for block,ids,cuts in partition.blocks():
        groups.setdefault((len(ids),len(cuts)-1),[]).append((block,ids,cuts))
    outputs={}; records=[]
    batch_size=options.get("hmw_batch_size") or (2 if retention<=.001 else 16)
    for (_,m),entries in groups.items():
        for k in range(0,len(entries),batch_size):
            current=entries[k:k+batch_size]
            with timed(times,"source_extraction_seconds",device):
                sources=[scene.gaussian_set(ids,device=device)[0] for _,ids,_ in current]
                mu=torch.stack([s.means for s in sources]); cov=torch.stack([s.covariances for s in sources])
                weights=torch.stack([s.weights for s in sources])
                labels=torch.stack([torch.repeat_interleave(torch.arange(m,device=device),
                    torch.as_tensor(np.diff(cuts),device=device)) for _,_,cuts in current])
            with timed(times,"native_lloyd_and_donor_seconds",device):
                centers,shapes,donors,_,details=solve(mu,cov,weights,labels,m)
            with timed(times,"export_conversion_seconds",device):
                for q,(block,ids,_) in enumerate(current):
                    donor_ids=np.asarray(ids)[donors[q].cpu().numpy()]
                    target=GaussianSet(weights.new_full((m,),1/m),centers[q],shapes[q])
                    output,export=geometry_to_scene(target,scene.sh[donor_ids].to(device),
                        scene.opacity_logits[donor_ids].to(device),spatial_lr_scale=scene.spatial_lr_scale)
                    outputs[block]=output
                    records.append({"block":block,**details[q],**export})
    result=SceneGaussians(*(torch.cat([outputs[k].tensors()[j] for k in sorted(outputs)])
                           for j in range(5)),scene.spatial_lr_scale)
    return result,{"method":"hmw-local","source_count":scene.count,"requested_count":requested,
        "actual_count":result.count,"retention_requested":retention,"partition":partition.metadata(),
        "blocks":records,"timings":times,"batch_size":batch_size,
        "cost":"squared mean L2 + squared covariance Frobenius; not Gaussian W2",
        "all_blocks_converged":all(r["converged"] for r in records)}


@torch.no_grad()
def ghap_native(scene,retention,options):
    if torch.device(options.get("device","cuda")).type!="cuda":
        raise ValueError("Native GHAP baseline requires its CUDA environment")
    requested=max(1,math.floor(scene.count*retention))
    if requested<min(scene.count,1024):
        raise ValueError("Native depth-10 GHAP would allocate zero targets to a KD leaf. "
                         "This budget is unsupported without changing the baseline partition.")
    repo,version=check_ghap(options["ghap_root"],options.get("allow_modified_ghap",False))
    sys.path.insert(0,str(repo))
    from scene.gaussian_model import GaussianModel
    from gmm_sampler import gaussian_model_reduction
    model=GaussianModel(scene.degree)
    for name,value in (("_xyz",scene.xyz),("_scaling",scene.log_scales),("_rotation",scene.rotations),
                       ("_opacity",scene.opacity_logits),("_features_dc",scene.sh[:,:1]),
                       ("_features_rest",scene.sh[:,1:])):
        setattr(model,name,torch.nn.Parameter(value.to(options["device"]).float(),requires_grad=False))
    times={}
    with timed(times,"native_ghap_seconds",options["device"]):
        reduced=gaussian_model_reduction(model,retention,random_seed=options.get("compression_seed",42))
    result=SceneGaussians(*(v.detach().cpu() for v in
        (reduced._xyz,reduced._scaling,reduced._rotation,reduced._opacity,
         torch.cat((reduced._features_dc,reduced._features_rest),1))),scene.spatial_lr_scale)
    return result,{"method":"ghap","source_count":scene.count,
        "requested_count":max(1,math.floor(scene.count*retention)),"actual_count":result.count,
        "retention_requested":retention,"timings":times,"upstream":version,
        "count_policy":"native empty-column deletion; no padding",
        "cost":"native GHAP TensorKL; not Gaussian W2"}
