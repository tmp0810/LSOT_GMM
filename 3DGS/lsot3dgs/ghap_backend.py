"""Pinned GHAP renderer, metric functions and unchanged training loop.

Imports happen only in dedicated subprocesses. No legacy HMW import chain is needed.
"""
from argparse import ArgumentParser,Namespace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .common import check_ghap,digest,seed_all,sha256,synchronize,write_json,select_device
from .model_io import load_scene


def activate(job):
    root,version=check_ghap(job["ghap_root"],job.get("allow_modified_ghap",False))
    sys.path.insert(0,str(root))
    if not torch.cuda.is_available():
        raise RuntimeError("GHAP renderer and fine-tuning require CUDA")
    select_device(job["solver"].get("device","cuda"))
    return root,version


def arguments(job,folder):
    from arguments import ModelParams,PipelineParams,OptimizationParams
    parser=ArgumentParser()
    mp,pp,op=ModelParams(parser),PipelineParams(parser),OptimizationParams(parser)
    args=parser.parse_args([])
    scene=job["scene"]
    args.source_path=scene["dataset"]
    args.model_path=str(Path(folder).resolve())
    args.images=scene.get("images","images")
    args.depths=""
    args.resolution=scene.get("resolution",-1)
    args.sh_degree=scene.get("sh_degree",3)
    args.white_background=scene.get("white_background",False)
    args.train_test_exp=False
    args.data_device="cpu"
    args.eval=True
    args.optimizer_type="default"
    args.densify_until_iter=0
    args.iterations=job["finetune"].get("first_iteration",30000)+job["finetune"].get("iterations",5000)
    return mp.extract(args),pp.extract(args),op.extract(args)


def camera_manifest(scene):
    def describe(view):
        return {"image_name":view.image_name,"width":view.image_width,"height":view.image_height,
                "R":np.asarray(view.R).tolist(),"T":np.asarray(view.T).tolist(),
                "FoVx":float(view.FoVx),"FoVy":float(view.FoVy)}
    rows={"train":[describe(v) for v in scene.getTrainCameras()],
          "test":[describe(v) for v in scene.getTestCameras()]}
    if not rows["test"]:
        raise RuntimeError("No held-out cameras; check COLMAP dataset and --eval split")
    # Training Scene shuffles cameras, so canonical split identity is order-independent.
    canonical={k:sorted(v,key=lambda row:row["image_name"]) for k,v in rows.items()}
    return {"views":rows,"split_sha256":digest(canonical),"test_count":len(rows["test"])}


def evaluate_job(job,folder,iteration):
    root,version=activate(job)
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    from PIL import Image
    import torchvision
    from torchvision.transforms.functional import to_tensor
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch import lpips
    try:
        from diff_gaussian_rasterization import SparseGaussianAdam
        separate_sh=True
    except ImportError:
        separate_sh=False
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    dataset,pipeline,_=arguments(job,folder)
    (folder/"cfg_args").write_text(str(Namespace(**vars(dataset))))
    gaussians=GaussianModel(dataset.sh_degree)
    scene=Scene(dataset,gaussians,load_iteration=iteration,shuffle=False)
    manifest=camera_manifest(scene)
    write_json(folder/"camera_manifest.json",manifest)
    name=folder.name
    render_dir,gt_dir=folder/"test"/name/"renders",folder/"test"/name/"gt"
    render_dir.mkdir(parents=True,exist_ok=True); gt_dir.mkdir(parents=True,exist_ok=True)
    background=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device="cuda")
    per_view={key:{} for key in ("SSIM","PSNR","LPIPS")}
    render_ms=[]; image_hashes={}
    start=time.perf_counter()
    with torch.no_grad():
        if job["evaluation"].get("render_fps",False):
            for view in scene.getTestCameras()[:min(10,len(scene.getTestCameras()))]:
                render(view,gaussians,pipeline,background,use_trained_exp=False,separate_sh=separate_sh)
            torch.cuda.synchronize()
        for index,view in enumerate(scene.getTestCameras()):
            if job["evaluation"].get("render_fps",False):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
            rendering=render(view,gaussians,pipeline,background,use_trained_exp=False,
                             separate_sh=separate_sh)["render"]
            if job["evaluation"].get("render_fps",False):
                end.record(); end.synchronize(); render_ms.append(begin.elapsed_time(end))
            filename=f"{index:05d}.png"
            torchvision.utils.save_image(rendering,render_dir/filename)
            torchvision.utils.save_image(view.original_image[:3],gt_dir/filename)
            image_hashes[filename]={"render":sha256(render_dir/filename),"gt":sha256(gt_dir/filename)}
            # Evaluate saved, quantized PNGs, exactly as the upstream metrics script.
            # Load one view at a time to avoid retaining every image on the GPU.
            with Image.open(render_dir/filename) as im:
                prediction=to_tensor(im).unsqueeze(0)[:,:3].cuda()
            with Image.open(gt_dir/filename) as im:
                truth=to_tensor(im).unsqueeze(0)[:,:3].cuda()
            values={"SSIM":float(ssim(prediction,truth)),"PSNR":float(psnr(prediction,truth).mean()),
                    "LPIPS":float(lpips(prediction,truth,net_type="vgg"))}
            if not all(np.isfinite(x) for x in values.values()):
                raise RuntimeError(f"Nonfinite metric for {filename}: {values}")
            for key,value in values.items():
                per_view[key][filename]=value
    expected={f"{i:05d}.png" for i in range(manifest["test_count"])}
    for directory in (render_dir,gt_dir):
        if {p.name for p in directory.glob("*.png")}!=expected:
            raise RuntimeError(f"Missing or stale rendered views in {directory}")
    means={key:float(np.mean(list(value.values()))) for key,value in per_view.items()}
    write_json(folder/"results.json",{name:means})
    write_json(folder/"per_view.json",{name:per_view})
    details={"metrics":means,"test_count":manifest["test_count"],"split_sha256":manifest["split_sha256"],
             "iteration":iteration,"evaluation_seconds":time.perf_counter()-start,"upstream":version,
             "metric_protocol":"upstream PSNR/SSIM/LPIPS-VGG on saved PNGs, one view at a time",
             "render_fps":1000/float(np.mean(render_ms)) if render_ms else None,
             "render_mean_ms":float(np.mean(render_ms)) if render_ms else None,
             "gaussian_count":len(gaussians._xyz),"image_sha256":image_hashes}
    write_json(folder/"evaluation.json",details)
    return details


def renderer_smoke(job,folder,iteration=1):
    activate(job)
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    dataset,pipeline,_=arguments(job,folder)
    gaussians=GaussianModel(dataset.sh_degree)
    scene=Scene(dataset,gaussians,load_iteration=iteration,shuffle=False)
    view=scene.getTrainCameras()[0]
    bg=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device="cuda")
    output=render(view,gaussians,pipeline,bg)["render"]
    output.mean().backward()
    if not bool(torch.isfinite(output).all()) or gaussians._xyz.grad is None:
        raise RuntimeError("Renderer forward/backward smoke failed")
    if not bool(torch.isfinite(gaussians._xyz.grad).all()):
        raise RuntimeError("Renderer returned nonfinite gradients")
    return {"forward_backward":"passed","shape":list(output.shape),**camera_manifest(scene)}


def finetune_job(job,pre,post):
    root,version=activate(job)
    from scene.gaussian_model import GaussianModel
    import train_and_prune
    post=Path(post); post.mkdir(parents=True,exist_ok=True)
    source=Path(pre)/"point_cloud/iteration_1/point_cloud.ply"
    original=load_scene(job["scene"]["checkpoint"],spatial_lr_scale=job["scene"].get("spatial_lr_scale"))
    if original.spatial_lr_scale is None or original.spatial_lr_scale<=0:
        raise ValueError("Fine-tuning a PLY input requires scene.spatial_lr_scale from its source checkpoint")
    expected=load_scene(source).count
    dataset,pipeline,opt=arguments(job,post)
    first=job["finetune"].get("first_iteration",30000)
    loops=job["finetune"].get("iterations",5000)
    if loops<2:
        raise ValueError("The official final iteration skips the update; need at least two loop iterations")
    seed=job["finetune"].get("seed",0)
    seed_all(seed)
    model=GaussianModel(dataset.sh_degree)
    model.load_ply(str(source))
    model.spatial_lr_scale=original.spatial_lr_scale
    model.max_radii2D=torch.zeros(expected,device="cuda")
    model._exposure=torch.nn.Parameter(torch.eye(3,4,device="cuda")[None])
    model.training_setup(opt)
    if model.optimizer.state:
        raise AssertionError("Fine-tuning must start with fresh Adam state")
    checkpoint=post/"start.pth"
    torch.save((model.capture(),first),checkpoint)
    del model,original
    torch.cuda.empty_cache()
    start=time.perf_counter()
    train_and_prune.training(dataset,opt,pipeline,[opt.iterations],[opt.iterations],[opt.iterations],
                            str(checkpoint),-1,compaction=Namespace(flag=False))
    synchronize("cuda")
    final=post/f"point_cloud/iteration_{opt.iterations}/point_cloud.ply"
    if not final.is_file() or load_scene(final).count!=expected:
        raise RuntimeError("Fine-tuning failed or changed the Gaussian count")
    result={"source_ply_sha256":sha256(source),"final_ply_sha256":sha256(final),
            "gaussian_count":expected,"first_iteration":first,"last_iteration":opt.iterations,
            "loop_iterations":loops,"optimizer_updates":loops-1,"seed":seed,
            "optimizer":"fresh Adam","geometry_trainable":True,"appearance_trainable":True,
            "densification":False,"compaction":False,"seconds":time.perf_counter()-start,
            "optimization_parameters":vars(opt),"upstream":version}
    write_json(post/"finetune.json",result)
    return result


def train_backbone(job,folder):
    _,version=activate(job)
    import train_and_prune
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    dataset,pipeline,opt=arguments(job,folder)
    from arguments import OptimizationParams
    parser=ArgumentParser(); group=OptimizationParams(parser)
    default=group.extract(parser.parse_args([]))
    default.iterations=job["scene"].get("backbone_iterations",30000)
    seed_all(job["scene"].get("backbone_seed",0))
    start=time.perf_counter()
    train_and_prune.training(dataset,default,pipeline,[default.iterations],[default.iterations],
        [default.iterations],None,-1,compaction=Namespace(flag=False))
    record={"provenance":"newly_trained_backbone; not the unavailable historical HMW checkpoint",
            "iterations":default.iterations,"seconds":time.perf_counter()-start,"upstream":version}
    write_json(folder/"backbone_provenance.json",record)
    return record
