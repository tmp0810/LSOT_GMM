"""Check CUDA linear algebra, all nine solvers, HCP and optional renderer backward."""
import _bootstrap
import argparse
from pathlib import Path
import torch

from lsot3dgs.common import APP,environment,resolve_path,select_device,write_json
from lsot3dgs.config import load_config,job_config
from lsot3dgs.geometry import GaussianSet,bures_update
from lsot3dgs.solvers import make_solver
from lsot3dgs.partition import hilbert_order
from lsot3dgs.pipeline import worker


def numerics(device):
    select_device(device); torch.set_num_threads(4)
    generator=torch.Generator().manual_seed(19)
    def sample(n):
        mean=torch.randn(n,3,generator=generator,dtype=torch.float64)
        factor=torch.randn(n,3,3,generator=generator,dtype=torch.float64)
        cov=factor@factor.transpose(-1,-2)+.1*torch.eye(3,dtype=torch.float64)
        weights=torch.rand(n,generator=generator,dtype=torch.float64)+.1
        return GaussianSet(*(t.to(device) for t in (weights/weights.sum(),mean,cov)))
    x,y=sample(24),sample(6)
    record={}
    options={"projections":8,"projection_batch":4}
    exact=make_solver("mw2",x,options,{}).solve(y)
    record["mw2"]={"objective":exact.objective,**exact.diagnostics}
    for kind in ("Mix","SMix","B","B1D"):
        values=[]
        for mode in ("min","avg"):
            result=make_solver(f"{mode}-{kind}",x,options,{}).solve(y)
            target,inner=bures_update(x,y,result.plan)
            record[f"{mode}-{kind}"]={"objective":result.objective,"bures":inner,
                                     "marginal_l1":result.plan.marginal_error(x.weights,y.weights)}
            values.append(result.objective)
        if not exact.objective<=values[0]+1e-8 or not values[0]<=values[1]+1e-8:
            raise RuntimeError("MW2 <= min-LSOT <= avg-LSOT failed")
    # Exercise QR as well as the eigensolver before attempting a real scene.
    torch.linalg.qr(x.covariances[:4])
    return {"status":"passed","environment":environment(device),"solvers":record}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=APP/"configs/hmw_protocol.json")
    p.add_argument("--device"); p.add_argument("--renderer",action="store_true")
    p.add_argument("--scene",default="train")
    p.add_argument("--skip-hcp",action="store_true")
    args=p.parse_args(); config=load_config(args.config)
    if args.device: config["solver"]["device"]=args.device
    result=numerics(config["solver"]["device"])
    if config["solver"]["partition"]=="hilbert" and not args.skip_hcp:
        import numpy as np
        order=hilbert_order(np.random.default_rng(0).normal(size=(100,3)),config["solver"]["hcp_extension"])
        result["hcp"]="passed" if len(set(order.tolist()))==100 else "failed"
    dest=resolve_path(config["output_dir"])/"preflight"
    write_json(dest/"numerics.json",result)
    if args.renderer:
        job=job_config(config,args.scene,"identity",1.)
        folder=dest/args.scene; write_json(folder/"job.json",job)
        worker("compress",folder/"job.json",folder,["--warmups","0","--repeats","1"])
        worker("renderer-smoke",folder/"job.json",folder/"pre")
    print(f"Preflight passed; reports in {dest}")


if __name__=="__main__": main()
