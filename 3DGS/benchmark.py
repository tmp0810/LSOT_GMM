"""Benchmark one growing OT problem on nested subsets of scene Gaussians."""
import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.common import APP,METHODS
from lsot3dgs.config import load_config
from lsot3dgs.scaling import run_scaling


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=APP/"configs/scaling.json")
    p.add_argument("--output-dir"); p.add_argument("--device")
    p.add_argument("--scenes",nargs="+"); p.add_argument("--methods",nargs="+",choices=METHODS)
    p.add_argument("--sizes",type=int,nargs="+"); p.add_argument("--retentions",type=float,nargs="+")
    p.add_argument("--timeout-seconds",type=int); p.add_argument("--no-resume",action="store_true")
    args=p.parse_args(); config=load_config(args.config)
    for name in ("output_dir","methods","sizes","retentions","timeout_seconds"):
        value=getattr(args,name)
        if value is not None: config[name]=value
    if args.device: config["solver"]["device"]=args.device
    if args.scenes: config["scenes"]={s:config["scenes"][s] for s in args.scenes}
    if any(m not in METHODS for m in config["methods"]):
        p.error("Solver scaling supports MW2 and LSOT only, not reduction baselines")
    run_scaling(config,resume=not args.no_resume)


if __name__=="__main__": main()
