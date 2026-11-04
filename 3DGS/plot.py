import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.reporting import plots,panels

if __name__=="__main__":
    p=argparse.ArgumentParser(description="Quality/runtime/memory plots and matched-camera panels")
    p.add_argument("--results-dir",required=True,type=Path)
    p.add_argument("--scene"); p.add_argument("--retention",type=float,default=.1)
    p.add_argument("--methods",nargs="+",default=["identity","ghap","hmw-local","mw2","min-B1D"])
    p.add_argument("--phase",choices=["pre","post"],default="post")
    p.add_argument("--views",type=int,nargs="+",default=[0])
    p.add_argument("--crop",type=int,nargs=4,metavar=("LEFT","TOP","RIGHT","BOTTOM"))
    args=p.parse_args(); plots(args.results_dir)
    if args.scene:
        panels(args.results_dir,args.scene,args.retention,args.methods,args.phase,args.views,args.crop)
