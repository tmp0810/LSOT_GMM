"""Run the matched 3DGS compression / render / fine-tune experiment."""
import _bootstrap
import argparse
from pathlib import Path

from lsot3dgs.common import APP,METHODS,BASELINES
from lsot3dgs.config import load_config
from lsot3dgs.pipeline import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=APP/"configs/hmw_protocol.json")
    p.add_argument("--output-dir")
    p.add_argument("--device")
    p.add_argument("--scenes",nargs="+")
    p.add_argument("--methods",nargs="+",choices=METHODS+BASELINES)
    p.add_argument("--retentions",type=float,nargs="+")
    p.add_argument("--block-cap",type=int)
    p.add_argument("--projections",type=int)
    p.add_argument("--outer-steps",type=int)
    p.add_argument("--warmups",type=int)
    p.add_argument("--repeats",type=int)
    p.add_argument("--compression-only",action="store_true")
    p.add_argument("--no-finetune",action="store_true")
    p.add_argument("--no-resume",action="store_true")
    p.add_argument("--continue-on-error",action="store_true")
    args=p.parse_args(); config=load_config(args.config)
    if args.output_dir: config["output_dir"]=args.output_dir
    if args.scenes: config["scenes"]={s:config["scenes"][s] for s in args.scenes}
    if args.methods: config["methods"]=args.methods
    if args.retentions: config["retentions"]=args.retentions
    for key in ("device","block_cap","projections","outer_steps"):
        value=getattr(args,key)
        if value is not None: config["solver"][key]=value
    for key in ("warmups","repeats"):
        value=getattr(args,key)
        if value is not None: config["timing"][key]=value
    if args.no_finetune or args.compression_only: config["finetune"]["enabled"]=False
    if args.compression_only: config["evaluation"]["enabled"]=False
    run(config,resume=not args.no_resume,continue_on_error=args.continue_on_error)


if __name__=="__main__": main()
