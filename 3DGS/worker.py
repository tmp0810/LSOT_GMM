"""Internal fresh-process entry point. User entry points are documented in README."""
import _bootstrap
import argparse
import json
from pathlib import Path
import traceback
import torch

from lsot3dgs.common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["compress","evaluate","finetune","scaling","renderer-smoke","backbone"])
    p.add_argument("--job",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--pre-dir",type=Path)
    p.add_argument("--iteration",type=int,default=1)
    p.add_argument("--warmups",type=int,default=1)
    p.add_argument("--repeats",type=int,default=3)
    args=p.parse_args(); job=json.loads(args.job.read_text())
    torch.set_num_threads(job.get("solver",{}).get("cpu_threads",4))
    try:
        if args.command=="compress":
            from lsot3dgs.pipeline import measure_compression
            measure_compression(job,args.output_dir,args.warmups,args.repeats)
        elif args.command=="scaling":
            from lsot3dgs.scaling import scaling_case
            scaling_case(job,args.output_dir)
        else:
            from lsot3dgs.ghap_backend import evaluate_job,finetune_job,renderer_smoke,train_backbone
            if args.command=="evaluate": evaluate_job(job,args.output_dir,args.iteration)
            elif args.command=="finetune": finetune_job(job,args.pre_dir,args.output_dir)
            elif args.command=="backbone": train_backbone(job,args.output_dir)
            else: write_json(args.output_dir/"renderer_smoke.json",renderer_smoke(job,args.output_dir,args.iteration))
    except Exception as exc:
        traceback.print_exc()
        if args.command=="scaling":
            from lsot3dgs.solvers.mw2 import BudgetExceeded
            status="skipped_budget" if isinstance(exc,BudgetExceeded) else (
                "out_of_memory" if isinstance(exc,(torch.OutOfMemoryError,MemoryError)) else "failed")
            write_json(args.output_dir/"worker_result.json",{"status":status,"case":job,
                       "error_type":type(exc).__name__,"message":str(exc)})
        raise


if __name__=="__main__": main()
