"""Shared block-cap or projection-budget ablations, each in its own output folder."""
import _bootstrap
import argparse
import copy
from pathlib import Path
from lsot3dgs.common import APP
from lsot3dgs.config import load_config
from lsot3dgs.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=APP/"configs/hmw_protocol.json")
    p.add_argument("--parameter",choices=["block_cap","projections"],required=True)
    p.add_argument("--values",type=int,nargs="+",required=True)
    p.add_argument("--scenes",nargs="+"); p.add_argument("--methods",nargs="+")
    a=p.parse_args(); base=load_config(a.config)
    if a.scenes: base["scenes"]={s:base["scenes"][s] for s in a.scenes}
    if a.methods: base["methods"]=a.methods
    if a.parameter=="block_cap" and "hmw-local" in base["methods"]:
        p.error("Native HMW-local has cap=100; choose controlled methods explicitly for this ablation")
    if a.parameter=="projections" and max(a.values)>base["solver"]["bank_size"]:
        p.error("Increase the common bank_size before the sweep so all budgets share a bank prefix")
    for value in a.values:
        c=copy.deepcopy(base); c["solver"][a.parameter]=value
        c["output_dir"]=str(Path(base["output_dir"])/f"{a.parameter}_{value}")
        run(c)
