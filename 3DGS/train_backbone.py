"""Explicitly train a new backbone when the historical checkpoint is unavailable."""
import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.common import APP,resolve_path,write_json
from lsot3dgs.config import load_config,job_config
from lsot3dgs.pipeline import worker

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=APP/"configs/hmw_protocol.json")
    p.add_argument("--scene",required=True); p.add_argument("--output-dir",required=True)
    args=p.parse_args(); config=load_config(args.config)
    job=job_config(config,args.scene,"identity",1.)
    folder=resolve_path(args.output_dir)
    if folder.exists() and any(folder.glob("chkpnt*.pth")):
        p.error("Output already contains a backbone; choose a new folder")
    write_json(folder/"job.json",job)
    worker("backbone",folder/"job.json",folder)
