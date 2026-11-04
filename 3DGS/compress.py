"""Single method convenience entry point; defaults to compression only."""
import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.common import APP,METHODS,BASELINES,resolve_path,write_json
from lsot3dgs.config import load_config,job_config
from lsot3dgs.pipeline import worker

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=APP/"configs/hmw_protocol.json")
    p.add_argument("--scene",required=True); p.add_argument("--method",required=True,choices=METHODS+BASELINES)
    p.add_argument("--retention",type=float,default=.1); p.add_argument("--output-dir",required=True)
    p.add_argument("--device"); a=p.parse_args(); config=load_config(a.config)
    if a.device: config["solver"]["device"]=a.device
    job=job_config(config,a.scene,a.method,a.retention)
    folder=resolve_path(a.output_dir)
    if (folder/"compression.json").exists(): p.error("Output exists; choose a new output directory")
    write_json(folder/"job.json",job)
    worker("compress",folder/"job.json",folder,["--warmups",str(config["timing"]["warmups"]),
                                                "--repeats",str(config["timing"]["repeats"])])
