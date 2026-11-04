"""Validate scene/checkpoint inputs and write reproducibility manifests."""
import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.common import APP,resolve_path,write_json
from lsot3dgs.config import load_config,job_config
from lsot3dgs.data import prepare_scene

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=APP/"configs/hmw_protocol.json")
    p.add_argument("--scenes",nargs="+")
    p.add_argument("--compression-only",action="store_true")
    args=p.parse_args(); config=load_config(args.config)
    for name in args.scenes or config["scenes"]:
        job=job_config(config,name,"identity",1.)
        record=prepare_scene(job["scene"],require_dataset=not args.compression_only)
        dest=resolve_path(config["output_dir"])/"inputs"/f"{name}.json"
        write_json(dest,record)
        print(f"{name}: {record['gaussian_count']} Gaussians, manifest: {dest}")
