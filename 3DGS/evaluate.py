import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.pipeline import worker

if __name__=="__main__":
    p=argparse.ArgumentParser(description="Render/evaluate an existing experiment phase")
    p.add_argument("--job",required=True,type=Path); p.add_argument("--model-dir",required=True,type=Path)
    p.add_argument("--iteration",required=True,type=int)
    a=p.parse_args()
    worker("evaluate",a.job.resolve(),a.model_dir.resolve(),["--iteration",str(a.iteration)])
