import _bootstrap
import argparse
from pathlib import Path
from lsot3dgs.pipeline import worker

if __name__=="__main__":
    p=argparse.ArgumentParser(description="Fine-tune a compressed PLY using the shared GHAP loop")
    p.add_argument("--job",required=True,type=Path); p.add_argument("--pre-dir",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args()
    worker("finetune",a.job.resolve(),a.output_dir.resolve(),["--pre-dir",str(a.pre_dir.resolve())])
