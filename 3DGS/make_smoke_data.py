"""Create a deterministic tiny numerical fixture, not a 3DGS quality dataset."""
import _bootstrap
import argparse
from pathlib import Path
import torch
from lsot3dgs.common import ROOT
from lsot3dgs.model_io import SceneGaussians,save_ply

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"results/3dgs_smoke_input/source.ply")
    p.add_argument("--count",type=int,default=40)
    a=p.parse_args(); g=torch.Generator().manual_seed(23); n=a.count
    scene=SceneGaussians(torch.randn(n,3,generator=g),torch.randn(n,3,generator=g)*.2-2,
        torch.randn(n,4,generator=g),torch.randn(n,1,generator=g),torch.randn(n,16,3,generator=g)*.1)
    save_ply(a.output,scene)
    print(a.output)
