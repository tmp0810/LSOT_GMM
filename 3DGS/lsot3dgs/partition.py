"""Shared XYZ order, exact output allocation and fixed-capacity initialization."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import numpy as np
import torch

from .common import ROOT, resolve_path, tensor_digest
from .geometry import GaussianSet, repair_spd


def hilbert_order(points, extension):
    source = ROOT / "HMW_code/src/hmw/hilbert.py"
    spec = importlib.util.spec_from_file_location("lsot3dgs_official_hcp", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OfficialHCP(resolve_path(extension)).order(points)


def kd_order(points, leaf_size=32):
    """Deterministic cyclic-axis median order, explicitly an HCP ablation."""
    def visit(ids, depth):
        if len(ids) <= leaf_size:
            return ids
        ids = ids[np.argsort(points[ids,depth % 3],kind="stable")]
        cut = len(ids)//2
        return np.concatenate((visit(ids[:cut],depth+1),visit(ids[cut:],depth+1)))
    return visit(np.arange(len(points)),0)


@dataclass
class Partition:
    order: np.ndarray  # original scene indices, with zero masses removed
    cuts: np.ndarray
    rep_cuts: np.ndarray
    original_count: int
    target_count: int
    kind: str

    def blocks(self):
        for block,(left,right) in enumerate(zip(self.rep_cuts[:-1],self.rep_cuts[1:])):
            begin,end = self.cuts[left],self.cuts[right]
            yield block,self.order[begin:end],self.cuts[left:right+1]-begin

    def metadata(self):
        return {"source_original":self.original_count,"source_positive":len(self.order),
                "zero_opacity_removed":self.original_count-len(self.order),"target":self.target_count,
                "blocks":len(self.rep_cuts)-1,"kind":self.kind,
                "order_sha256":tensor_digest(self.order),
                "partition_sha256":tensor_digest(self.order,self.cuts,self.rep_cuts)}


def make_partition(scene, target_count, options):
    if not 1 <= target_count <= scene.count:
        raise ValueError("Target count must be between 1 and source count")
    positive = torch.where(scene.opacity_logits.double().sigmoid().flatten()>0)[0].numpy()
    if target_count > len(positive):
        raise ValueError("Output budget exceeds the number of positive-opacity sources")
    points = scene.xyz[positive].double().numpy()
    kind = options.get("partition","hilbert")
    if kind == "hilbert":
        order = hilbert_order(points,options.get("hcp_extension","HMW_code/build/hcp/base.so"))
    elif kind == "kd":
        order = kd_order(points)
    elif kind == "input":
        order = np.arange(len(points))  # Small numerical tests only, recorded explicitly.
    else:
        raise ValueError(f"Unknown partition {kind}")
    cap = int(options.get("block_cap",100))
    if cap < 1:
        raise ValueError("block_cap must be positive")
    blocks = (target_count+cap-1)//cap
    cuts = np.arange(target_count+1,dtype=np.int64)*len(positive)//target_count
    rep_cuts = np.arange(blocks+1,dtype=np.int64)*target_count//blocks
    return Partition(positive[order],cuts,rep_cuts,scene.count,target_count,kind)


def initialize(source, cuts):
    lengths = torch.as_tensor(np.diff(cuts),device=source.means.device)
    m = len(lengths)
    labels = torch.repeat_interleave(torch.arange(m,device=source.means.device),lengths)
    b = source.weights.new_zeros(m).index_add_(0,labels,source.weights)
    means = source.means.new_zeros((m,3)).index_add_(0,labels,source.weights[:,None]*source.means)
    cov = source.covariances.new_zeros((m,3,3)).index_add_(0,labels,
                                                       source.weights[:,None,None]*source.covariances)
    means /= b[:,None]
    cov,repairs = repair_spd(cov/b[:,None,None])
    target = GaussianSet(b/b.sum(),means,cov)
    return target,{"capacity_sha256":tensor_digest(target.weights),
                   "initialization_sha256":tensor_digest(target.weights,means,cov),
                   "initialization_repairs":repairs}
