"""One fresh-process compression measurement, with input/output audits."""
import json
from pathlib import Path
import time
import torch

from .common import environment,memory,seed_all,sha256,synchronize,write_json,select_device
from .model_io import load_scene,save_ply
from .reduction import compress_controlled
from .baselines import ghap_native,hmw_local


def compress_job(job,folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    options=job["solver"]
    device=options["device"]
    if torch.device(device).type=="cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu for numerical smoke only")
        select_device(device)
    seed_all(options.get("compression_seed",42))
    start=time.perf_counter()
    scene=load_scene(job["scene"]["checkpoint"],spatial_lr_scale=job["scene"].get("spatial_lr_scale"))
    load_seconds=time.perf_counter()-start
    if torch.device(device).type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device); start=time.perf_counter()
    method=job["method"]
    with (folder/"history.jsonl").open("w") as history:
        if method=="identity":
            result=scene
            record={"method":method,"source_count":scene.count,"actual_count":scene.count,
                    "requested_count":scene.count,"retention_requested":1.,"timings":{}}
        elif method=="ghap":
            result,record=ghap_native(scene,job["retention"],options)
        elif method=="hmw-local":
            result,record=hmw_local(scene,job["retention"],options)
        else:
            result,record=compress_controlled(scene,method,job["retention"],options,history=history)
    synchronize(device)
    record["compression_seconds"]=time.perf_counter()-start
    record["checkpoint_load_seconds"]=load_seconds
    record.update(memory(device))
    path=folder/"point_cloud/iteration_1/point_cloud.ply"
    start=time.perf_counter(); save_ply(path,result)
    restored=load_scene(path)
    if restored.count!=result.count:
        raise RuntimeError("PLY round-trip count mismatch")
    record["ply_export_and_audit_seconds"]=time.perf_counter()-start
    record["total_with_load_export_seconds"]=(load_seconds+record["compression_seconds"]
                                                +record["ply_export_and_audit_seconds"])
    record["actual_retention"]=result.count/scene.count
    record["ply_sha256"]=sha256(path)
    record["spatial_lr_scale"]=scene.spatial_lr_scale
    record["source_iteration"]=scene.iteration
    record["environment"]=environment(device)
    write_json(folder/"compression.json",record)
    write_json(folder/"job.json",job)
    write_json(folder/"complete.json",{"ply_sha256":record["ply_sha256"],"actual_count":result.count})
    return record
