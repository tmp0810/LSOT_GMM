"""Isolated experiment jobs, audited resume, repeated timing and pre/post evaluation."""
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

import torch

from .common import APP,checked_process,code_digest,digest,resolve_path,sha256,write_json,runtime_fingerprint
from .compression import compress_job
from .config import job_config,validate_config
from .data import prepare_scene


def measure_compression(job,folder,warmups,repeats):
    folder=Path(folder); measurements=[]
    if repeats<1 or warmups<0:
        raise ValueError("Invalid timing budget")
    for trial in range(-warmups,repeats):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage=folder/"timing"/(f"warmup_{-trial}" if trial<0 else f"repeat_{trial}")
        record=compress_job(job,stage)
        if trial>=0:
            measurements.append(record)
        ply=stage/"point_cloud/iteration_1/point_cloud.ply"
        if trial==0:
            pre=folder/"pre/point_cloud/iteration_1/point_cloud.ply"
            pre.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ply,pre)
            shutil.copy2(stage/"history.jsonl",folder/"history.jsonl")
        # These files were created by this call; retain records, not duplicate scene PLYs.
        shutil.rmtree(stage/"point_cloud")
    hashes={row["ply_sha256"] for row in measurements}
    result=copy.deepcopy(measurements[0])
    result["timing_policy"]={"warmups":warmups,"repeats":repeats,
        "cache":"rebuilt for every compression call; runtime warmed in this worker",
        "cpu_peak_scope":"fresh configuration worker lifetime, including warmups",
        "exported_repeat":0,"identical_output_hashes":len(hashes)==1}
    result["compression_seconds_samples"]=[r["compression_seconds"] for r in measurements]
    result["compression_seconds_median"]=statistics.median(result["compression_seconds_samples"])
    result["cpu_peak_rss_bytes"]=max(r["cpu_peak_rss_bytes"] for r in measurements)
    for field in ("gpu_peak_allocated_bytes","gpu_peak_reserved_bytes"):
        result[field]=max(r.get(field,0) for r in measurements)
    write_json(folder/"compression.json",result)
    return result


def validate_evaluation(folder,expected_count=None):
    folder=Path(folder)
    record=json.loads((folder/"evaluation.json").read_text())
    if expected_count is not None and record["gaussian_count"]!=expected_count:
        raise RuntimeError("Evaluation Gaussian count mismatch")
    per=json.loads((folder/"per_view.json").read_text())[folder.name]
    expected={f"{i:05d}.png" for i in range(record["test_count"])}
    import math
    for metric in ("PSNR","SSIM","LPIPS"):
        if set(per[metric])!=expected or not all(math.isfinite(v) for v in per[metric].values()):
            raise RuntimeError(f"Incomplete or nonfinite {metric} values in {folder}")
    for sub in ("renders","gt"):
        if {p.name for p in (folder/"test"/folder.name/sub).glob("*.png")}!=expected:
            raise RuntimeError("Incomplete rendered images")
        for filename in expected:
            key="render" if sub=="renders" else "gt"
            if sha256(folder/"test"/folder.name/sub/filename)!=record["image_sha256"][filename][key]:
                raise RuntimeError("Rendered image hash mismatch")
    return record


def complete_matches(folder,signature,job):
    folder=Path(folder)
    try:
        completion=json.loads((folder/"complete.json").read_text())
        if completion["signature"]!=signature:
            return False
        for rel,expected in completion["files"].items():
            if sha256(folder/rel)!=expected:
                return False
        comp=json.loads((folder/"compression.json").read_text())
        if job["evaluation"]["enabled"]:
            validate_evaluation(folder/"pre",comp["actual_count"])
            if job["finetune"]["enabled"] and job["method"]!="identity":
                validate_evaluation(folder/"post",comp["actual_count"])
        return True
    except (OSError,KeyError,ValueError,RuntimeError):
        return False


def worker(command,job_path,folder,extra=(),timeout=None):
    env=os.environ.copy()
    env["PYTHONUNBUFFERED"]="1"
    # GHAP's trusted local checkpoints use the old Python object capture format.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
    cmd=[sys.executable,APP/"worker.py",command,"--job",job_path,"--output-dir",folder,*extra]
    return checked_process(cmd,Path(folder)/f"{command}.log",cwd=APP.parent,timeout=timeout,env=env)


def finish_run(folder,signature,job):
    folder=Path(folder)
    required=["compression.json","history.jsonl","pre/point_cloud/iteration_1/point_cloud.ply"]
    if job["evaluation"]["enabled"]:
        required += [f"pre/{name}" for name in ("evaluation.json","results.json","per_view.json","camera_manifest.json")]
    if job["finetune"]["enabled"] and job["method"]!="identity":
        last=job["finetune"].get("first_iteration",30000)+job["finetune"].get("iterations",5000)
        required += ["post/finetune.json",f"post/point_cloud/iteration_{last}/point_cloud.ply"]
        if job["evaluation"]["enabled"]:
            required += [f"post/{name}" for name in ("evaluation.json","results.json","per_view.json","camera_manifest.json")]
    write_json(folder/"complete.json",{"signature":signature,"status":"complete",
                "files":{rel:sha256(folder/rel) for rel in required}})


def run(config,*,resume=True,continue_on_error=False):
    validate_config(config)
    output=resolve_path(config["output_dir"]); output.mkdir(parents=True,exist_ok=True)
    write_json(output/"resolved_config.json",config)
    code=code_digest()
    runtime=runtime_fingerprint(config)
    write_json(output/"environment.json",runtime)
    failures=[]; split_hashes={}
    for name in config["scenes"]:
        template=job_config(config,name,"identity",1.)
        needs_data=config["evaluation"]["enabled"] or config["finetune"]["enabled"]
        manifest=prepare_scene(template["scene"],require_dataset=needs_data)
        write_json(output/"inputs"/f"{name}.json",manifest)
        cases=[("identity",1.)] if config.get("include_identity",True) else []
        for rho in config["retentions"]:
            cases += [(method,rho) for method in config["methods"] if method!="identity"]
        for method,rho in cases:
            job=job_config(config,name,method,rho)
            folder=output/name/("original" if method=="identity" else f"rho_{rho:g}")/method
            signature=digest({"job":job,"input_sha256":manifest["input_sha256"],"code_sha256":code,
                              "timing":config["timing"],"runtime":runtime})
            if resume and complete_matches(folder,signature,job):
                print(f"REUSE {name} {rho} {method}",flush=True)
                if job["evaluation"]["enabled"]:
                    split=json.loads((folder/"pre/evaluation.json").read_text())["split_sha256"]
                    if name in split_hashes and split_hashes[name]!=split:
                        raise RuntimeError("Camera split differs across resumed methods")
                    split_hashes[name]=split
                continue
            if (folder/"complete.json").exists():
                raise RuntimeError(f"Existing completed run has different inputs/config/code: {folder}. "
                                   "Choose a new output directory; results are not overwritten.")
            folder.mkdir(parents=True,exist_ok=True)
            # Partial files can be resumed only for exactly the same signature.
            provenance=folder/"provenance.json"
            if provenance.exists() and json.loads(provenance.read_text())["signature"]!=signature:
                raise RuntimeError(f"Partial run signature changed: {folder}; choose another output directory")
            write_json(provenance,{"signature":signature,"input_sha256":manifest["input_sha256"],
                                   "code_sha256":code,"checkpoint_sha256":manifest["checkpoint_sha256"]})
            write_json(folder/"job.json",job)
            if method=="ghap" and max(1,math.floor(manifest["gaussian_count"]*rho))<min(manifest["gaussian_count"],1024):
                write_json(folder/"failure.json",{"type":"UnsupportedNativeBudget",
                    "status":"unsupported_native_budget","signature":signature,
                    "message":"Native depth-10 GHAP would allocate zero targets to a KD leaf; "
                              "baseline skipped without changing its partition."})
                print(f"UNSUPPORTED {name} {rho} ghap: output budget below native KD leaf count",flush=True)
                continue
            try:
                print(f"RUN {name} rho={rho} {method}",flush=True)
                worker("compress",folder/"job.json",folder,
                    ["--warmups",str(config["timing"]["warmups"]),"--repeats",str(config["timing"]["repeats"])],
                    config.get("compression_timeout_seconds"))
                comp=json.loads((folder/"compression.json").read_text())
                if config["evaluation"]["enabled"]:
                    worker("evaluate",folder/"job.json",folder/"pre",["--iteration","1"])
                    pre=validate_evaluation(folder/"pre",comp["actual_count"])
                    if name in split_hashes and split_hashes[name]!=pre["split_sha256"]:
                        raise RuntimeError("Camera split differs across methods")
                    split_hashes[name]=pre["split_sha256"]
                if config["finetune"]["enabled"] and method!="identity":
                    worker("finetune",folder/"job.json",folder/"post",["--pre-dir",str(folder/"pre")])
                    if config["evaluation"]["enabled"]:
                        last=job["finetune"].get("first_iteration",30000)+job["finetune"].get("iterations",5000)
                        worker("evaluate",folder/"job.json",folder/"post",["--iteration",str(last)])
                        post=validate_evaluation(folder/"post",comp["actual_count"])
                        if post["split_sha256"]!=split_hashes[name]:
                            raise RuntimeError("Pre/post camera split mismatch")
                finish_run(folder,signature,job)
            except Exception as exc:
                write_json(folder/"failure.json",{"type":type(exc).__name__,"message":str(exc),"signature":signature})
                failures.append(str(folder))
                if not continue_on_error:
                    raise
    from .reporting import summarize
    summarize(output)
    if failures:
        raise RuntimeError(f"{len(failures)} configurations failed; see failure.json and summary.csv")
