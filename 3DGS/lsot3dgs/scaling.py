"""Growing one OT problem on nested subsets of actual scene Gaussians."""
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from .common import APP,code_digest,digest,environment,memory,resolve_path,seed_all,sha256,synchronize,tensor_digest,write_json,select_device,runtime_fingerprint
from .config import validate_config
from .model_io import load_scene
from .partition import hilbert_order,kd_order,initialize
from .solvers import make_solver
from .solvers.mw2 import BudgetExceeded


def scaling_case(case,folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    options=case["solver"]; device=options["device"]
    select_device(device)
    seed_all(case.get("sampling_seed",42))
    scene=load_scene(case["checkpoint"])
    positive=torch.where(scene.opacity_logits.double().sigmoid().flatten()>0)[0].numpy()
    n,m=case["n"],case["m"]
    if not 1<=m<=n<=len(positive):
        raise BudgetExceeded(f"Requested {n}->{m}, but source has {len(positive)} positive Gaussians")
    # Recreating the same permutation makes smaller subsets prefixes of larger ones.
    ids=np.random.default_rng(case.get("sampling_seed",42)).permutation(positive)[:n]
    points=scene.xyz[ids].double().numpy()
    if options.get("partition")=="hilbert":
        order=hilbert_order(points,options["hcp_extension"])
    else:
        order=kd_order(points)
    ids=ids[order]
    source,_=scene.gaussian_set(ids,device=device)
    cuts=np.arange(m+1,dtype=np.int64)*n//m
    target,initialization=initialize(source,cuts)
    # No partition of the OT problem here: every source may couple to every target.
    records=[]; last=None
    for trial in range(-case.get("warmups",1),case.get("repeats",3)):
        gc.collect()
        if torch.device(device).type=="cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        times={}; synchronize(device); start=time.perf_counter()
        solver=make_solver(case["method"],source,options,times)
        result=solver.solve(target)
        synchronize(device); cold=time.perf_counter()-start
        cold_times=dict(times)
        cold_memory=memory(device)
        objective=result.objective
        last=result.diagnostics
        del result
        warm=None
        if case.get("measure_warm",True):
            synchronize(device); start=time.perf_counter()
            solver.solve(target)
            synchronize(device); warm=time.perf_counter()-start
        if trial>=0:
            records.append({"cold_solver_seconds":cold,"warm_solver_seconds":warm,
                            "objective":objective,"timings":cold_times,**cold_memory})
        del solver
    result={"status":"complete","case":case,"n":n,"m":m,"method":case["method"],
            "problem":"one full OT problem; no local blocking",
            "cold_solver_seconds":statistics.median(r["cold_solver_seconds"] for r in records),
            "warm_solver_seconds":statistics.median(r["warm_solver_seconds"] for r in records)
                if case.get("measure_warm",True) else None,
            "objective":records[0]["objective"],"measurements":records,
            "input_subset_sha256":tensor_digest(ids),**initialization,
            "diagnostics":last,"environment":environment(device),
            "cpu_peak_rss_bytes":max(r["cpu_peak_rss_bytes"] for r in records),
            "gpu_peak_allocated_bytes":max(r.get("gpu_peak_allocated_bytes",0) for r in records)}
    # Only the parent owns result.json and its resume signature. A separate
    # worker record cannot overwrite that completed, signed result.
    write_json(folder/"worker_result.json",result)
    return result


def run_scaling(config,*,resume=True):
    validate_config(config)
    output=resolve_path(config["output_dir"]); output.mkdir(parents=True,exist_ok=True)
    write_json(output/"scaling_config.json",config)
    code=code_digest()
    runtime=runtime_fingerprint(config)
    write_json(output/"environment.json",runtime)
    failures=[]
    for name,scene in config["scenes"].items():
        checkpoint=resolve_path(scene["checkpoint"])
        input_hash=sha256(checkpoint)
        for rho in config.get("retentions",[.1]):
            for n in config.get("sizes",[1000,2000,5000,10000,20000,50000,100000]):
                for method in config["methods"]:
                    case={"scene":name,"checkpoint":str(checkpoint),"n":n,"m":max(1,int(n*rho)),
                          "method":method,"solver":config["solver"],"sampling_seed":config.get("sampling_seed",42),
                          "warmups":config["timing"]["warmups"],"repeats":config["timing"]["repeats"],
                          "measure_warm":config.get("measure_warm",True)}
                    signature=digest({"case":case,"input":input_hash,"code":code,"runtime":runtime})
                    folder=output/name/f"rho_{rho:g}"/f"N_{n}"/method
                    result_file=folder/"result.json"
                    if result_file.exists():
                        saved=json.loads(result_file.read_text())
                        if saved.get("signature") is not None and saved["signature"]!=signature:
                            raise RuntimeError(f"Scaling signature changed in {folder}; choose new output directory")
                        if resume and saved.get("signature")==signature:
                            if saved.get("status")=="failed":
                                failures.append(str(folder))
                            print(f"REUSE {name} {n}->{case['m']} {method}: {saved.get('status')}",flush=True)
                            continue
                    folder.mkdir(parents=True,exist_ok=True)
                    write_json(folder/"case.json",case)
                    worker_result=folder/"worker_result.json"
                    worker_result.unlink(missing_ok=True)
                    print(f"SCALING {name} {n}->{case['m']} {method}",flush=True)
                    cmd=[sys.executable,str(APP/"worker.py"),"scaling","--job",str(folder/"case.json"),
                         "--output-dir",str(folder)]
                    try:
                        with (folder/"worker.log").open("w") as log:
                            process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                                timeout=config.get("timeout_seconds",1800),env={**os.environ,"PYTHONUNBUFFERED":"1"})
                        if not worker_result.exists():
                            raise RuntimeError(f"Worker exited {process.returncode} without a result")
                        result=json.loads(worker_result.read_text())
                        # A killed worker may leave a result from an earlier run.
                        # Never turn a nonzero exit into a successful measurement.
                        if process.returncode and result.get("status")=="complete":
                            raise RuntimeError(f"Worker exited {process.returncode} despite a complete result file")
                    except subprocess.TimeoutExpired:
                        result={"status":"timeout","case":case,"message":"Configuration worker exceeded time budget"}
                    except Exception as exc:
                        result={"status":"failed","case":case,"message":str(exc)}
                    result["signature"]=signature
                    write_json(result_file,result)
                    if result["status"]=="failed":
                        failures.append(str(folder))
    from .reporting import summarize_scaling
    summarize_scaling(output)
    if failures:
        raise RuntimeError(f"{len(failures)} scaling cases failed unexpectedly; inspect worker.log")
