import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from lsot3dgs.baselines import hmw_local,import_hmw_solver
from lsot3dgs.common import ROOT,digest,sha256,write_json
from lsot3dgs.config import load_config
from lsot3dgs.geometry import GaussianSet,bures_update,GaussianCost
from lsot3dgs.model_io import SceneGaussians,save_ply,load_scene
from lsot3dgs.partition import make_partition,initialize,hilbert_order
from lsot3dgs.pipeline import measure_compression,complete_matches,finish_run
from lsot3dgs.reporting import summarize_scaling
from lsot3dgs.solvers import make_solver


def fixture_scene(n=12):
    g=torch.Generator().manual_seed(44)
    return SceneGaussians(torch.randn(n,3,generator=g),torch.randn(n,3,generator=g)*.2-2,
        torch.randn(n,4,generator=g),torch.randn(n,1,generator=g),torch.randn(n,16,3,generator=g))


def test_native_hmw_adapter_matches_reviewed_solver():
    scene=fixture_scene()
    options={"device":"cpu","partition":"input","block_cap":100}
    result,record=hmw_local(scene,.25,options)
    source,_=scene.gaussian_set()
    labels=torch.repeat_interleave(torch.arange(3),4)[None]
    centers,cov,donors,_,_=import_hmw_solver()(source.means[None],source.covariances[None],
                                             source.weights[None],labels,3)
    restored,_=result.gaussian_set()
    torch.testing.assert_close(restored.means,centers[0],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(restored.covariances,cov[0],rtol=3e-6,atol=1e-9)
    torch.testing.assert_close(result.sh,scene.sh[donors[0]])
    assert record["actual_count"]==3


def test_compression_repeat_export_and_resume_tamper_detection(tmp_path):
    path=tmp_path/"source.ply"; save_ply(path,fixture_scene())
    config=load_config(ROOT/"3DGS/configs/smoke.json")
    options={**config["solver"],"outer_steps":1,"projections":3}
    job={"scene":{"checkpoint":str(path)},"solver":options,"method":"min-B1D","retention":.25,
         "evaluation":{"enabled":False},"finetune":{"enabled":False}}
    folder=tmp_path/"run"
    measured=measure_compression(job,folder,1,2)
    assert len(measured["compression_seconds_samples"])==2
    assert measured["actual_count"]==3
    assert load_scene(folder/"pre/point_cloud/iteration_1/point_cloud.ply").count==3
    finish_run(folder,"example",job)
    assert complete_matches(folder,"example",job)
    assert not complete_matches(folder,"different",job)
    with (folder/"pre/point_cloud/iteration_1/point_cloud.ply").open("ab") as f:
        f.write(b"tampered")
    assert not complete_matches(folder,"example",job)


def test_scaling_does_not_invent_missing_mw2_speedup(tmp_path):
    base={"scene":"s","solver":{"projections":100},"n":1000,"m":100}
    for method,status in (("mw2","timeout"),("min-B1D","complete")):
        write_json(tmp_path/f"s/rho_0.1/N_1000/{method}/result.json",
                   {"case":{**base,"method":method},"status":status,
                    "cold_solver_seconds":2. if status=="complete" else None,
                    "objective":1. if status=="complete" else None})
    rows=summarize_scaling(tmp_path)
    assert all(r["speedup_vs_mw2"] is None for r in rows)


def test_scaling_rejects_failed_worker_with_stale_complete_result(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from lsot3dgs import scaling
    config=load_config(ROOT/"3DGS/configs/smoke.json")
    checkpoint=tmp_path/"source.ply"; save_ply(checkpoint,fixture_scene())
    config.update(output_dir=str(tmp_path/"scaling"),methods=["mw2"],sizes=[12],
                  retentions=[.25],scenes={"synthetic":{"checkpoint":str(checkpoint)}})
    folder=tmp_path/"scaling/synthetic/rho_0.25/N_12/mw2"
    write_json(folder/"result.json",{"status":"complete"})
    write_json(folder/"worker_result.json",{"status":"complete"})
    monkeypatch.setattr(scaling,"runtime_fingerprint",lambda config:{})
    monkeypatch.setattr(scaling.subprocess,"run",lambda *args,**kwargs:SimpleNamespace(returncode=-9))
    with pytest.raises(RuntimeError,match="failed unexpectedly"):
        scaling.run_scaling(config,resume=False)
    result=json.loads((folder/"result.json").read_text())
    assert result["status"]=="failed" and "-9" in result["message"]
    # Resuming a recorded failure must not falsely report a successful benchmark.
    with pytest.raises(RuntimeError,match="failed unexpectedly"):
        scaling.run_scaling(config)


def test_thin_rotated_covariances_remain_finite():
    scene=fixture_scene(12)
    scene.log_scales[:,0]=-10
    scene.log_scales[:,1]=-5
    source,_=scene.gaussian_set()
    target,_=initialize(source,np.array([0,4,8,12]))
    result=make_solver("min-B1D",source,{"projections":8},{}).solve(target)
    updated,info=bures_update(source,target,result.plan,steps=100,tolerance=1e-6)
    assert torch.isfinite(updated.covariances).all()
    assert GaussianCost(source).plan_cost(updated,result.plan)<=result.objective+1e-8


@pytest.mark.skipif(not (ROOT/"HMW_code/build/hcp/base.so").exists(),reason="Build HCP first")
def test_native_hcp_partition_is_exact_and_deterministic():
    points=np.random.default_rng(7).normal(size=(31,3))
    a=hilbert_order(points,"HMW_code/build/hcp/base.so")
    b=hilbert_order(points,"HMW_code/build/hcp/base.so")
    assert np.array_equal(np.sort(a),np.arange(31)) and np.array_equal(a,b)
