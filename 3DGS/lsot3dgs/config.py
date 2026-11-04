import copy
import json
from pathlib import Path

from .common import APP,METHODS,BASELINES,resolve_path

DEFAULT_SOLVER={"device":"cuda","partition":"hilbert","block_cap":100,"projections":100,
    "bank_size":200,"projection_seed":0,"compression_seed":42,"projection_batch":8,
    "component_chunk":4096,"pair_chunk":4096,"outer_steps":10,"bures_steps":64,
    "bures_tolerance":5e-4,"geometry_tolerance":5e-4,"objective_tolerance":1e-6,
    "plan_tolerance":1e-6,"stable_rounds":2,"appearance":"incoming","export_floor":"relative",
    "max_dense_bytes":17179869184,"emd_iterations":1000000,"emd_threads":1,
    "hcp_extension":"HMW_code/build/hcp/base.so"}


def load_config(path):
    config=json.loads(Path(path).read_text())
    config["solver"]={**DEFAULT_SOLVER,**config.get("solver",{})}
    config.setdefault("methods",["mw2","min-Mix","min-SMix","min-B","min-B1D","ghap","hmw-local"])
    config.setdefault("retentions",[.1,.01,.001])
    config.setdefault("ghap_root","3DGS/third_party/GHAP")
    config.setdefault("output_dir","results/3dgs")
    config.setdefault("timing",{"warmups":1,"repeats":3})
    config.setdefault("finetune",{"enabled":True,"first_iteration":30000,"iterations":5000,"seed":0})
    config.setdefault("evaluation",{"enabled":True,"render_fps":False})
    validate_config(config)
    return config


def validate_config(config):
    if any(method not in METHODS+BASELINES for method in config["methods"]):
        raise ValueError(f"Unknown method; expected {METHODS+BASELINES}")
    if any(not 0<rho<=1 for rho in config["retentions"]):
        raise ValueError("Retentions must be in (0,1]")
    if config["timing"]["repeats"]<1 or config["timing"]["warmups"]<0:
        raise ValueError("Need positive timing repeats and nonnegative warmups")
    if not config["methods"] or not config["scenes"]:
        raise ValueError("At least one scene and method are required")
    s=config["solver"]
    for key in ("block_cap","projections","bank_size","projection_batch","component_chunk", "pair_chunk",
                "bures_steps","stable_rounds","max_dense_bytes","emd_threads"):
        if not isinstance(s[key],int) or s[key]<1:
            raise ValueError(f"solver.{key} must be a positive integer")
    if s["outer_steps"]<0 or s["projections"]>s["bank_size"]:
        raise ValueError("Invalid outer_steps or projection bank size")
    if s["appearance"] not in ("incoming","full-block","mean"):
        raise ValueError("Unknown appearance mode")
    if s["export_floor"] not in ("relative","ghap"):
        raise ValueError("Unknown covariance export policy")
    if s["partition"] not in ("hilbert","kd","input"):
        raise ValueError("Unknown spatial order")


def job_config(config,name,method,retention):
    scene=copy.deepcopy(config["scenes"][name])
    scene["name"]=name
    scene["checkpoint"]=str(resolve_path(scene["checkpoint"]))
    if scene.get("dataset"):
        scene["dataset"]=str(resolve_path(scene["dataset"]))
    return {"scene":scene,"method":method,"retention":retention,
            "solver":{**config["solver"],"ghap_root":str(resolve_path(config["ghap_root"]))},
            "ghap_root":str(resolve_path(config["ghap_root"])),
            "finetune":copy.deepcopy(config["finetune"]),
            "evaluation":copy.deepcopy(config["evaluation"])}
