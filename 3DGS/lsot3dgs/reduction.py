"""Shared blockwise alternating reduction and honest LSOT stopping diagnostics."""
import json
import math
import numpy as np
import torch

from .appearance import transfer
from .common import digest,timed
from .geometry import bures_update
from .model_io import SceneGaussians,geometry_to_scene
from .partition import make_partition,initialize
from .plans import plan_change
from .solvers import make_solver


def reduce_block(source, initial, method, options, timings, emit=lambda row:None):
    solver = make_solver(method,source,options,timings)
    current = initial
    result = solver.solve(current)
    best = (current,result.plan,result.objective,0,result.diagnostics,
            "initial_geometry_refreshed_plan")
    emit({"iteration":0,"objective":result.objective,"selected_best":True,**result.diagnostics})
    stable_rounds = 0
    converged = False
    updates = options.get("outer_steps",10)
    last = {}
    for step in range(1,updates+1):
        with timed(timings,"bures_update_seconds",source.means.device):
            candidate,inner = bures_update(source,current,result.plan,
                steps=options.get("bures_steps",64),tolerance=options.get("bures_tolerance",5e-4),
                chunk=options.get("pair_chunk",4096))
        with timed(timings,"fixed_plan_audit_seconds",source.means.device):
            fixed_cost = solver.cost.plan_cost(candidate,result.plan)
        # An inaccurate inner solve must never silently be treated as descent.
        if fixed_cost > result.objective + 1e-8*max(abs(result.objective),1e-12):
            raise RuntimeError(f"Fixed-plan barycenter increased cost: {result.objective} -> {fixed_cost}")
        refreshed = solver.solve(candidate)
        with timed(timings,"convergence_audit_seconds",source.means.device):
            delta_plan = plan_change(result.plan,refreshed.plan,options.get("diagnostic_max_edges",2_000_000))
            mean_scale = torch.linalg.vector_norm(current.means).clamp_min(1e-12)
            cov_scale = torch.linalg.vector_norm(current.covariances).clamp_min(1e-12)
            geom = max(float(torch.linalg.vector_norm(candidate.means-current.means)/mean_scale),
                       float(torch.linalg.vector_norm(candidate.covariances-current.covariances)/cov_scale))
        relative = abs(refreshed.objective-result.objective)/max(abs(result.objective),1e-12)
        selected = refreshed.objective < best[2]
        if selected:
            # Appearance uses the plan that GENERATED the selected geometry.
            best = (candidate,result.plan,refreshed.objective,step,refreshed.diagnostics,
                    "previous_iterate_plan_generating_geometry")
        last = {"iteration":step,"objective":refreshed.objective,"fixed_plan_objective":fixed_cost,
                "geometry_relative_change":geom,"objective_relative_change":relative,
                "plan_l1_change_upper_bound":delta_plan,"selected_best":selected,
                **inner,**refreshed.diagnostics}
        emit(last)
        stopped = (geom <= options.get("geometry_tolerance",5e-4)
                   and relative <= options.get("objective_tolerance",1e-6)
                   and delta_plan is not None and delta_plan <= options.get("plan_tolerance",1e-6)
                   and inner["inner_converged"])
        stable_rounds = stable_rounds+1 if stopped else 0
        current,result = candidate,refreshed
        if stable_rounds >= options.get("stable_rounds",2):
            converged = True
            break
    geometry,appearance_plan,objective,best_step,diagnostics,role = best
    return geometry,appearance_plan,{"objective":objective,"best_iteration":best_step,
        "iterations":last.get("iteration",0),"converged":converged,
        "status":"converged" if converged else "iteration_cap",
        "last_objective":result.objective,"appearance_plan_role":role,
        "selected_plan_diagnostics":diagnostics,"last_iteration":last}


@torch.no_grad()
def compress_controlled(scene,method,retention,options,*,history=None):
    requested = int(options.get("target_count") or max(1,math.floor(scene.count*retention)))
    device = options.get("device","cuda")
    dtype = torch.float64
    timings = {}
    with timed(timings,"partition_seconds"):
        partition = make_partition(scene,requested,options)
    outputs,records = [],[]
    original_mass = scene.opacity_logits.double().sigmoid().sum().item()
    global_objective = 0.0
    for block,ids,cuts in partition.blocks():
        with timed(timings,"source_extraction_seconds",device):
            source,repairs = scene.gaussian_set(ids,device=device,dtype=dtype)
        with timed(timings,"initialization_seconds",device):
            initial,init_record = initialize(source,cuts)
        def emit(row):
            if history is not None:
                history.write(json.dumps({"block":block,**row},allow_nan=False)+"\n")
                history.flush()
        target,plan,record = reduce_block(source,initial,method,options,timings,emit)
        with timed(timings,"appearance_seconds",device):
            sh,opacity,appearance = transfer(scene,ids,source,target,plan,
                rule=options.get("appearance","incoming"),chunk=options.get("pair_chunk",4096))
        with timed(timings,"export_conversion_seconds",device):
            output,export = geometry_to_scene(target,sh,opacity,spatial_lr_scale=scene.spatial_lr_scale,
                                               export_floor=options.get("export_floor","relative"))
        block_mass = scene.opacity_logits[ids].double().sigmoid().sum().item()/original_mass
        global_objective += block_mass*record["objective"]
        outputs.append(output)
        records.append({"block":block,"source":len(ids),"target":target.count,"block_mass":block_mass,
                        "source_covariance_repairs":repairs,**init_record,**record,**export,
                        "appearance_rule":appearance["rule"],
                        "donor_sha256":digest(appearance.get("donor_original_ids",[]))})
        print(f"{method}: block {block+1}/{len(partition.rep_cuts)-1}, {len(ids)} -> {target.count}, "
              f"J={record['objective']:.6g}, {record['status']}",flush=True)
        del source,initial,target,plan
    result = SceneGaussians(*(torch.cat([s.tensors()[k] for s in outputs]) for k in range(5)),
                            scene.spatial_lr_scale)
    if result.count != requested:
        raise AssertionError("Controlled reducer changed the output budget")
    return result,{"method":method,"retention_requested":retention,"requested_count":requested,
        "actual_count":result.count,"source_count":scene.count,"objective":global_objective,
        "objective_scope":"blockwise, block mass weighted; not global exact MW2",
        "partition":partition.metadata(),"blocks":records,"timings":timings,
        "all_blocks_converged":all(x["converged"] for x in records),
        "initialization_sha256":digest([x["initialization_sha256"] for x in records]),
        "capacities_sha256":digest([x["capacity_sha256"] for x in records])}
