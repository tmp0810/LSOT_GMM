"""Atomic per-outer-iteration PLY export using GHAP's existing serializer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

from application.gaussian_reduction.ghap_integration.ghap_hot_bridge import PARAMETER_FIELDS


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_checkpoint_callback(source_model, model_root):
    """Save independent render models; STOP requests normal loop finalization.

    No optimizer step is taken and the source model is never modified. The
    copied cfg_args remains the source configuration; the official renderer's
    explicit -m argument overrides its model_path for each round directory.
    PLYs are renderable checkpoints, not exact optimizer/outer-loop resumes.
    """
    root = Path(model_root)
    previous_algorithm_seconds = 0.0

    def save(tensors, record):
        nonlocal previous_algorithm_seconds
        started = perf_counter()
        iteration = int(record["outer_iteration"])
        directory = root / f"round_{iteration:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(root / "cfg_args", directory / "cfg_args")
        exported = type(source_model)(source_model.max_sh_degree)
        exported.active_sh_degree = source_model.active_sh_degree
        for name, attribute in PARAMETER_FIELDS.items():
            setattr(exported, attribute, tensors[name].detach())
        ply = directory / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        temporary = ply.with_suffix(".ply.tmp")
        exported.save_ply(str(temporary))
        temporary.replace(ply)
        audit = dict(record)
        audit.update(
            outer_iteration=iteration,
            target_gaussians=int(tensors["xyz"].shape[0]),
            ply_sha256=sha256(ply),
            ply_bytes=ply.stat().st_size,
            fine_tuning_steps=0,
            saved_model="renderable_PLY_not_optimizer_resume",
            algorithm_iteration_seconds=(
                record["elapsed_algorithm_seconds"] - previous_algorithm_seconds
            ),
            checkpoint_write_and_hash_seconds=perf_counter() - started,
        )
        previous_algorithm_seconds = record["elapsed_algorithm_seconds"]
        write_json(directory / "snapshot.json", audit)
        (directory / "READY").touch(exist_ok=False)
        print("HOT_ROUND_SAVED " + json.dumps(audit, sort_keys=True), flush=True)
        return (root / "STOP").exists()

    return save
