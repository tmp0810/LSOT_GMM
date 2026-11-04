"""Paths, reproducibility records and atomic output helpers."""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "3DGS"
GHAP_COMMIT = "27b3a581c861069cc97e5629e216dc3d1f42a98a"
METHODS = ("mw2",) + tuple(f"{mode}-{kind}" for mode in ("min", "avg")
                           for kind in ("Mix", "SMix", "B", "B1D"))
BASELINES = ("ghap", "hmw-local", "identity")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def tensor_digest(*tensors):
    h = hashlib.sha256()
    for value in tensors:
        a = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def code_digest():
    # Include algorithm dependencies, not results or this generated manifest.
    paths = []
    for folder in (APP, ROOT / "lsot", ROOT / "param_proj", ROOT / "distribution_proj",
                   ROOT / "HMW_code/experiments/scene_compression/original",
                   ROOT / "HMW_code/src/hmw"):
        paths.extend(folder.rglob("*.py"))
    return digest({str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)
                   if "third_party" not in p.parts and "__pycache__" not in p.parts})


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def select_device(device):
    device=torch.device(device)
    if device.type=="cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        index=device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        return torch.device("cuda",index)
    return device


@contextmanager
def timed(record, key, device="cpu"):
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        record[key] = record.get(key, 0.0) + time.perf_counter() - start


def memory(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"cpu_peak_rss_bytes": int(rss * (1 if sys.platform == "darwin" else 1024))}
    if torch.device(device).type == "cuda":
        result.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def environment(device):
    result = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "numpy": np.__version__, "cuda_runtime": torch.version.cuda, "device": str(device),
              "cpu_threads": torch.get_num_threads(), "code_sha256": code_digest()}
    if torch.device(device).type == "cuda":
        p = torch.cuda.get_device_properties(device)
        result.update(gpu=p.name, gpu_total_bytes=p.total_memory,
                      capability=list(torch.cuda.get_device_capability(device)))
    return result


def runtime_fingerprint(config):
    """Changing hardware/dependencies invalidates cached timing and quality records."""
    import ot
    result=environment(config["solver"]["device"])
    result.update(pot=ot.__version__,host=platform.node(),cpu_count=os.cpu_count())
    if config["solver"]["partition"]=="hilbert":
        result["hcp_binary_sha256"]=sha256(resolve_path(config["solver"]["hcp_extension"]))
    needs_ghap=(config["evaluation"]["enabled"] or config["finetune"]["enabled"]
                or "ghap" in config["methods"])
    if needs_ghap:
        root,version=check_ghap(config["ghap_root"])
        version["submodules"]=subprocess.check_output(
            ["git","-C",str(root),"submodule","status"],text=True).strip()
        result["ghap"]=version
    return result


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def checked_process(command, log, *, cwd=None, timeout=None, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log.open("w") as f:
        f.write(json.dumps([str(s) for s in command]) + "\n")
        f.flush()
        proc = subprocess.run([str(s) for s in command], cwd=cwd, stdout=f,
                              stderr=subprocess.STDOUT, timeout=timeout, env=env)
    if proc.returncode:
        raise RuntimeError(f"Process exited {proc.returncode}; see {log}")
    return time.perf_counter() - start


def check_ghap(path, allow_modified=False):
    path = resolve_path(path)
    if not (path / "train_and_prune.py").is_file():
        raise FileNotFoundError(f"Missing GHAP checkout: {path}; run 3DGS/setup_env.sh")
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "diff", "--name-only", "HEAD"], text=True).strip()
    if not allow_modified and (head != GHAP_COMMIT or dirty):
        raise RuntimeError(f"GHAP must be clean at {GHAP_COMMIT}; got {head}, changed={dirty!r}")
    return path, {"commit": head, "modified": bool(dirty)}
