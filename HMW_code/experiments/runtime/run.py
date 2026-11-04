"""Explicit launcher for the unchanged Figure 3 server timing scripts.

Must run on the original-compatible Linux environment pinned to CPU 52.
This script does NOT SSH, reserve a GPU, or silently change the protocol.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("sample", "dimension"))
    p.add_argument("--official-repository", type=Path, required=True,
                   help="clean upstream DSMW git checkout at the pinned commit")
    p.add_argument("--hcp-root", type=Path, default=ROOT / "build/hcp")
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    args = p.parse_args()
    if not hasattr(os, "sched_getaffinity") or sorted(os.sched_getaffinity(0)) != [52]:
        p.error("historical protocol requires Linux CPU 52: launch with taskset -c 52")
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    original = ROOT / "experiments/_original"
    command = [sys.executable, str(original / f"benchmark_hmw_{args.kind}_scaling_server.py"),
               "--experiment-root", str(original), "--hcp-root", str(args.hcp_root.resolve()),
               "--official-repository", str(args.official_repository.resolve()), "--output", str(args.output.resolve())]
    if args.kind == "sample":
        command += ["--sizes", "100", "1000", "10000", "100000"]
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
