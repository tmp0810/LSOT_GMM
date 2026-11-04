"""Build the SHA-verified original base.cpp without editing it."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import subprocess
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "8f39fe76d99be69459b80612b5c6adb64176da6c9059d678a948b355cef0cddd"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eigen-include", type=Path, help="directory containing Eigen/Core")
    p.add_argument("--output", type=Path, default=ROOT / "build/hcp")
    args = p.parse_args()
    source = ROOT / "third_party/hcp/src/base.cpp"
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED:
        p.error("upstream base.cpp hash mismatch")
    import pybind11
    options = [args.eigen_include, Path(os.environ.get("EIGEN3_INCLUDE_DIR", "/nonexistent")),
               Path("/usr/include/eigen3"), Path("/opt/homebrew/include/eigen3"),
               Path("/usr/local/include/eigen3")]
    eigen = next((d for d in options if d is not None and (d / "Eigen/Core").is_file()), None)
    if eigen is None:
        p.error("Eigen3 not found; pass --eigen-include (see docs/REPRODUCTION.md)")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        p.error("C++ compiler not found")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "base.so"
    command = [compiler, "-O3", "-shared", "-std=c++11", "-fPIC",
               "-I" + str(eigen), "-I" + pybind11.get_include(),
               "-I" + sysconfig.get_path("include"), str(source), "-o", str(target)]
    if sysconfig.get_platform().startswith("macosx"):
        command[1:1] = ["-undefined", "dynamic_lookup"]
    subprocess.run(command, check=True)
    print(target)


if __name__ == "__main__":
    main()
