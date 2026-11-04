"""Create a source/data ZIP; exclude Git history, caches and local outputs."""
from pathlib import Path
import argparse
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", "generated", "build", "dist", ".pytest_cache", "checkouts"}


def selected():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(p in EXCLUDED or p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"refuse to follow a symlink while packing: {relative}")
        if not path.is_file() or path.name == ".DS_Store":
            continue
        if path.suffix in (".pyc", ".pyo", ".so", ".dylib", ".pyd", ".nbc", ".nbi", ".zip"):
            continue
        yield path, relative


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    target = args.output.resolve()
    if target.exists():
        p.error("refuse to overwrite an existing archive")
    if ROOT == target or ROOT in target.parents:
        p.error("put the archive outside code/ to avoid recursive packaging")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in selected():
            content = path.read_bytes()
            archive.write(path, "code/" + relative.as_posix())
            manifest.append(hashlib.sha256(content).hexdigest()+"  "+relative.as_posix())
        archive.writestr("code/PACKAGE_SHA256.txt", "\n".join(manifest)+"\n")
    with zipfile.ZipFile(target) as archive:
        failed = archive.testzip()
        if failed:
            raise RuntimeError(f"archive CRC failure: {failed}")
    print(f"{target}: {len(manifest)} files, {target.stat().st_size:,} bytes")
    print("sha256",hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
