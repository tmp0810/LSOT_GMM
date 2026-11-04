"""Read-only integrity check of every retained original file."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    manifest = json.loads((ROOT / "docs/SOURCE_MANIFEST.json").read_text())
    mismatches = []
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            mismatches.append(record["path"])
    syntax = []
    for path in ROOT.rglob("*.py"):
        if any(p in (".venv", "build", "generated") for p in path.parts):
            continue
        ast.parse(path.read_text(), filename=str(path))
        syntax.append(str(path.relative_to(ROOT)))
    result = dict(original_files=len(manifest["files"]), mismatches=mismatches,
                  python_files_parsed=len(syntax), paper_commit=manifest["paper_commit"])
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)
