"""Resolve the retained legacy package without duplicating its source files."""
from pathlib import Path
__path__.append(str(Path(__file__).resolve().parents[3] / "_original"))
