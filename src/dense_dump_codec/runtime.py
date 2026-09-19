"""Locate command helpers in a source checkout or an installed wheel."""

from pathlib import Path


def runtime_project() -> Path:
    package = Path(__file__).resolve().parent
    for candidate in (package.parent.parent, package / "_runtime"):
        if (candidate / "scripts" / "decode_dense_sequence.py").is_file():
            return candidate
    raise FileNotFoundError("DDC command helpers are missing; reinstall the complete distribution")
