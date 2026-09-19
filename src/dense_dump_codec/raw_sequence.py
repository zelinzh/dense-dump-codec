"""Cached physical-time indices for retained raw KHARMA PHDF sequences."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py

from .sequence import DenseSequenceIndex, FrameRecord


PHDF_SEQUENCE = re.compile(r"\.out\d+\.(\d+)\.phdf$")


def phdf_sequence(path: Path) -> int:
    match = PHDF_SEQUENCE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse PHDF sequence from {path}")
    return int(match.group(1))


def candidate_files(directory: Path, dense_sequence_stride: int) -> list[Path]:
    if dense_sequence_stride < 1:
        raise ValueError("dense_sequence_stride must be positive")
    rows = []
    for path in directory.glob("*.phdf"):
        try:
            sequence = phdf_sequence(path)
        except ValueError:
            continue
        if sequence % dense_sequence_stride == 0:
            rows.append((sequence, path.resolve()))
    rows.sort()
    if not rows:
        raise ValueError(
            f"No PHDF files in {directory} match stride {dense_sequence_stride}"
        )
    sequences = [sequence for sequence, _ in rows]
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"Duplicate PHDF sequences found in {directory}")
    return [path for _, path in rows]


def file_signature(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def read_frame(path: Path) -> FrameRecord:
    with h5py.File(path, "r") as handle:
        if "Info" not in handle or "Time" not in handle["Info"].attrs:
            raise ValueError(f"{path} has no Info/Time attribute")
        time = float(handle["Info"].attrs["Time"])
    return FrameRecord(
        sequence=phdf_sequence(path),
        time=time,
        path=str(path),
        size_bytes=path.stat().st_size,
        exact_keyframe=True,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_raw_sequence_index(
    directory: Path,
    *,
    dense_sequence_stride: int,
    cache_path: Path,
) -> tuple[DenseSequenceIndex, dict[str, Any], bool]:
    directory = directory.resolve()
    paths = candidate_files(directory, dense_sequence_stride)
    signature = file_signature(paths)
    identity = {
        "directory": str(directory),
        "dense_sequence_stride": dense_sequence_stride,
        "source_signature": signature,
    }
    reused = False
    payload: dict[str, Any]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("identity") == identity and cached.get("frames"):
            payload = cached
            reused = True
        else:
            payload = {}
    else:
        payload = {}
    if not reused:
        frames = [read_frame(path) for path in paths]
        payload = {
            "format": "kharma_raw_sequence_index_v1",
            "identity": identity,
            "frames": [frame.to_json() for frame in frames],
            "frame_count": len(frames),
            "time_start_M": frames[0].time,
            "time_end_M": frames[-1].time,
        }
        write_json(cache_path, payload)
    index = DenseSequenceIndex(FrameRecord.from_json(row) for row in payload["frames"])
    return index, payload, reused
