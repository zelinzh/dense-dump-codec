#!/usr/bin/env python3
"""Validate a contiguous KHARMA PHDF interval before destructive DDC encoding."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import h5py
import numpy as np


DEFAULT_DATASETS = ("prims.rho", "prims.u", "prims.uvec", "prims.B")
SEQUENCE_RE = re.compile(r"\.out0\.(\d+)\.phdf$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start-sequence", type=int, required=True)
    parser.add_argument("--end-sequence", type=int, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--expected-cadence", type=float, default=0.1)
    parser.add_argument("--time-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--minimum-size-ratio", type=float, default=0.95)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sequence(path: Path) -> int:
    match = SEQUENCE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer sequence from {path}")
    return int(match.group(1))


def selected_files(directory: Path, start: int, end: int) -> list[Path]:
    by_sequence = {
        sequence(path): path
        for path in directory.glob("*.phdf")
        if SEQUENCE_RE.search(path.name)
    }
    missing = [value for value in range(start, end + 1) if value not in by_sequence]
    if missing:
        raise ValueError(f"Missing PHDF sequences: {missing[:20]}")
    return [by_sequence[value] for value in range(start, end + 1)]


def read_dataset(dataset: h5py.Dataset) -> tuple[int, int]:
    if not dataset.shape:
        values = np.asarray(dataset[()])
        finite = int(np.count_nonzero(np.isfinite(values))) if values.dtype.kind in "fc" else values.size
        return int(values.size), finite
    element_count = 0
    finite_count = 0
    for index in range(dataset.shape[0]):
        values = np.asarray(dataset[index])
        element_count += int(values.size)
        finite_count += (
            int(np.count_nonzero(np.isfinite(values)))
            if values.dtype.kind in "fc"
            else int(values.size)
        )
    return element_count, finite_count


def validate_file(
    path: Path,
    datasets: tuple[str, ...],
    schema: dict[str, tuple[tuple[int, ...], str]] | None,
    *,
    metadata_only: bool,
) -> tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], str]]]:
    row: dict[str, Any] = {
        "path": str(path.resolve()),
        "sequence": sequence(path),
        "size_bytes": path.stat().st_size,
        "datasets": {},
    }
    with h5py.File(path, "r") as handle:
        if "Info" not in handle or "Time" not in handle["Info"].attrs:
            raise ValueError("Missing Info/Time attribute")
        row["time"] = float(handle["Info"].attrs["Time"])
        if not math.isfinite(row["time"]):
            raise ValueError("Info/Time is not finite")
        current_schema: dict[str, tuple[tuple[int, ...], str]] = {}
        for name in datasets:
            if name not in handle:
                raise ValueError(f"Missing dataset {name}")
            dataset = handle[name]
            descriptor = (tuple(int(value) for value in dataset.shape), str(dataset.dtype))
            current_schema[name] = descriptor
            if schema is not None and descriptor != schema[name]:
                raise ValueError(f"Schema mismatch for {name}: {descriptor} != {schema[name]}")
            dataset_row: dict[str, Any] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "storage_bytes": int(dataset.id.get_storage_size()),
            }
            if dataset.size and dataset_row["storage_bytes"] <= 0:
                raise ValueError(f"Dataset {name} has no allocated storage")
            if not metadata_only:
                element_count, finite_count = read_dataset(dataset)
                dataset_row.update(
                    {
                        "element_count": element_count,
                        "finite_count": finite_count,
                    }
                )
                if finite_count != element_count:
                    raise ValueError(
                        f"Dataset {name} contains {element_count - finite_count} non-finite values"
                    )
            row["datasets"][name] = dataset_row
    return row, current_schema


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.end_sequence < args.start_sequence:
        raise ValueError("--end-sequence must be >= --start-sequence")
    if not 0.0 < args.minimum_size_ratio <= 1.0:
        raise ValueError("--minimum-size-ratio must be in (0, 1]")
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    files = selected_files(args.directory, args.start_sequence, args.end_sequence)
    typical_size = float(median(path.stat().st_size for path in files))
    report: dict[str, Any] = {
        "format": "kharma_phdf_integrity_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "directory": str(args.directory.resolve()),
        "start_sequence": args.start_sequence,
        "end_sequence": args.end_sequence,
        "file_count": len(files),
        "datasets": list(datasets),
        "full_dataset_read": not args.metadata_only,
        "typical_size_bytes": typical_size,
        "files": [],
        "errors": [],
    }
    schema = None
    first_time = None
    for path in files:
        try:
            size_ratio = path.stat().st_size / typical_size
            if size_ratio < args.minimum_size_ratio:
                raise ValueError(
                    f"File size ratio {size_ratio:.6f} is below {args.minimum_size_ratio:.6f}"
                )
            row, current_schema = validate_file(
                path,
                datasets,
                schema,
                metadata_only=args.metadata_only,
            )
            schema = current_schema if schema is None else schema
            first_time = row["time"] if first_time is None else first_time
            expected_time = first_time + (row["sequence"] - args.start_sequence) * args.expected_cadence
            if abs(row["time"] - expected_time) > args.time_tolerance:
                raise ValueError(
                    f"Time {row['time']:.16g} differs from expected {expected_time:.16g}"
                )
            row["size_ratio"] = size_ratio
            report["files"].append(row)
        except Exception as error:
            report["errors"].append(
                {"path": str(path.resolve()), "sequence": sequence(path), "error": str(error)}
            )
    report["passed"] = not report["errors"] and len(report["files"]) == len(files)
    return report


def main() -> int:
    args = parse_args()
    report = validate(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "file_count": report["file_count"],
                "validated_file_count": len(report["files"]),
                "errors": report["errors"],
                "output": str(args.output.resolve()) if args.output is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
