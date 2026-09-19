#!/usr/bin/env python3
"""Materialize a requested DDC frame and a bounded forward prefetch batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from decode_dense_sequence import decode_sequence_frame, decode_sequence_frames

from dense_dump_codec.sequence import DDCFrameMaterializer, DenseSequenceIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    requested = parser.add_mutually_exclusive_group(required=True)
    requested.add_argument("--time", type=float)
    requested.add_argument("--sequence", type=int)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scheme")
    parser.add_argument("--datasets")
    parser.add_argument("--maximum-cache-files", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = DenseSequenceIndex.from_manifest(args.manifest)
    if args.sequence is not None:
        first_record = index.by_sequence(args.sequence)
    else:
        first_record = index.bracket(args.time, tolerance=args.tolerance).lower
    first_position = index.frames.index(first_record)
    prefetch_records = index.frames[
        first_position : first_position + args.maximum_cache_files
    ]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    prefetch_outputs = {
        record.sequence: args.cache_dir / f"ddc_frame_{record.sequence:05d}.phdf"
        for record in prefetch_records
        if not (args.cache_dir / f"ddc_frame_{record.sequence:05d}.phdf").exists()
    }
    if prefetch_outputs:
        decode_sequence_frames(
            args.manifest,
            prefetch_outputs,
            requested_scheme=args.scheme,
            datasets=args.datasets,
            overwrite=False,
        )

    def decode(sequence: int, output: Path) -> dict:
        return decode_sequence_frame(
            args.manifest,
            sequence,
            output,
            requested_scheme=args.scheme,
            datasets=args.datasets,
            overwrite=False,
        )

    materializer = DDCFrameMaterializer(
        index,
        args.cache_dir,
        decode,
        maximum_cache_files=args.maximum_cache_files,
    )
    if args.sequence is not None:
        result = materializer.materialize_sequence(args.sequence)
    else:
        result = materializer.materialize_time(args.time, tolerance=args.tolerance)
    result["prefetched_sequences"] = [record.sequence for record in prefetch_records]
    result["batch_decoded_sequences"] = sorted(prefetch_outputs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
