#!/usr/bin/env python3
"""Repack logical DDC members as indexed channel-major bzip2 chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import (  # noqa: E402
    open_archive,
    repack_zip_to_channel_bzip2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--workers", type=int, default=1)
    temporal = parser.add_mutually_exclusive_group()
    temporal.add_argument("--temporal-delta", action="store_true")
    temporal.add_argument("--temporal-delta-shuffle", action="store_true")
    temporal.add_argument("--adaptive-temporal-order", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = repack_zip_to_channel_bzip2(
        args.source,
        args.output,
        chunk_frames=args.chunk_frames,
        compression_level=args.compression_level,
        workers=args.workers,
        temporal_delta=args.temporal_delta,
        temporal_delta_shuffle=args.temporal_delta_shuffle,
        adaptive_temporal_order=args.adaptive_temporal_order,
        metadata_updates={
            "compression": (
                "channel_bzip2_adaptive"
                if args.adaptive_temporal_order
                else (
                    "channel_bzip2_delta_shuffle"
                    if args.temporal_delta_shuffle
                    else (
                        "channel_bzip2_delta"
                        if args.temporal_delta
                        else "channel_bzip2"
                    )
                )
            ),
            "compression_level": args.compression_level,
            "channel_chunk_frames": args.chunk_frames,
        },
        overwrite=args.overwrite,
    )
    if not args.skip_verify:
        with open_archive(args.output) as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Archive verification failed at {bad_member}")
        result["verified"] = True
    else:
        result["verified"] = False
    result["saving_vs_source_fraction"] = 1.0 - (
        result["output_bytes"] / result["source_bytes"]
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
