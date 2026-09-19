#!/usr/bin/env python3
"""Compress or reconstruct a lossless byte-shuffled PHDF keyframe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import compress_keyframe, decompress_keyframe  # noqa: E402


DEFAULT_DATASETS = "prims.rho,prims.u,prims.uvec,prims.B,cons.fB,divB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compress = subparsers.add_parser("compress")
    compress.add_argument("source", type=Path)
    compress.add_argument("output", type=Path)
    compress.add_argument("--datasets", default=DEFAULT_DATASETS)
    compress.add_argument("--compression-level", type=int, default=9)
    compress.add_argument("--workers", type=int, default=1)
    compress.add_argument("--overwrite", action="store_true")
    decompress = subparsers.add_parser("decompress")
    decompress.add_argument("source", type=Path)
    decompress.add_argument("output", type=Path)
    decompress.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compress":
        datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
        result = compress_keyframe(
            args.source,
            args.output,
            datasets,
            compression_level=args.compression_level,
            workers=args.workers,
            overwrite=args.overwrite,
        )
    else:
        result = decompress_keyframe(args.source, args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
