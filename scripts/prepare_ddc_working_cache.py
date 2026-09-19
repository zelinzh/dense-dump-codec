#!/usr/bin/env python3
"""Create validated, reusable spatial work files without changing the DDC archive."""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from dense_dump_codec.native import NativeSequenceDecoder
from dense_dump_codec.spatial_cache import SpatialWorkingArchive, cache_path, prepare_spatial_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheme")
    parser.add_argument("--sequence-min", type=int, required=True)
    parser.add_argument("--sequence-max", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--slab-cells", type=int, default=32)
    parser.add_argument("--minimum-free-gib", type=float, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.sequence_min > args.sequence_max or args.minimum_free_gib < 0:
        parser.error("Invalid frame interval or free-space reserve")
    decoder = NativeSequenceDecoder(args.manifest, scheme=args.scheme)
    sequences = [sequence for sequence in decoder.times
                 if args.sequence_min <= sequence <= args.sequence_max and sequence not in decoder.keyframes]
    if not sequences:
        parser.error("The requested interval contains no intermediate DDC states")
    _, scheme = decoder.sequence.select_scheme(decoder.manifest, args.scheme)
    paths = decoder.sequence.group_archive_frames(scheme["archive_paths"], sequences)
    if args.dry_run:
        print(json.dumps({"source_archives": list(map(str, paths)),
                          "output_dir": str(args.output_dir), "source_unchanged": True}, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = cache_path(source, args.output_dir)
        if target.exists():
            with SpatialWorkingArchive(target, source=source) as archive:
                if archive.index["slab_cells"] != args.slab_cells:
                    raise ValueError("Existing working cache uses another slab size; choose another directory")
                archive.validate_all()
            print(json.dumps({"reused_validated_cache": str(target)}), flush=True)
            continue
        if shutil.disk_usage(args.output_dir).free < args.minimum_free_gib * 1024**3:
            raise RuntimeError("Free-space reserve reached; no further GOP will be started")
        result = prepare_spatial_cache(source, args.output_dir, codec=decoder.codec,
                                       slab_cells=args.slab_cells, workers=args.workers)
        record = target.with_suffix(".json")
        record.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
