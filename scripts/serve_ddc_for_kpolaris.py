#!/usr/bin/env python3
"""Serve bounded, atomically materialized DDC frames over a Unix socket."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from decode_dense_sequence import decode_sequence_frames, decode_sequence_frames_to_arrays

from dense_dump_codec.kpolaris_service import (
    DDCFrameService,
    DDCUnixServer,
    write_service_statistics,
)
from dense_dump_codec.sequence import DenseSequenceIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scheme")
    parser.add_argument("--datasets")
    parser.add_argument("--maximum-cache-files", type=int, default=4)
    parser.add_argument("--prefetch-files", type=int)
    parser.add_argument("--native-decode-workers", type=int, default=1)
    parser.add_argument("--native-archive-cache-chunks", type=int, default=128)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--stats-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = DenseSequenceIndex.from_manifest(args.manifest)

    def decode(outputs: dict[int, Path]) -> dict[int, dict]:
        return decode_sequence_frames(
            args.manifest,
            outputs,
            requested_scheme=args.scheme,
            datasets=args.datasets,
            overwrite=False,
        )

    def decode_staged(sequences: tuple[int, ...]) -> dict[int, dict]:
        return decode_sequence_frames_to_arrays(
            args.manifest,
            sequences,
            requested_scheme=args.scheme,
            workers=args.native_decode_workers,
            archive_cache_chunks=args.native_archive_cache_chunks,
        )

    service = DDCFrameService(
        index,
        args.cache_dir,
        decode,
        decode_staged_frames=decode_staged,
        maximum_cache_files=args.maximum_cache_files,
        prefetch_files=args.prefetch_files,
    )
    with DDCUnixServer(args.socket, service) as server:
        def stop_server(*_: object) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop_server)
        signal.signal(signal.SIGINT, stop_server)
        if args.ready_file is not None:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text(
                json.dumps(
                    {
                        "format": "kpolaris_ddc_service_ready_v2",
                        "native_staged_arrays": True,
                        "socket": str(args.socket),
                        "cache_dir": str(args.cache_dir),
                        "frames": len(index.frames),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        server.serve_forever(poll_interval=0.1)
    if args.ready_file is not None:
        args.ready_file.unlink(missing_ok=True)
    if args.stats_file is not None:
        write_service_statistics(args.stats_file, service)
    print(json.dumps(service.statistics(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
