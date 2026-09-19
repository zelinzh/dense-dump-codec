#!/usr/bin/env python3
"""Serve native DDC arrays using bounded channel workers and per-frame publication."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dense_dump_codec.kpolaris_service import (
    DDCUnixServer, _native_wire_parts, write_service_statistics,
)
from dense_dump_codec.native import NativeSequenceDecoder, native_capabilities
from dense_dump_codec.sequence import DenseSequenceIndex
from dense_dump_codec.streaming import StreamingFrameService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scheme")
    parser.add_argument("--datasets", default="prims.rho,prims.u,prims.uvec,prims.B")
    parser.add_argument("--maximum-cache-files", type=int, default=32)
    parser.add_argument("--prefetch-files", type=int, default=25)
    parser.add_argument("--native-decode-workers", type=int, default=16)
    parser.add_argument("--native-archive-cache-chunks", type=int, default=8)
    parser.add_argument("--native-working-cache", type=Path)
    parser.add_argument("--native-predictor-library", type=Path)
    parser.add_argument("--native-radius-max", type=float,
                        help="Opt-in radial prefix with halo; must cover the GRRT radiative region")
    parser.add_argument("--native-reconstruction", choices=("auto", "reference", "numpy"),
                        default="auto")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--stats-file", type=Path)
    args = parser.parse_args()
    if set(args.datasets.split(",")) != set(NativeSequenceDecoder.datasets):
        parser.error("The native service requires all four primitive datasets")
    compact = os.environ.get("KPOLARIS_DDC_COMPACT") == "1"
    if compact:
        from dense_dump_codec.compact import CompactSequenceDecoder, compact_wire_parts
    decoder_type = CompactSequenceDecoder if compact else NativeSequenceDecoder
    decoder = decoder_type(
        args.manifest, workers=args.native_decode_workers,
        cache_chunks=args.native_archive_cache_chunks, scheme=args.scheme,
        reconstruction=args.native_reconstruction, maximum_batch_frames=args.prefetch_files,
        radial_max=args.native_radius_max, working_cache=args.native_working_cache,
        predictor_library=args.native_predictor_library,
    )
    service = StreamingFrameService(
        DenseSequenceIndex.from_manifest(args.manifest), args.cache_dir, decoder,
        maximum_cache_files=args.maximum_cache_files, prefetch_files=args.prefetch_files,
        validate_frame=compact_wire_parts if compact else _native_wire_parts,
    )
    try:
        with DDCUnixServer(args.socket, service) as server:
            def stop(*_):
                threading.Thread(target=server.shutdown, daemon=True).start()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            if args.ready_file:
                args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                args.ready_file.write_text(json.dumps({
                    "format": "ddc_native_ready_v1", "native_staged_arrays": True,
                    "socket": str(args.socket), **native_capabilities(),
                    "requested_radius_max": args.native_radius_max,
                    "compact_transport": compact,
                    "wire_protocol": "STAGEQ1" if compact else "STAGE1",
                }) + "\n")
            try:
                server.serve_forever(poll_interval=0.1)
            finally:
                service.close()
    finally:
        service.close()
        if args.ready_file:
            args.ready_file.unlink(missing_ok=True)
        if args.stats_file:
            write_service_statistics(args.stats_file, service)
    print(json.dumps(service.statistics(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
