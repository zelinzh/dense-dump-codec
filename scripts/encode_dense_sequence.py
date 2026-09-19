#!/usr/bin/env python3
"""Encode a PHDF sequence as independently decodable dense-dump codec GOPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import (  # noqa: E402
    compress_keyframe,
    open_archive,
    validate_keyframe_archive,
)

from prototype_dump_codec import (  # noqa: E402
    DEFAULT_DATASETS,
    evaluate_codec,
    file_size,
    parse_bits,
    parse_datasets,
    phdf_sequence,
    phdf_time,
    sequence_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        required=True,
        help="Number of dense output steps between adjacent keyframes.",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--bits", default="6,8")
    parser.add_argument("--dataset-bits", default="")
    parser.add_argument("--dataset-tile-shapes", default="",
                        help="Local scales, e.g. prims.u=8x16x32 in phi,theta,r order")
    parser.add_argument("--scale-mode", choices=("frame", "block-channel"), default="block-channel")
    parser.add_argument("--scale-percentile", type=float, default=99.9)
    parser.add_argument("--dataset-scale-percentiles", default="")
    parser.add_argument("--discard-outliers", action="store_true")
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument(
        "--archive-backend",
        choices=(
            "zip-deflate",
            "channel-bzip2",
            "channel-bzip2-delta",
            "channel-bzip2-delta-shuffle",
            "channel-bzip2-adaptive",
        ),
        default="zip-deflate",
    )
    parser.add_argument("--channel-chunk-frames", type=int, default=5)
    parser.add_argument("--channel-compression-level", type=int, default=9)
    parser.add_argument("--archive-workers", type=int, default=1)
    parser.add_argument(
        "--keyframe-backend",
        choices=(
            "raw",
            "bzip2-shuffle",
            "bzip2-shuffle-temporal",
            "bzip2-xz-temporal",
            "bzip2-xz-zigzag-temporal",
        ),
        default="raw",
    )
    parser.add_argument("--keyframe-compression-level", type=int, default=9)
    parser.add_argument("--keyframe-workers", type=int, default=1)
    parser.add_argument("--keyframe-anchor-interval", type=int, default=4)
    parser.add_argument("--allow-short-final-gop", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--stream-wait-seconds", type=float, default=1800.0)
    parser.add_argument("--ignore-stream-state", action="store_true")
    parser.add_argument(
        "--delete-middle-frames",
        action="store_true",
        help="Delete encoded non-key PHDF/XDMF files only after every archive validates.",
    )
    parser.add_argument(
        "--delete-keyframes-after-encode",
        action="store_true",
        help="Delete raw keyframe PHDF/XDMF files after validating compressed keyframes.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def aggregate_error_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(row["count"]) for row in rows)
    sse = sum(float(row["sse"]) for row in rows)
    target_sse = sum(float(row["target_sse"]) for row in rows)
    sae = sum(float(row["sae"]) for row in rows)
    target_abs = sum(float(row["target_abs"]) for row in rows)
    rmse = math.sqrt(sse / max(count, 1))
    target_rms = math.sqrt(target_sse / max(count, 1))
    return {
        "count": count,
        "rmse": rmse,
        "target_rms": target_rms,
        "nrmse": rmse / max(target_rms, 1.0e-30),
        "rel_l1": sae / max(target_abs, 1.0e-30),
        "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
        "min_reconstructed_value": min(float(row["min_reconstructed_value"]) for row in rows),
        "sse": sse,
        "target_sse": target_sse,
        "sae": sae,
        "target_abs": target_abs,
    }


def aggregate_named_errors(
    summaries: list[dict[str, Any]],
    section: str,
    subsection: str,
) -> dict[str, Any]:
    names = sorted(
        {
            name
            for summary in summaries
            for name in summary[section][subsection]
        }
    )
    return {
        name: aggregate_error_rows(
            [summary[section][subsection][name] for summary in summaries if name in summary[section][subsection]]
        )
        for name in names
    }


def build_gop_ranges(file_count: int, stride: int, allow_short_final: bool) -> list[tuple[int, int]]:
    if stride < 2:
        raise ValueError("--keyframe-stride must be at least 2")
    if file_count < 3:
        raise ValueError("A dense sequence needs at least three frames")
    remainder = (file_count - 1) % stride
    if remainder == 1 and allow_short_final:
        raise ValueError("Final GOP has no intermediate frame; choose a stride that covers every state")
    if remainder and not allow_short_final:
        raise ValueError(
            f"{file_count} frames do not form complete stride-{stride} GOPs; "
            "pass --allow-short-final-gop to encode the remainder"
        )
    ranges = []
    for start in range(0, file_count - 1, stride):
        end = min(start + stride, file_count - 1)
        if end - start >= 2:
            ranges.append((start, end))
    return ranges


def validate_archives(codec_schemes: dict[str, Any]) -> None:
    for scheme in codec_schemes.values():
        for value in scheme["archive_paths"]:
            with open_archive(value) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"CRC failure in {value}: {bad_member}")
                if "metadata.json" not in archive.namelist():
                    raise ValueError(f"Missing metadata.json in {value}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_streaming_manifest(output_dir: Path, timeout_seconds: float) -> bool:
    state_path = output_dir / "stream_state.json"
    if not state_path.exists():
        return False
    if timeout_seconds < 0:
        raise ValueError("--stream-wait-seconds must be non-negative")
    manifest_path = output_dir / "sequence_manifest.json"
    deadline = time.monotonic() + timeout_seconds
    while True:
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
            if (
                manifest.get("format") == "dense_dump_codec_sequence_v1"
                and manifest.get("complete") is True
            ):
                return True
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Streaming codec state exists but did not finalize within {timeout_seconds}s"
            )
        time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))


def delete_middle_frames(files: list[Path], keyframes: dict[str, Path]) -> list[str]:
    keyframe_paths = {path.resolve() for path in keyframes.values()}
    deleted: list[str] = []
    for path in files:
        if path.resolve() in keyframe_paths:
            continue
        path.unlink()
        deleted.append(str(path))
        xdmf_path = Path(f"{path}.xdmf")
        if xdmf_path.exists():
            xdmf_path.unlink()
    return deleted


def compress_sequence_keyframes(
    keyframes: list[Path],
    output_dir: Path,
    datasets: tuple[str, ...],
    *,
    backend: str,
    compression_level: int,
    workers: int,
    anchor_interval: int = 4,
) -> tuple[list[dict[str, Any]], int]:
    if backend == "raw":
        return [], sum(file_size(path) for path in keyframes)
    if backend not in (
        "bzip2-shuffle",
        "bzip2-shuffle-temporal",
        "bzip2-xz-temporal",
        "bzip2-xz-zigzag-temporal",
    ):
        raise ValueError(f"Unsupported keyframe backend {backend!r}")
    if anchor_interval < 2:
        raise ValueError("keyframe anchor interval must be at least 2")
    archive_dir = output_dir / "keyframes"
    archive_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for keyframe_index, path in enumerate(keyframes):
        position = keyframe_index % anchor_interval
        temporal_order = (
            0
            if backend == "bzip2-shuffle" or position == 0
            else 1 if position == 1 else 2
        )
        reference_paths = tuple(
            keyframes[keyframe_index - offset]
            for offset in range(1, temporal_order + 1)
        )
        xz_temporal = backend in (
            "bzip2-xz-temporal",
            "bzip2-xz-zigzag-temporal",
        ) and bool(temporal_order)
        compression = "xz" if xz_temporal else "bzip2"
        temporal_zigzag = backend == "bzip2-xz-zigzag-temporal" and bool(
            temporal_order
        )
        anchor_index = keyframe_index - position if temporal_order else keyframe_index
        archive_path = archive_dir / f"{path.name}.ddckf"
        result = compress_keyframe(
            path,
            archive_path,
            datasets,
            compression_level=compression_level,
            workers=workers,
            overwrite=True,
            reference_paths=reference_paths,
            temporal_order=temporal_order,
            compression=compression,
            temporal_zigzag=temporal_zigzag,
        )
        validation = validate_keyframe_archive(
            archive_path,
            reference_paths=reference_paths,
        )
        rows.append(
            {
                "sequence": phdf_sequence(path),
                "source_path": str(path),
                "archive_path": str(archive_path),
                "source_bytes": result["source_bytes"],
                "archive_bytes": result["output_bytes"],
                "saving_fraction": result["saving_vs_source_fraction"],
                "format": result["format"],
                "chunk_count": validation["chunk_count"],
                "elapsed_seconds": result["elapsed_seconds"],
                "temporal_order": temporal_order,
                "reference_sequences": [
                    phdf_sequence(reference) for reference in reference_paths
                ],
                "anchor_sequence": phdf_sequence(keyframes[anchor_index]),
                "target_crc_verified": bool(validation["target_crc_verified"]),
                "compression": compression,
                "temporal_zigzag": temporal_zigzag,
            }
        )
    return rows, sum(int(row["archive_bytes"]) for row in rows)


def delete_keyframe_files(keyframes: list[Path]) -> list[str]:
    deleted = []
    for path in keyframes:
        path.unlink()
        deleted.append(str(path))
        Path(f"{path}.xdmf").unlink(missing_ok=True)
    return deleted


def main() -> int:
    wall_started = time.time()
    args = parse_args()
    args.segment_dir = args.segment_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.ignore_stream_state and wait_for_streaming_manifest(
        args.output_dir, args.stream_wait_seconds
    ):
        print(f"reuse finalized streaming manifest: {args.output_dir / 'sequence_manifest.json'}")
        return 0
    datasets = parse_datasets(args.datasets)
    bits_values = parse_bits(args.bits)
    files = sequence_files(args.segment_dir)
    gop_ranges = build_gop_ranges(len(files), args.keyframe_stride, args.allow_short_final_gop)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    keyframes: dict[str, Path] = {}
    for start_index, end_index in gop_ranges:
        start_file = files[start_index]
        end_file = files[end_index]
        start_sequence = phdf_sequence(start_file)
        end_sequence = phdf_sequence(end_file)
        gop_dir = args.output_dir / "gops" / f"{start_sequence:05d}_{end_sequence:05d}"
        codec_args = argparse.Namespace(
            segment_dir=args.segment_dir,
            output_dir=gop_dir,
            datasets=args.datasets,
            bits=args.bits,
            dataset_bits=args.dataset_bits,
            dataset_tile_shapes=getattr(args, "dataset_tile_shapes", ""),
            dataset_scale_percentiles=args.dataset_scale_percentiles,
            scale_mode=args.scale_mode,
            scale_percentile=args.scale_percentile,
            discard_outliers=args.discard_outliers,
            compression_level=args.compression_level,
            archive_backend=args.archive_backend,
            channel_chunk_frames=args.channel_chunk_frames,
            channel_compression_level=args.channel_compression_level,
            archive_workers=args.archive_workers,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            max_middle_frames=0,
            output_json=None,
            skip_archives=False,
        )
        summaries.append(evaluate_codec(codec_args))
        keyframes[str(start_file)] = start_file
        keyframes[str(end_file)] = end_file

    raw_size = sum(file_size(path) for path in files)
    frame_records = [
        {
            "sequence": phdf_sequence(path),
            "time": phdf_time(path),
            "path": str(path),
            "size_bytes": file_size(path),
            "exact_keyframe": str(path) in keyframes,
        }
        for path in files
    ]
    keyframe_paths = list(keyframes.values())
    source_keyframe_size = sum(file_size(path) for path in keyframe_paths)
    keyframe_archives, stored_keyframe_size = compress_sequence_keyframes(
        keyframe_paths,
        args.output_dir,
        datasets,
        backend=args.keyframe_backend,
        compression_level=args.keyframe_compression_level,
        workers=args.keyframe_workers,
        anchor_interval=args.keyframe_anchor_interval,
    )
    middle_frame_count = sum(summary["selected_middle_file_count"] for summary in summaries)
    scheme_names = sorted(summaries[0]["codec_schemes"])
    codec_schemes: dict[str, Any] = {}
    for scheme_name in scheme_names:
        scheme_rows = [summary["codec_schemes"][scheme_name] for summary in summaries]
        archive_size = sum(int(row["archive_size_bytes"]) for row in scheme_rows)
        total_size = stored_keyframe_size + archive_size
        wrapped = [{"scheme": row} for row in scheme_rows]
        codec_schemes[scheme_name] = {
            "bits": int(scheme_rows[0]["bits"]),
            "dataset_tile_shapes": dict(scheme_rows[0].get("dataset_tile_shapes", {})),
            "dataset_bits": dict(scheme_rows[0].get("dataset_bits", {})),
            "scale_percentile": float(scheme_rows[0]["scale_percentile"]),
            "dataset_scale_percentiles": dict(
                scheme_rows[0].get("dataset_scale_percentiles", {})
            ),
            "archive_backend": scheme_rows[0].get(
                "archive_backend", "zip-deflate"
            ),
            "channel_chunk_frames": scheme_rows[0].get("channel_chunk_frames"),
            "archive_compression_level": scheme_rows[0].get(
                "archive_compression_level"
            ),
            "preserve_outliers": bool(scheme_rows[0]["preserve_outliers"]),
            "exception_count": sum(int(row["exception_count"]) for row in scheme_rows),
            "archive_size_bytes": archive_size,
            "keyframe_backend": args.keyframe_backend,
            "keyframe_size_bytes": stored_keyframe_size,
            "total_with_keyframes_bytes": total_size,
            "ratio_vs_dense_phdf": raw_size / max(total_size, 1),
            "archive_bytes_per_middle_frame": archive_size / max(middle_frame_count, 1),
            "archive_paths": [row["archive_path"] for row in scheme_rows],
            "errors": aggregate_named_errors(wrapped, "scheme", "errors"),
            "derived_errors": aggregate_named_errors(wrapped, "scheme", "derived_errors"),
        }

    manifest = {
        "format": "dense_dump_codec_sequence_v1",
        "complete": True,
        "segment_dir": str(args.segment_dir),
        "datasets": datasets,
        "dense_frame_count": len(files),
        "dense_files": [str(path) for path in files],
        "frames": frame_records,
        "keyframe_stride": args.keyframe_stride,
        "keyframe_count": len(keyframes),
        "keyframes": list(keyframes),
        "keyframe_storage": {
            "backend": args.keyframe_backend,
            "anchor_interval": (
                args.keyframe_anchor_interval
                if args.keyframe_backend
                in (
                    "bzip2-shuffle-temporal",
                    "bzip2-xz-temporal",
                    "bzip2-xz-zigzag-temporal",
                )
                else 1
            ),
            "maximum_dependency_chain": (
                args.keyframe_anchor_interval - 1
                if args.keyframe_backend
                in (
                    "bzip2-shuffle-temporal",
                    "bzip2-xz-temporal",
                    "bzip2-xz-zigzag-temporal",
                )
                else 0
            ),
            "source_bytes": source_keyframe_size,
            "stored_bytes": stored_keyframe_size,
            "saving_fraction": 1.0 - stored_keyframe_size / source_keyframe_size,
            "archives": keyframe_archives,
        },
        "gop_count": len(summaries),
        "middle_frame_count": middle_frame_count,
        "storage": {
            "dense_phdf_bytes": raw_size,
            "source_keyframe_bytes": source_keyframe_size,
            "keyframe_bytes": stored_keyframe_size,
        },
        "codec_schemes": codec_schemes,
        "gop_summaries": [summary["output_json"] for summary in summaries],
        "middle_frames_deleted": False,
        "deleted_middle_files": [],
        "keyframes_deleted": False,
        "deleted_keyframe_files": [],
    }
    output_json = args.output_json or (args.output_dir / "sequence_manifest.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    validate_archives(codec_schemes)
    if args.skip_sha256:
        manifest["integrity"] = {"sha256_enabled": False}
    else:
        archive_paths = sorted(
            {
                Path(value)
                for scheme in codec_schemes.values()
                for value in scheme["archive_paths"]
            }
        )
        manifest["integrity"] = {
            "sha256_enabled": True,
            "keyframes": [
                {
                    "path": str(path),
                    "size_bytes": file_size(path),
                    "sha256": sha256_file(path),
                }
                for path in keyframe_paths
            ],
            "keyframe_archives": [
                {
                    "path": row["archive_path"],
                    "size_bytes": row["archive_bytes"],
                    "sha256": sha256_file(Path(row["archive_path"])),
                }
                for row in keyframe_archives
            ],
            "archives": [
                {
                    "path": str(path),
                    "size_bytes": file_size(path),
                    "sha256": sha256_file(path),
                }
                for path in archive_paths
            ],
        }
    if args.delete_middle_frames:
        manifest["deleted_middle_files"] = delete_middle_frames(files, keyframes)
        manifest["middle_frames_deleted"] = True
    if args.delete_keyframes_after_encode:
        if args.keyframe_backend == "raw":
            raise ValueError(
                "--delete-keyframes-after-encode requires a compressed keyframe backend"
            )
        manifest["deleted_keyframe_files"] = delete_keyframe_files(keyframe_paths)
        manifest["keyframes_deleted"] = True
    elapsed_seconds = time.time() - wall_started
    manifest["performance"] = {
        "elapsed_seconds": elapsed_seconds,
        "gop_encoding_seconds": sum(float(summary["elapsed_seconds"]) for summary in summaries),
        "keyframe_encoding_seconds": sum(
            float(row["elapsed_seconds"]) for row in keyframe_archives
        ),
        "dense_input_bytes_per_second": raw_size / max(elapsed_seconds, 1.0e-30),
        "dense_input_mib_per_second": raw_size / 1048576.0 / max(elapsed_seconds, 1.0e-30),
        "middle_frames_per_second": middle_frame_count / max(elapsed_seconds, 1.0e-30),
    }
    output_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"sequence: {args.segment_dir}")
    print(f"frames={len(files)} keyframes={len(keyframes)} gops={len(summaries)}")
    for name, scheme in codec_schemes.items():
        print(
            f"{name}: ratio={scheme['ratio_vs_dense_phdf']:.3f}x "
            f"total={scheme['total_with_keyframes_bytes'] / 1048576:.1f} MiB "
            f"rho={scheme['errors'].get('rho', {}).get('nrmse', float('nan')):.4g} "
            f"B1={scheme['errors'].get('B1', {}).get('nrmse', float('nan')):.4g}"
        )
    print(f"manifest: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
