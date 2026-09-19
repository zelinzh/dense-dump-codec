#!/usr/bin/env python3
"""Repack every archive in a DDC sequence as indexed channel bzip2."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import open_archive, repack_zip_to_channel_bzip2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--archive-jobs", type=int, default=1)
    temporal = parser.add_mutually_exclusive_group()
    temporal.add_argument("--temporal-delta", action="store_true")
    temporal.add_argument("--temporal-delta-shuffle", action="store_true")
    temporal.add_argument("--adaptive-temporal-order", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def repack_sequence(
    manifest_path: Path,
    output_dir: Path,
    output_manifest: Path,
    *,
    chunk_frames: int = 5,
    compression_level: int = 9,
    workers: int = 8,
    archive_jobs: int = 1,
    temporal_delta: bool = False,
    temporal_delta_shuffle: bool = False,
    adaptive_temporal_order: bool = False,
    overwrite: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    if archive_jobs < 1:
        raise ValueError("archive_jobs must be positive")
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_manifest = output_manifest.resolve()
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("format") != "dense_dump_codec_sequence_v1":
        raise ValueError("Unsupported sequence manifest format")
    if len(source_manifest.get("codec_schemes", {})) != 1:
        raise ValueError("Sequence repacking currently requires exactly one codec scheme")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_manifest = copy.deepcopy(source_manifest)
    backend = (
        "channel-bzip2-adaptive"
        if adaptive_temporal_order
        else (
            "channel-bzip2-delta-shuffle"
            if temporal_delta_shuffle
            else ("channel-bzip2-delta" if temporal_delta else "channel-bzip2")
        )
    )
    tasks: list[dict[str, Any]] = []
    for scheme_index, (scheme_name, scheme) in enumerate(
        sorted(source_manifest["codec_schemes"].items())
    ):
        scheme_dir = output_dir / f"scheme_{scheme_index:02d}"
        for archive_index, value in enumerate(scheme["archive_paths"]):
            source = resolve_path(value, manifest_path)
            output = scheme_dir / f"{archive_index:05d}_{source.name}"
            tasks.append(
                {
                    "scheme_name": scheme_name,
                    "archive_index": archive_index,
                    "source": source,
                    "output": output,
                }
            )

    def run_task(task: dict[str, Any]) -> dict[str, Any]:
        output = task["output"]
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        result = repack_zip_to_channel_bzip2(
            task["source"],
            output,
            chunk_frames=chunk_frames,
            compression_level=compression_level,
            workers=workers,
            temporal_delta=temporal_delta,
            temporal_delta_shuffle=temporal_delta_shuffle,
            adaptive_temporal_order=adaptive_temporal_order,
            metadata_updates={
                "compression": backend.replace("-", "_"),
                "compression_level": compression_level,
                "channel_chunk_frames": chunk_frames,
            },
            overwrite=overwrite,
        )
        if verify:
            with open_archive(output) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Archive verification failed at {bad_member}")
        result.update(
            {
                "scheme_name": task["scheme_name"],
                "archive_index": task["archive_index"],
                "verified": verify,
                "sha256": sha256_file(output),
            }
        )
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=archive_jobs) as executor:
        rows = list(executor.map(run_task, tasks))

    rows_by_scheme: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_scheme.setdefault(row["scheme_name"], []).append(row)
    dense_bytes = int(source_manifest["storage"]["dense_phdf_bytes"])
    keyframe_bytes = int(source_manifest["storage"]["keyframe_bytes"])
    middle_frames = int(source_manifest["middle_frame_count"])
    for scheme_name, scheme in result_manifest["codec_schemes"].items():
        scheme_rows = sorted(
            rows_by_scheme[scheme_name], key=lambda row: row["archive_index"]
        )
        archive_bytes = sum(int(row["output_bytes"]) for row in scheme_rows)
        total_bytes = keyframe_bytes + archive_bytes
        scheme.update(
            {
                "archive_backend": backend,
                "channel_chunk_frames": chunk_frames,
                "archive_compression_level": compression_level,
                "archive_paths": [row["output"] for row in scheme_rows],
                "archive_size_bytes": archive_bytes,
                "total_with_keyframes_bytes": total_bytes,
                "ratio_vs_dense_phdf": dense_bytes / max(total_bytes, 1),
                "archive_bytes_per_middle_frame": (
                    archive_bytes / max(middle_frames, 1)
                ),
            }
        )
    source_archive_bytes = sum(int(row["source_bytes"]) for row in rows)
    output_archive_bytes = sum(int(row["output_bytes"]) for row in rows)
    adaptive_candidate_chunk_count = sum(
        int(row.get("adaptive_candidate_chunk_count", 0)) for row in rows
    )
    adaptive_second_order_chunk_count = sum(
        int(row.get("adaptive_second_order_chunk_count", 0)) for row in rows
    )
    adaptive_first_order_stream_bytes = sum(
        int(row.get("adaptive_first_order_stream_bytes") or 0) for row in rows
    )
    adaptive_selected_stream_bytes = sum(
        int(row.get("adaptive_selected_stream_bytes") or 0) for row in rows
    )
    adaptive_saving_vs_first_order_fraction = (
        1.0
        - adaptive_selected_stream_bytes / adaptive_first_order_stream_bytes
        if adaptive_first_order_stream_bytes
        else None
    )
    result_manifest["lossless_repack"] = {
        "source_manifest": str(manifest_path),
        "backend": backend,
        "preconditioner": (
            "adaptive-min-temporal-npy-delta1-delta2-quant-zigzag-byte-shuffle-v1"
            if adaptive_temporal_order
            else (
                "temporal-npy-delta-zigzag-byte-shuffle-v1"
                if temporal_delta_shuffle
                else ("temporal-npy-delta-xor-v1" if temporal_delta else None)
            )
        ),
        "chunk_frames": chunk_frames,
        "compression_level": compression_level,
        "member_crc_verified": verify,
        "archive_count": len(rows),
        "source_archive_bytes": source_archive_bytes,
        "output_archive_bytes": output_archive_bytes,
        "archive_saving_fraction": 1.0
        - output_archive_bytes / source_archive_bytes,
        "adaptive_candidate_chunk_count": adaptive_candidate_chunk_count,
        "adaptive_second_order_chunk_count": adaptive_second_order_chunk_count,
        "adaptive_first_order_stream_bytes": adaptive_first_order_stream_bytes,
        "adaptive_selected_stream_bytes": adaptive_selected_stream_bytes,
        "adaptive_saving_vs_first_order_fraction": (
            adaptive_saving_vs_first_order_fraction
        ),
    }
    integrity = result_manifest.setdefault("integrity", {})
    integrity["archives"] = [
        {
            "path": row["output"],
            "size_bytes": row["output_bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "format": "dense_dump_codec_sequence_repack_v1",
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_manifest),
        "archive_count": len(rows),
        "all_members_crc_verified": verify,
        "backend": backend,
        "source_archive_bytes": source_archive_bytes,
        "output_archive_bytes": output_archive_bytes,
        "archive_saving_fraction": 1.0 - output_archive_bytes / source_archive_bytes,
        "adaptive_candidate_chunk_count": adaptive_candidate_chunk_count,
        "adaptive_second_order_chunk_count": adaptive_second_order_chunk_count,
        "adaptive_first_order_stream_bytes": adaptive_first_order_stream_bytes,
        "adaptive_selected_stream_bytes": adaptive_selected_stream_bytes,
        "adaptive_saving_vs_first_order_fraction": (
            adaptive_saving_vs_first_order_fraction
        ),
        "source_total_with_keyframes_bytes": keyframe_bytes + source_archive_bytes,
        "output_total_with_keyframes_bytes": keyframe_bytes + output_archive_bytes,
        "total_saving_fraction": 1.0
        - (keyframe_bytes + output_archive_bytes)
        / (keyframe_bytes + source_archive_bytes),
        "rows": rows,
    }
    return result


def main() -> None:
    args = parse_args()
    result = repack_sequence(
        args.manifest,
        args.output_dir,
        args.output_manifest,
        chunk_frames=args.chunk_frames,
        compression_level=args.compression_level,
        workers=args.workers,
        archive_jobs=args.archive_jobs,
        temporal_delta=args.temporal_delta,
        temporal_delta_shuffle=args.temporal_delta_shuffle,
        adaptive_temporal_order=args.adaptive_temporal_order,
        overwrite=args.overwrite,
        verify=not args.skip_verify,
    )
    output_json = args.output_json or args.output_dir / "repack_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"archives={result['archive_count']} "
        f"archive_saving={result['archive_saving_fraction']:.3%} "
        f"total_saving={result['total_saving_fraction']:.3%}"
    )
    print(f"manifest: {args.output_manifest}")
    print(f"summary: {output_json}")


if __name__ == "__main__":
    main()
