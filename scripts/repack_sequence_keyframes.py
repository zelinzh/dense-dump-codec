#!/usr/bin/env python3
"""Repack an existing DDC sequence with lossless compressed keyframes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import (  # noqa: E402
    compress_keyframe,
    validate_keyframe_archive,
)
from encode_dense_sequence import delete_keyframe_files, sha256_file  # noqa: E402
from prototype_dump_codec import phdf_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=(
            "bzip2-shuffle",
            "bzip2-shuffle-temporal",
            "bzip2-xz-temporal",
            "bzip2-xz-zigzag-temporal",
        ),
        default="bzip2-shuffle",
    )
    parser.add_argument("--anchor-interval", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-source-keyframes", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def existing_result(
    source: Path,
    archive: Path,
    temporal_order: int,
    reference_paths: tuple[Path, ...],
    compression: str,
    temporal_zigzag: bool,
) -> dict[str, Any] | None:
    result_path = archive.with_name(f"{source.name}.json")
    if not archive.exists() or not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if Path(result.get("source", "")).resolve() != source.resolve():
        return None
    if Path(result.get("output", "")).resolve() != archive.resolve():
        return None
    if int(result.get("source_bytes", -1)) != source.stat().st_size:
        return None
    if int(result.get("output_bytes", -1)) != archive.stat().st_size:
        return None
    if int(result.get("temporal_order", 0)) != temporal_order:
        return None
    if str(result.get("compression", "bzip2")) != compression:
        return None
    if bool(result.get("temporal_zigzag", False)) != temporal_zigzag:
        return None
    if [Path(value).resolve() for value in result.get("reference_paths", [])] != [
        path.resolve() for path in reference_paths
    ]:
        return None
    return result


def temporal_configuration(
    keyframe_index: int,
    keyframes: list[Path],
    backend: str,
    anchor_interval: int,
) -> tuple[int, tuple[Path, ...], int]:
    if backend == "bzip2-shuffle":
        return 0, (), keyframe_index
    anchor_index = keyframe_index - keyframe_index % anchor_interval
    position = keyframe_index - anchor_index
    if position == 0:
        return 0, (), anchor_index
    if position == 1:
        return 1, (keyframes[keyframe_index - 1],), anchor_index
    return (
        2,
        (keyframes[keyframe_index - 1], keyframes[keyframe_index - 2]),
        anchor_index,
    )


def repack_manifest(
    manifest_path: Path,
    output_manifest: Path,
    archive_dir: Path,
    *,
    compression_level: int = 9,
    workers: int = 1,
    overwrite: bool = False,
    delete_source_keyframes: bool = False,
    sha256_enabled: bool = True,
    backend: str = "bzip2-shuffle",
    anchor_interval: int = 4,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_manifest = output_manifest.resolve()
    archive_dir = archive_dir.resolve()
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(output_manifest)
    if backend not in (
        "bzip2-shuffle",
        "bzip2-shuffle-temporal",
        "bzip2-xz-temporal",
        "bzip2-xz-zigzag-temporal",
    ):
        raise ValueError(f"Unsupported keyframe backend {backend!r}")
    if anchor_interval < 2:
        raise ValueError("anchor_interval must be at least 2")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = tuple(manifest["datasets"])
    keyframes = [Path(value).resolve() for value in manifest["keyframes"]]
    archive_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for keyframe_index, source in enumerate(keyframes):
        temporal_order, reference_paths, anchor_index = temporal_configuration(
            keyframe_index,
            keyframes,
            backend,
            anchor_interval,
        )
        xz_temporal = backend in (
            "bzip2-xz-temporal",
            "bzip2-xz-zigzag-temporal",
        ) and bool(temporal_order)
        compression = "xz" if xz_temporal else "bzip2"
        temporal_zigzag = backend == "bzip2-xz-zigzag-temporal" and bool(
            temporal_order
        )
        archive = archive_dir / f"{source.name}.ddckf"
        result = existing_result(
            source,
            archive,
            temporal_order,
            reference_paths,
            compression,
            temporal_zigzag,
        )
        if result is None:
            result = compress_keyframe(
                source,
                archive,
                datasets,
                compression_level=compression_level,
                workers=workers,
                overwrite=archive.exists(),
                reference_paths=reference_paths,
                temporal_order=temporal_order,
                compression=compression,
                temporal_zigzag=temporal_zigzag,
            )
            result_path = archive.with_name(f"{source.name}.json")
            temporary_result = result_path.with_name(f".{result_path.name}.tmp")
            temporary_result.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_result.replace(result_path)
        validation = validate_keyframe_archive(
            archive,
            reference_paths=reference_paths,
        )
        reference_sequences = [phdf_sequence(path) for path in reference_paths]
        rows.append(
            {
                "sequence": phdf_sequence(source),
                "source_path": str(source),
                "archive_path": str(archive),
                "source_bytes": int(result["source_bytes"]),
                "archive_bytes": int(result["output_bytes"]),
                "saving_fraction": float(result["saving_vs_source_fraction"]),
                "format": result["format"],
                "chunk_count": validation["chunk_count"],
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "temporal_order": temporal_order,
                "reference_sequences": reference_sequences,
                "anchor_sequence": phdf_sequence(keyframes[anchor_index]),
                "target_crc_verified": bool(validation["target_crc_verified"]),
                "compression": compression,
                "temporal_zigzag": temporal_zigzag,
            }
        )

    source_bytes = sum(int(row["source_bytes"]) for row in rows)
    stored_bytes = sum(int(row["archive_bytes"]) for row in rows)
    manifest["keyframe_storage"] = {
        "backend": backend,
        "anchor_interval": anchor_interval if backend != "bzip2-shuffle" else 1,
        "maximum_dependency_chain": (
            anchor_interval - 1 if backend != "bzip2-shuffle" else 0
        ),
        "source_bytes": source_bytes,
        "stored_bytes": stored_bytes,
        "saving_fraction": 1.0 - stored_bytes / source_bytes,
        "archives": rows,
    }
    manifest.setdefault("storage", {})["source_keyframe_bytes"] = source_bytes
    manifest["storage"]["keyframe_bytes"] = stored_bytes
    dense_bytes = int(manifest["storage"]["dense_phdf_bytes"])
    for scheme in manifest["codec_schemes"].values():
        total_bytes = int(scheme["archive_size_bytes"]) + stored_bytes
        scheme["keyframe_backend"] = backend
        scheme["keyframe_size_bytes"] = stored_bytes
        scheme["total_with_keyframes_bytes"] = total_bytes
        scheme["ratio_vs_dense_phdf"] = dense_bytes / total_bytes
    manifest["lossless_keyframe_repack"] = {
        "format": "dense_dump_codec_sequence_keyframe_repack_v1",
        "source_manifest": str(manifest_path),
        "archive_dir": str(archive_dir),
        "source_keyframe_bytes": source_bytes,
        "output_keyframe_bytes": stored_bytes,
        "saving_fraction": 1.0 - stored_bytes / source_bytes,
        "all_chunks_verified": True,
        "backend": backend,
        "anchor_interval": anchor_interval if backend != "bzip2-shuffle" else 1,
    }
    manifest["keyframes_deleted"] = False
    manifest["deleted_keyframe_files"] = []
    integrity = manifest.setdefault("integrity", {})
    integrity["sha256_enabled"] = sha256_enabled
    integrity["keyframe_archives"] = (
        [
            {
                "path": row["archive_path"],
                "size_bytes": row["archive_bytes"],
                "sha256": sha256_file(Path(row["archive_path"])),
            }
            for row in rows
        ]
        if sha256_enabled
        else []
    )
    if delete_source_keyframes:
        manifest["deleted_keyframe_files"] = delete_keyframe_files(keyframes)
        manifest["keyframes_deleted"] = True
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_manifest.with_name(f".{output_manifest.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_manifest)
    return {
        "format": "dense_dump_codec_sequence_keyframe_repack_v1",
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_manifest),
        "archive_dir": str(archive_dir),
        "keyframe_count": len(rows),
        "source_keyframe_bytes": source_bytes,
        "output_keyframe_bytes": stored_bytes,
        "saving_fraction": 1.0 - stored_bytes / source_bytes,
        "all_chunks_verified": True,
        "keyframes_deleted": delete_source_keyframes,
        "backend": backend,
        "anchor_interval": anchor_interval if backend != "bzip2-shuffle" else 1,
    }


def main() -> None:
    args = parse_args()
    suffix = args.backend.replace("-", "_")
    output_manifest = args.output_manifest or args.manifest.with_name(
        f"sequence_manifest_keyframe_{suffix}.json"
    )
    archive_dir = args.archive_dir or args.manifest.parent / f"keyframes_{suffix}"
    result = repack_manifest(
        args.manifest,
        output_manifest,
        archive_dir,
        compression_level=args.compression_level,
        workers=args.workers,
        overwrite=args.overwrite,
        delete_source_keyframes=args.delete_source_keyframes,
        sha256_enabled=not args.skip_sha256,
        backend=args.backend,
        anchor_interval=args.anchor_interval,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
