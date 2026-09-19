#!/usr/bin/env python3
"""Incrementally encode completed PHDF GOPs while KHARMA continues running."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec.checkpoint import prune_native_checkpoints  # noqa: E402
from dense_dump_codec.tiled import parse_dataset_tiles

from encode_dense_sequence import (  # noqa: E402
    aggregate_named_errors,
    compress_sequence_keyframes,
    delete_keyframe_files,
    sha256_file,
    validate_archives,
)
from prototype_dump_codec import (  # noqa: E402
    DEFAULT_DATASETS,
    evaluate_codec,
    file_size,
    parse_bits,
    parse_dataset_bits,
    parse_dataset_scale_percentiles,
    parse_datasets,
    phdf_sequence,
    phdf_time,
    require_h5py,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frame-count", type=int, required=True)
    parser.add_argument("--keyframe-stride", type=int, required=True)
    parser.add_argument("--start-sequence", type=int, default=0)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--bits", default="8")
    parser.add_argument("--dataset-bits", default="")
    parser.add_argument("--dataset-tile-shapes", default="",
                        help="Local scales, e.g. prims.u=8x16x32 in phi,theta,r order")
    parser.add_argument("--dataset-scale-percentiles", default="")
    parser.add_argument("--scale-mode", choices=("frame", "block-channel"), default="block-channel")
    parser.add_argument("--scale-percentile", type=float, default=99.9)
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
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--delete-middle-frames", action="store_true")
    parser.add_argument("--delete-keyframes-after-encode", action="store_true")
    parser.add_argument("--allow-short-final-gop", action="store_true")
    parser.add_argument("--retain-every", type=int)
    parser.add_argument("--retain-dir", type=Path)
    parser.add_argument("--truth-dir", type=Path)
    parser.add_argument("--truth-window-json", action="append", default=[])
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=0,
        help="Validate, hash, and retain only this many periodic native checkpoints; 0 disables.",
    )
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument(
        "--repair-source-permissions-after-final-checkpoint",
        action="store_true",
        help=(
            "After a native *.final.rhdf exists, add owner-read permission only to "
            "the pending GOP and its successor before readiness checks."
        ),
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def parse_truth_windows(values: list[str]) -> list[dict[str, Any]]:
    windows = []
    for value in values:
        row = json.loads(value)
        if not isinstance(row, dict):
            raise ValueError("--truth-window-json must decode to an object")
        window_id = str(row.get("window_id", "")).strip()
        start = float(row["start"])
        end = float(row["end"])
        if not window_id or end < start:
            raise ValueError("Truth windows require a non-empty id and end >= start")
        windows.append({"window_id": window_id, "start": start, "end": end})
    return windows


def configured_truth_windows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if hasattr(args, "parsed_truth_windows"):
        return args.parsed_truth_windows
    return parse_truth_windows(args.truth_window_json)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o644)
    temporary.replace(path)
    path.chmod(0o644)


def ensure_owner_readable(path: Path) -> bool:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & stat.S_IRUSR:
        return False
    path.chmod(mode | 0o644)
    return True


def sha256_readable_file(path: Path, *, attempts: int = 5) -> str:
    for attempt in range(attempts):
        try:
            ensure_owner_readable(path)
            return sha256_file(path)
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.2)
    raise AssertionError("unreachable")


def repair_unreadable_codec_artifacts(output_dir: Path) -> list[str]:
    repaired = []
    for path in output_dir.rglob("*"):
        if path.is_file() and ensure_owner_readable(path):
            repaired.append(str(path))
    return repaired


def repair_completed_artifacts(args: argparse.Namespace) -> list[str]:
    roots = [args.output_dir, args.segment_dir, args.retain_dir, args.truth_dir]
    repaired: list[str] = []
    visited: set[Path] = set()
    for root in roots:
        if root is None:
            continue
        resolved = Path(root).resolve()
        if resolved in visited or not resolved.exists():
            continue
        visited.add(resolved)
        repaired.extend(repair_unreadable_codec_artifacts(resolved))
    return repaired


def read_codec_summary(path: Path) -> dict[str, Any]:
    ensure_owner_readable(path)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except PermissionError:
        path.chmod(0o644)
        summary = json.loads(path.read_text(encoding="utf-8"))
    for scheme in summary.get("codec_schemes", {}).values():
        archive_paths = list(scheme.get("archive_paths", []))
        if scheme.get("archive_path"):
            archive_paths.append(scheme["archive_path"])
        for archive_value in archive_paths:
            archive_path = Path(archive_value)
            if not archive_path.is_absolute():
                archive_path = path.parent / archive_path
            if archive_path.is_file():
                ensure_owner_readable(archive_path)
    return summary


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    datasets = parse_datasets(args.datasets)
    result = {
        "segment_dir": str(args.segment_dir),
        "output_dir": str(args.output_dir),
        "expected_frame_count": args.expected_frame_count,
        "keyframe_stride": args.keyframe_stride,
        "start_sequence": args.start_sequence,
        "datasets": list(datasets),
        "bits": list(parse_bits(args.bits)),
        "dataset_bits": parse_dataset_bits(args.dataset_bits, datasets),
        "dataset_scale_percentiles": parse_dataset_scale_percentiles(
            args.dataset_scale_percentiles,
            datasets,
        ),
        "scale_mode": args.scale_mode,
        "scale_percentile": args.scale_percentile,
        "preserve_outliers": not args.discard_outliers,
        "compression_level": args.compression_level,
        "archive_backend": args.archive_backend,
        "channel_chunk_frames": args.channel_chunk_frames,
        "channel_compression_level": args.channel_compression_level,
        "archive_workers": args.archive_workers,
        "keyframe_backend": args.keyframe_backend,
        "keyframe_compression_level": args.keyframe_compression_level,
        "keyframe_workers": args.keyframe_workers,
        "delete_middle_frames": args.delete_middle_frames,
        "delete_keyframes_after_encode": args.delete_keyframes_after_encode,
        "retention": {
            "retain_every": args.retain_every,
            "retain_dir": str(args.retain_dir) if args.retain_dir is not None else None,
            "truth_dir": str(args.truth_dir) if args.truth_dir is not None else None,
            "truth_windows": configured_truth_windows(args),
        },
        "checkpoint_keep": args.checkpoint_keep,
    }
    tiles = parse_dataset_tiles(getattr(args, "dataset_tile_shapes", ""), datasets)
    if tiles:
        result["dataset_tile_shapes"] = {name: list(shape) for name, shape in tiles.items()}
    if args.keyframe_backend in (
        "bzip2-shuffle-temporal",
        "bzip2-xz-temporal",
        "bzip2-xz-zigzag-temporal",
    ):
        result["keyframe_anchor_interval"] = args.keyframe_anchor_interval
    return result


def safe_configuration_extension(
    previous: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    old = json.loads(json.dumps(previous))
    new = json.loads(json.dumps(expected))
    old_count = int(old.pop("expected_frame_count"))
    new_count = int(new.pop("expected_frame_count"))
    old_windows = old["retention"].pop("truth_windows")
    new_windows = new["retention"].pop("truth_windows")
    if old != new or new_count <= old_count:
        return None
    old_by_id = {window["window_id"]: window for window in old_windows}
    new_by_id = {window["window_id"]: window for window in new_windows}
    if old_by_id.keys() != new_by_id.keys():
        return None
    for window_id, old_window in old_by_id.items():
        new_window = new_by_id[window_id]
        if (
            float(new_window["start"]) != float(old_window["start"])
            or float(new_window["end"]) < float(old_window["end"])
        ):
            return None
    return {
        "migrated_at_unix": time.time(),
        "reason": "extended_geodesic_time_support",
        "previous_expected_frame_count": old_count,
        "expected_frame_count": new_count,
        "previous_truth_windows": old_windows,
        "truth_windows": new_windows,
    }


def load_state(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    expected_configuration = configuration(args)
    if not path.exists():
        return {
            "format": "dense_dump_codec_stream_state_v1",
            "configuration": expected_configuration,
            "started_at_unix": time.time(),
            "gops": [],
            "frames": {},
            "deleted_middle_files": [],
            "retained_raw_files": {},
            "finalized": False,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("configuration") != expected_configuration:
        migration = safe_configuration_extension(
            state.get("configuration", {}),
            expected_configuration,
        )
        if migration is None or state.get("finalized"):
            raise ValueError("Streaming codec configuration does not match the existing state")
        state.setdefault("configuration_migrations", []).append(migration)
        state["configuration"] = expected_configuration
    return state


def expected_last_sequence(args: argparse.Namespace) -> int:
    if args.expected_frame_count < 3:
        raise ValueError("--expected-frame-count must be at least 3")
    if args.keyframe_stride < 2:
        raise ValueError("--keyframe-stride must be at least 2")
    intervals = args.expected_frame_count - 1
    if intervals % args.keyframe_stride and not args.allow_short_final_gop:
        raise ValueError(
            "Expected frame count does not form complete GOPs; "
            "pass --allow-short-final-gop"
        )
    return args.start_sequence + intervals


def next_gop(state: dict[str, Any], args: argparse.Namespace) -> tuple[int, int] | None:
    final_sequence = expected_last_sequence(args)
    start = (
        int(state["gops"][-1]["end_sequence"])
        if state["gops"]
        else args.start_sequence
    )
    if start >= final_sequence:
        return None
    end = min(start + args.keyframe_stride, final_sequence)
    if end - start < 2:
        raise ValueError(f"Final GOP {start}:{end} has no middle frame")
    return start, end


def current_files(segment_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in segment_dir.glob("*.phdf"):
        try:
            sequence = phdf_sequence(path)
        except ValueError:
            continue
        files[sequence] = path.resolve()
    return files


def gop_ready(files: dict[int, Path], start: int, end: int) -> bool:
    required = [files.get(sequence) for sequence in range(start, end + 1)]
    if any(path is None for path in required):
        return False
    h5py = require_h5py()
    try:
        for path in required:
            if not stat.S_IMODE(path.stat().st_mode) & stat.S_IRUSR:
                return False
            with h5py.File(path, "r") as handle:
                if "Info" not in handle or "Time" not in handle["Info"].attrs:
                    return False
    except (OSError, RuntimeError):
        return False
    return True


def repair_pending_source_permissions(
    args: argparse.Namespace,
    files: dict[int, Path],
    start: int,
    end: int,
    final_sequence: int,
) -> list[str]:
    if not args.repair_source_permissions_after_final_checkpoint:
        return []
    if not any(args.segment_dir.glob("*.final.rhdf")):
        return []
    repaired = []
    for sequence in range(start, min(end + 1, final_sequence) + 1):
        path = files.get(sequence)
        if path is None:
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if not mode & stat.S_IRUSR:
            path.chmod(mode | stat.S_IRUSR)
            repaired.append(str(path))
    return repaired


def gop_endpoint_complete(
    files: dict[int, Path],
    end: int,
    final_sequence: int,
    previous_final_signature: tuple[int, int] | None,
) -> tuple[bool, tuple[int, int] | None]:
    if end < final_sequence:
        successor = files.get(end + 1)
        if successor is None or not gop_ready(files, end + 1, end + 1):
            return False, previous_final_signature
        return True, previous_final_signature
    endpoint = files.get(end)
    if endpoint is None:
        return False, None
    signature = (endpoint.stat().st_size, endpoint.stat().st_mtime_ns)
    return signature == previous_final_signature, signature


def delete_gop_middle(files: dict[int, Path], start: int, end: int) -> list[str]:
    deleted = []
    for sequence in range(start + 1, end):
        path = files[sequence]
        path.unlink()
        deleted.append(str(path))
        xdmf = Path(f"{path}.xdmf")
        xdmf.unlink(missing_ok=True)
    return deleted


def retain_file(source: Path, target: Path) -> str:
    ensure_owner_readable(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        ensure_owner_readable(target)
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Retained target has the wrong size: {target}")
        return "existing"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, temporary)
        method = "copy"
    ensure_owner_readable(temporary)
    temporary.replace(target)
    ensure_owner_readable(target)
    return method


def retention_targets(
    path: Path,
    sequence: int,
    frame_time: float,
    args: argparse.Namespace,
) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    if args.retain_every is not None and sequence % args.retain_every == 0:
        targets.append(("raw_baseline", args.retain_dir / path.name))
    for window in configured_truth_windows(args):
        tolerance = 1.0e-8 * max(1.0, abs(frame_time), abs(window["end"]))
        if window["start"] - tolerance <= frame_time <= window["end"] + tolerance:
            targets.append(
                (
                    f"truth:{window['window_id']}",
                    args.truth_dir / window["window_id"] / path.name,
                )
            )
    return targets


def retain_gop_files(
    state: dict[str, Any],
    args: argparse.Namespace,
    files: dict[int, Path],
    start: int,
    end: int,
) -> None:
    retained = state.setdefault("retained_raw_files", {})
    for sequence in range(start, end + 1):
        path = files[sequence]
        frame_time = float(state["frames"][str(sequence)]["time"])
        for purpose, target in retention_targets(path, sequence, frame_time, args):
            method = retain_file(path, target)
            xdmf_source = Path(f"{path}.xdmf")
            if xdmf_source.is_file():
                retain_file(xdmf_source, Path(f"{target}.xdmf"))
            retained[str(target)] = {
                "sequence": sequence,
                "time": frame_time,
                "source": str(path),
                "target": str(target),
                "purpose": purpose,
                "method": method,
                "size_bytes": file_size(path),
            }


def encode_gop(
    state: dict[str, Any],
    args: argparse.Namespace,
    files: dict[int, Path],
    start: int,
    end: int,
) -> None:
    gop_dir = args.output_dir / "gops" / f"{start:05d}_{end:05d}"
    if gop_dir.exists():
        shutil.rmtree(gop_dir)
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
        start_sequence=start,
        end_sequence=end,
        max_middle_frames=0,
        output_json=None,
        skip_archives=False,
    )
    summary = evaluate_codec(codec_args)
    artifact_paths = [Path(summary["output_json"])] + [
        Path(row["archive_path"])
        for row in summary["codec_schemes"].values()
    ]
    for path in artifact_paths:
        ensure_owner_readable(path)
    partial_schemes = {
        name: {"archive_paths": [row["archive_path"]]}
        for name, row in summary["codec_schemes"].items()
    }
    validate_archives(partial_schemes)
    for sequence in range(start, end + 1):
        path = files[sequence]
        state["frames"][str(sequence)] = {
            "path": str(path),
            "size_bytes": file_size(path),
            "time": phdf_time(path),
        }
    state["gops"].append(
        {
            "start_sequence": start,
            "end_sequence": end,
            "summary": summary["output_json"],
        }
    )
    retain_gop_files(state, args, files, start, end)
    if args.delete_middle_frames:
        state["deleted_middle_files"].extend(delete_gop_middle(files, start, end))


def try_encode_gop(
    state: dict[str, Any],
    args: argparse.Namespace,
    files: dict[int, Path],
    start: int,
    end: int,
) -> tuple[dict[str, Any], bool]:
    candidate = copy.deepcopy(state)
    try:
        encode_gop(candidate, args, files, start, end)
    except PermissionError as error:
        print(
            f"transient source permission error for GOP {start}:{end}; retrying: {error}",
            flush=True,
        )
        return state, False
    return candidate, True


def build_manifest(
    state: dict[str, Any],
    args: argparse.Namespace,
    *,
    finalized: bool,
) -> dict[str, Any]:
    summaries = [
        read_codec_summary(Path(row["summary"]))
        for row in state["gops"]
    ]
    frame_rows = {
        int(sequence): row for sequence, row in state["frames"].items()
    }
    keyframe_sequences = sorted(
        {int(row["start_sequence"]) for row in state["gops"]}
        | {int(row["end_sequence"]) for row in state["gops"]}
    )
    keyframes = [Path(frame_rows[sequence]["path"]) for sequence in keyframe_sequences]
    raw_size = sum(int(row["size_bytes"]) for row in frame_rows.values())
    source_keyframe_size = sum(
        int(frame_rows[sequence]["size_bytes"]) for sequence in keyframe_sequences
    )
    keyframe_archives, stored_keyframe_size = (
        compress_sequence_keyframes(
            keyframes,
            args.output_dir,
            tuple(parse_datasets(args.datasets)),
            backend=args.keyframe_backend,
            compression_level=args.keyframe_compression_level,
            workers=args.keyframe_workers,
            anchor_interval=args.keyframe_anchor_interval,
        )
        if finalized
        else ([], source_keyframe_size)
    )
    middle_frame_count = len(frame_rows) - len(keyframe_sequences)
    codec_schemes: dict[str, Any] = {}
    if summaries:
        for scheme_name in sorted(summaries[0]["codec_schemes"]):
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
                "channel_chunk_frames": scheme_rows[0].get(
                    "channel_chunk_frames"
                ),
                "archive_compression_level": scheme_rows[0].get(
                    "archive_compression_level"
                ),
                "preserve_outliers": bool(scheme_rows[0]["preserve_outliers"]),
                "exception_count": sum(int(row["exception_count"]) for row in scheme_rows),
                "archive_size_bytes": archive_size,
                "keyframe_backend": args.keyframe_backend if finalized else "raw",
                "keyframe_size_bytes": stored_keyframe_size,
                "total_with_keyframes_bytes": total_size,
                "ratio_vs_dense_phdf": raw_size / max(total_size, 1),
                "archive_bytes_per_middle_frame": archive_size / max(middle_frame_count, 1),
                "archive_paths": [row["archive_path"] for row in scheme_rows],
                "errors": aggregate_named_errors(wrapped, "scheme", "errors"),
                "derived_errors": aggregate_named_errors(wrapped, "scheme", "derived_errors"),
            }
    gop_seconds = sum(float(summary["elapsed_seconds"]) for summary in summaries)
    manifest: dict[str, Any] = {
        "format": "dense_dump_codec_sequence_v1",
        "complete": finalized,
        "segment_dir": str(args.segment_dir),
        "datasets": list(parse_datasets(args.datasets)),
        "dense_frame_count": len(frame_rows),
        "dense_files": [frame_rows[sequence]["path"] for sequence in sorted(frame_rows)],
        "frames": [
            {
                "sequence": sequence,
                "time": float(frame_rows[sequence]["time"]),
                "path": frame_rows[sequence]["path"],
                "size_bytes": int(frame_rows[sequence]["size_bytes"]),
                "exact_keyframe": sequence in keyframe_sequences,
            }
            for sequence in sorted(frame_rows)
        ],
        "keyframe_stride": args.keyframe_stride,
        "keyframe_count": len(keyframes),
        "keyframes": [str(path) for path in keyframes],
        "keyframe_storage": {
            "backend": args.keyframe_backend if finalized else "raw",
            "anchor_interval": (
                args.keyframe_anchor_interval
                if finalized
                and args.keyframe_backend
                in (
                    "bzip2-shuffle-temporal",
                    "bzip2-xz-temporal",
                    "bzip2-xz-zigzag-temporal",
                )
                else 1
            ),
            "maximum_dependency_chain": (
                args.keyframe_anchor_interval - 1
                if finalized
                and args.keyframe_backend
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
        "gop_summaries": [row["summary"] for row in state["gops"]],
        "middle_frames_deleted": bool(state["deleted_middle_files"]),
        "deleted_middle_files": state["deleted_middle_files"],
        "keyframes_deleted": False,
        "deleted_keyframe_files": [],
        "retention": {
            **configuration(args)["retention"],
            "files": [
                state["retained_raw_files"][path]
                for path in sorted(state["retained_raw_files"])
            ],
            "file_count": len(state["retained_raw_files"]),
        },
        "streaming": {
            "enabled": True,
            "finalized": finalized,
            "expected_frame_count": args.expected_frame_count,
            "processed_frame_count": len(frame_rows),
        },
        "performance": {
            "gop_encoding_seconds": gop_seconds,
            "keyframe_encoding_seconds": sum(
                float(row["elapsed_seconds"]) for row in keyframe_archives
            ),
            "dense_input_bytes_per_second": raw_size / max(gop_seconds, 1.0e-30),
            "dense_input_mib_per_second": raw_size / 1048576.0 / max(gop_seconds, 1.0e-30),
            "middle_frames_per_second": middle_frame_count / max(gop_seconds, 1.0e-30),
            "watcher_wall_seconds": time.time() - float(state["started_at_unix"]),
        },
    }
    if finalized and not args.skip_sha256:
        archives = sorted(
            {
                Path(value)
                for scheme in codec_schemes.values()
                for value in scheme["archive_paths"]
            }
        )
        manifest["integrity"] = {
            "sha256_enabled": True,
            "keyframes": [
                {"path": str(path), "size_bytes": file_size(path), "sha256": sha256_readable_file(path)}
                for path in keyframes
            ],
            "keyframe_archives": [
                {
                    "path": row["archive_path"],
                    "size_bytes": row["archive_bytes"],
                    "sha256": sha256_readable_file(Path(row["archive_path"])),
                }
                for row in keyframe_archives
            ],
            "archives": [
                {"path": str(path), "size_bytes": file_size(path), "sha256": sha256_readable_file(path)}
                for path in archives
            ],
        }
    else:
        manifest["integrity"] = {"sha256_enabled": False}
    if finalized and args.delete_keyframes_after_encode:
        if args.keyframe_backend == "raw":
            raise ValueError(
                "--delete-keyframes-after-encode requires a compressed keyframe backend"
            )
        manifest["deleted_keyframe_files"] = delete_keyframe_files(keyframes)
        manifest["keyframes_deleted"] = True
    return manifest


def main() -> int:
    args = parse_args()
    args.segment_dir = args.segment_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if (args.retain_every is None) != (args.retain_dir is None):
        raise ValueError("--retain-every and --retain-dir must be provided together")
    if args.retain_every is not None and args.retain_every <= 0:
        raise ValueError("--retain-every must be positive")
    if bool(args.truth_window_json) != (args.truth_dir is not None):
        raise ValueError("--truth-dir and --truth-window-json must be provided together")
    if args.retain_dir is not None:
        args.retain_dir = args.retain_dir.resolve()
    if args.truth_dir is not None:
        args.truth_dir = args.truth_dir.resolve()
    args.parsed_truth_windows = parse_truth_windows(args.truth_window_json)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.checkpoint_keep < 0:
        raise ValueError("--checkpoint-keep must be non-negative")
    expected_last_sequence(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repaired = repair_unreadable_codec_artifacts(args.output_dir)
    if repaired:
        print(f"repaired unreadable codec artifacts: {len(repaired)}", flush=True)
    state_path = args.output_dir / "stream_state.json"
    manifest_path = args.output_dir / "sequence_manifest.json"
    state = load_state(state_path, args)
    if state.get("finalized"):
        print(f"already finalized: {manifest_path}")
        return 0

    checkpoint_signature: tuple[tuple[str, int, int], ...] | None = None
    final_endpoint_signature: tuple[int, int] | None = None
    while True:
        if args.checkpoint_keep:
            current_signature = tuple(
                sorted(
                    (path.name, path.stat().st_size, path.stat().st_mtime_ns)
                    for path in args.segment_dir.glob("*.rhdf")
                )
            )
            if current_signature != checkpoint_signature:
                checkpoint_manifest = prune_native_checkpoints(
                    args.segment_dir,
                    keep=args.checkpoint_keep,
                )
                checkpoint_signature = tuple(
                    sorted(
                        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
                        for path in args.segment_dir.glob("*.rhdf")
                    )
                )
                print(
                    "checkpoint retention: "
                    f"{checkpoint_manifest['retained_checkpoint_count']} retained, "
                    f"{len(checkpoint_manifest['pruned'])} pruned",
                    flush=True,
                )
        pending = next_gop(state, args)
        if pending is None:
            finalized_repairs = repair_completed_artifacts(args)
            if finalized_repairs:
                print(
                    f"repaired unreadable completed artifacts: {len(finalized_repairs)}",
                    flush=True,
                )
            state["finalized"] = True
            manifest = build_manifest(state, args, finalized=True)
            atomic_json(manifest_path, manifest)
            atomic_json(state_path, state)
            print(f"finalized: {manifest_path}")
            return 0
        start, end = pending
        files = current_files(args.segment_dir)
        source_repairs = repair_pending_source_permissions(
            args,
            files,
            start,
            end,
            expected_last_sequence(args),
        )
        if source_repairs:
            print(
                f"repaired pending source permissions: {len(source_repairs)}",
                flush=True,
            )
        endpoint_complete, final_endpoint_signature = gop_endpoint_complete(
            files,
            end,
            expected_last_sequence(args),
            final_endpoint_signature,
        )
        if endpoint_complete and gop_ready(files, start, end):
            print(f"encoding completed GOP {start}:{end}", flush=True)
            state, encoded = try_encode_gop(state, args, files, start, end)
            if not encoded:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            atomic_json(state_path, state)
            atomic_json(manifest_path, build_manifest(state, args, finalized=False))
            if args.once:
                return 0
            continue
        if args.once:
            print(f"GOP {start}:{end} is not ready")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
