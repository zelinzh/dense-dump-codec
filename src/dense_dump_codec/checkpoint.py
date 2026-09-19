"""Native KHARMA checkpoint validation, pruning, and sidecar rollback."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_FORMAT = "kharma_native_checkpoint_manifest_v1"
RECOVERY_FORMAT = "kharma_native_checkpoint_recovery_v1"
PERIODIC_CHECKPOINT_RE = re.compile(r"\.out1\.(\d+)\.rhdf$")
PHDF_SEQUENCE_RE = re.compile(r"\.out0\.(\d+)\.phdf$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def periodic_checkpoint_number(path: Path) -> int | None:
    match = PERIODIC_CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else None


def phdf_sequence(path: Path) -> int | None:
    match = PHDF_SEQUENCE_RE.search(path.name)
    return int(match.group(1)) if match else None


def _input_section(input_text: str, section: str) -> str:
    match = re.search(
        rf"^<{re.escape(section)}>\s*$\n(.*?)(?=^<[^>]+>\s*$|\Z)",
        input_text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Checkpoint input is missing <{section}>")
    return match.group(1)


def _input_value(section_text: str, key: str) -> str:
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*([^#\n]+?)\s*(?:#.*)?$",
        section_text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Checkpoint input section is missing {key}")
    return match.group(1).strip()


def _input_value_or(section_text: str, key: str, default: str) -> str:
    try:
        return _input_value(section_text, key)
    except ValueError:
        return default


@dataclass(frozen=True)
class NativeCheckpoint:
    path: str
    checkpoint_number: int
    time: float
    cycle: int
    dt: float
    includes_ghost_zones: bool
    problem_id: str
    output0_file_number: int
    output0_next_time: float
    output1_file_number: int
    output1_next_time: float
    output2_file_number: int
    output2_next_time: float

    @property
    def last_dense_sequence(self) -> int:
        return self.output0_file_number - 1

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["last_dense_sequence"] = self.last_dense_sequence
        return result


@dataclass(frozen=True)
class CheckpointMeshLayout:
    root_grid_shape: tuple[int, int, int]
    meshblock_shape: tuple[int, int, int]
    meshblock_count: int
    blocks_per_rank: int | tuple[int, ...]
    ghost_zones: int
    pack_size: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _integer_triplet(section_text: str) -> tuple[int, int, int]:
    return tuple(int(_input_value(section_text, f"nx{axis}")) for axis in (1, 2, 3))


def read_checkpoint_mesh_layout(path: Path) -> CheckpointMeshLayout:
    """Read the decomposition embedded in a native Parthenon checkpoint."""

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for checkpoint mesh validation") from exc

    with h5py.File(path, "r") as handle:
        if "Info" not in handle or "Input" not in handle:
            raise ValueError(f"Checkpoint is missing Info or Input: {path}")
        input_value = handle["Input"].attrs.get("File")
        if input_value is None:
            raise ValueError(f"Checkpoint is missing Input/File: {path}")
        input_text = (
            input_value.decode("utf-8") if isinstance(input_value, bytes) else str(input_value)
        )
        mesh = _input_section(input_text, "parthenon/mesh")
        meshblock = _input_section(input_text, "parthenon/meshblock")
        root_shape = _integer_triplet(mesh)
        block_shape = _integer_triplet(meshblock)
        if any(root % block != 0 for root, block in zip(root_shape, block_shape)):
            raise ValueError(
                f"Root grid {root_shape} is not divisible by MeshBlock {block_shape}"
            )
        expected_count = 1
        for root, block in zip(root_shape, block_shape):
            expected_count *= root // block
        info = handle["Info"].attrs
        recorded_count = int(info.get("NumMeshBlocks", expected_count))
        if recorded_count != expected_count:
            raise ValueError(
                f"Checkpoint records {recorded_count} MeshBlocks, expected {expected_count}"
            )
        rank_counts = info.get("BlocksPerPE", recorded_count)
        if getattr(rank_counts, "ndim", 0) > 0:
            if rank_counts.ndim != 1:
                raise ValueError("BlocksPerPE must be a scalar or one-dimensional array")
            counts = tuple(int(count) for count in rank_counts)
            if not counts or any(count < 0 for count in counts) or sum(counts) != recorded_count:
                raise ValueError("BlocksPerPE does not match NumMeshBlocks")
            blocks_per_rank = counts[0] if len(set(counts)) == 1 else counts
        else:
            blocks_per_rank = int(rank_counts)
        return CheckpointMeshLayout(
            root_grid_shape=root_shape,
            meshblock_shape=block_shape,
            meshblock_count=recorded_count,
            blocks_per_rank=blocks_per_rank,
            ghost_zones=int(info.get("NGhost", 0)),
            pack_size=int(_input_value(mesh, "pack_size")),
        )


def read_native_checkpoint(path: Path) -> NativeCheckpoint:
    number = periodic_checkpoint_number(path)
    if number is None:
        raise ValueError(f"Not a periodic native checkpoint: {path}")
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for native checkpoint validation") from exc

    with h5py.File(path, "r") as handle:
        if "Info" not in handle or "Input" not in handle:
            raise ValueError(f"Checkpoint is missing Info or Input: {path}")
        info = handle["Info"].attrs
        input_value = handle["Input"].attrs.get("File")
        if input_value is None:
            raise ValueError(f"Checkpoint is missing Input/File: {path}")
        if isinstance(input_value, bytes):
            input_text = input_value.decode("utf-8")
        else:
            input_text = str(input_value)

        job = _input_section(input_text, "parthenon/job")
        outputs = {
            output: _input_section(input_text, f"parthenon/{output}")
            for output in ("output0", "output1", "output2")
        }
        return NativeCheckpoint(
            path=str(path.resolve()),
            checkpoint_number=number,
            time=float(info["Time"]),
            cycle=int(info["NCycle"]),
            dt=float(info["dt"]),
            includes_ghost_zones=bool(info["IncludesGhost"]),
            problem_id=_input_value(job, "problem_id"),
            output0_file_number=int(_input_value_or(outputs["output0"], "file_number", "0")),
            output0_next_time=float(_input_value_or(outputs["output0"], "next_time", "-1")),
            output1_file_number=int(_input_value_or(outputs["output1"], "file_number", "0")),
            output1_next_time=float(_input_value_or(outputs["output1"], "next_time", "-1")),
            output2_file_number=int(_input_value_or(outputs["output2"], "file_number", "0")),
            output2_next_time=float(_input_value_or(outputs["output2"], "next_time", "-1")),
        )


def complete_periodic_checkpoints(output_dir: Path) -> list[NativeCheckpoint]:
    checkpoints = []
    for path in output_dir.glob("*.rhdf"):
        if periodic_checkpoint_number(path) is None:
            continue
        try:
            checkpoints.append(read_native_checkpoint(path))
        except (OSError, RuntimeError, KeyError, TypeError, ValueError):
            continue
    return sorted(checkpoints, key=lambda row: row.checkpoint_number)


def latest_complete_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = complete_periodic_checkpoints(output_dir)
    return Path(checkpoints[-1].path) if checkpoints else None


def _checkpoint_cache(path: Path) -> dict[str, dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(row["path"]): row
        for row in manifest.get("checkpoints", [])
        if isinstance(row, dict) and row.get("path")
    }


def prune_native_checkpoints(
    output_dir: Path,
    *,
    keep: int,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if keep < 1:
        raise ValueError("keep must be positive")
    manifest_path = manifest_path or output_dir / "checkpoint_manifest.json"
    previous_cache = _checkpoint_cache(manifest_path)
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}

    complete = complete_periodic_checkpoints(output_dir)
    retained = complete[-keep:]
    pruned_rows = list(previous.get("pruned", []))
    for checkpoint in complete[:-keep]:
        path = Path(checkpoint.path)
        row = checkpoint.to_json()
        row.update(
            {
                "size_bytes": path.stat().st_size,
                "pruned_at_utc": utc_now(),
            }
        )
        path.unlink()
        Path(f"{path}.xdmf").unlink(missing_ok=True)
        pruned_rows.append(row)

    rows = []
    for checkpoint in retained:
        path = Path(checkpoint.path)
        stat = path.stat()
        cached = previous_cache.get(str(path))
        digest = None
        if (
            cached is not None
            and int(cached.get("size_bytes", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
        ):
            digest = cached.get("sha256")
        if not digest:
            digest = sha256_file(path)
        row = checkpoint.to_json()
        row.update(
            {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
        )
        rows.append(row)

    manifest = {
        "format": CHECKPOINT_FORMAT,
        "updated_at_utc": utc_now(),
        "output_dir": str(output_dir.resolve()),
        "keep": keep,
        "complete_checkpoint_count": len(complete),
        "retained_checkpoint_count": len(rows),
        "latest_checkpoint": rows[-1]["path"] if rows else None,
        "checkpoints": rows,
        "pruned": pruned_rows,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _archive_json(path: Path, archive_dir: Path) -> str | None:
    if not path.is_file():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / path.name
    shutil.copy2(path, target)
    return str(target)


def _history_parts(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    comments = [line for line in lines if line.lstrip().startswith("#") or not line.strip()]
    data = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return comments, data


def history_segment_paths(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("*.hst")
        if path.name != "kharma_history_combined.hst"
    )


def truncate_histories(output_dir: Path, data_rows: int) -> list[dict[str, Any]]:
    segments = history_segment_paths(output_dir)
    parsed = [(path, *_history_parts(path)) for path in segments]
    total_rows = sum(len(data) for _path, _comments, data in parsed)
    if total_rows < data_rows:
        raise ValueError(
            f"Histories have {total_rows} rows, fewer than checkpoint count {data_rows}"
        )
    remaining = data_rows
    reports = []
    for path, comments, data in parsed:
        retained_count = min(len(data), remaining)
        retained = comments + data[:retained_count]
        temporary = path.with_name(f".{path.name}.resume.tmp")
        temporary.write_text("".join(retained), encoding="utf-8")
        temporary.replace(path)
        reports.append(
            {
                "path": str(path),
                "rows_before": len(data),
                "rows_after": retained_count,
                "rows_removed": len(data) - retained_count,
            }
        )
        remaining -= retained_count
    (output_dir / "kharma_history_combined.hst").unlink(missing_ok=True)
    return reports


def combine_histories(output_dir: Path) -> dict[str, Any] | None:
    segments = history_segment_paths(output_dir)
    if not segments:
        return None
    comments: list[str] = []
    data: list[str] = []
    segment_rows = []
    for path in segments:
        path_comments, path_data = _history_parts(path)
        if not comments and path_comments:
            comments = path_comments
        data.extend(path_data)
        segment_rows.append({"path": str(path), "rows": len(path_data)})
    target = output_dir / "kharma_history_combined.hst"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text("".join(comments + data), encoding="utf-8")
    temporary.replace(target)
    return {
        "path": str(target),
        "row_count": len(data),
        "segments": segment_rows,
    }


def _remove_phdf_after(output_dir: Path, last_sequence: int) -> list[str]:
    removed = []
    for path in output_dir.glob("*.phdf"):
        sequence = phdf_sequence(path)
        if sequence is None or sequence <= last_sequence:
            continue
        path.unlink()
        Path(f"{path}.xdmf").unlink(missing_ok=True)
        removed.append(str(path))
    return sorted(removed)


def _remove_raw_after(raw_dir: Path, last_sequence: int) -> list[str]:
    removed = []
    if not raw_dir.exists():
        return removed
    for path in raw_dir.rglob("*.phdf"):
        sequence = phdf_sequence(path)
        if sequence is None or sequence <= last_sequence:
            continue
        path.unlink()
        Path(f"{path}.xdmf").unlink(missing_ok=True)
        removed.append(str(path))
    return sorted(removed)


def _rollback_codec_state(codec_dir: Path, last_sequence: int) -> dict[str, Any]:
    state_path = codec_dir / "stream_state.json"
    manifest_path = codec_dir / "sequence_manifest.json"
    if not state_path.is_file():
        manifest_path.unlink(missing_ok=True)
        return {
            "state_present": False,
            "removed_gops": [],
            "removed_retained_files": [],
        }

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("finalized"):
        raise ValueError("Cannot resume a finalized DDC sequence")
    stride = int(state["configuration"]["keyframe_stride"])
    start = int(state["configuration"].get("start_sequence", 0))
    if (last_sequence - start) % stride:
        raise ValueError(
            f"Checkpoint sequence {last_sequence} is not aligned to DDC GOP stride {stride}"
        )

    kept_gops = []
    removed_gops = []
    for row in state.get("gops", []):
        if int(row["end_sequence"]) <= last_sequence:
            kept_gops.append(row)
        else:
            removed_gops.append(row)

    kept_gop_dirs = {
        str(Path(row["summary"]).resolve().parent)
        for row in kept_gops
    }
    gops_dir = codec_dir / "gops"
    removed_gop_dirs = []
    if gops_dir.is_dir():
        for path in gops_dir.iterdir():
            if not path.is_dir() or str(path.resolve()) in kept_gop_dirs:
                continue
            shutil.rmtree(path)
            removed_gop_dirs.append(str(path))

    retained = state.get("retained_raw_files", {})
    kept_retained = {}
    removed_retained = []
    for target, row in retained.items():
        if int(row["sequence"]) <= last_sequence:
            kept_retained[target] = row
            continue
        path = Path(target)
        path.unlink(missing_ok=True)
        Path(f"{path}.xdmf").unlink(missing_ok=True)
        removed_retained.append(target)

    state["gops"] = kept_gops
    state["frames"] = {
        sequence: row
        for sequence, row in state.get("frames", {}).items()
        if int(sequence) <= last_sequence
    }
    state["deleted_middle_files"] = [
        value
        for value in state.get("deleted_middle_files", [])
        if (phdf_sequence(Path(value)) or -1) <= last_sequence
    ]
    state["retained_raw_files"] = kept_retained
    state["finalized"] = False
    atomic_json(state_path, state)
    manifest_path.unlink(missing_ok=True)
    return {
        "state_present": True,
        "stride": stride,
        "kept_gop_count": len(kept_gops),
        "removed_gops": removed_gops,
        "removed_gop_dirs": sorted(removed_gop_dirs),
        "removed_retained_files": sorted(removed_retained),
    }


def rollback_to_native_checkpoint(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    codec_dir: Path | None = None,
    raw_dirs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    if checkpoint_path.parent != output_dir:
        raise ValueError("Native checkpoint must be inside the simulation output directory")
    checkpoint = read_native_checkpoint(checkpoint_path)
    if not checkpoint.includes_ghost_zones:
        raise ValueError(
            "Native checkpoint does not include ghost zones; refusing a "
            "restart that can perturb physical boundary cells"
        )
    last_sequence = checkpoint.last_dense_sequence
    if last_sequence < 0 and (
        codec_dir is not None or raw_dirs or any(output_dir.glob("*.phdf"))
    ):
        raise ValueError(
            "Checkpoint has no dense output frame and cannot roll back existing outputs"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = output_dir / "resume_audit" / stamp
    archived = {
        "status": _archive_json(output_dir / "status.json", archive_dir),
        "checkpoint_manifest": _archive_json(
            output_dir / "checkpoint_manifest.json", archive_dir
        ),
    }
    if codec_dir is not None:
        archived["stream_state"] = _archive_json(
            codec_dir / "stream_state.json", archive_dir
        )
        archived["sequence_manifest"] = _archive_json(
            codec_dir / "sequence_manifest.json", archive_dir
        )

    removed_phdf = _remove_phdf_after(output_dir, last_sequence)
    removed_raw = {
        str(path): _remove_raw_after(path.resolve(), last_sequence) for path in raw_dirs
    }
    codec = (
        _rollback_codec_state(codec_dir.resolve(), last_sequence)
        if codec_dir is not None
        else {"state_present": False}
    )
    histories = truncate_histories(output_dir, checkpoint.output2_file_number)

    removed_checkpoints = []
    for path in output_dir.glob("*.rhdf"):
        number = periodic_checkpoint_number(path)
        remove = path.name.endswith(".final.rhdf") or (
            number is not None and number > checkpoint.checkpoint_number
        )
        if not remove:
            continue
        path.unlink()
        Path(f"{path}.xdmf").unlink(missing_ok=True)
        removed_checkpoints.append(str(path))

    report = {
        "format": RECOVERY_FORMAT,
        "recovered_at_utc": utc_now(),
        "checkpoint": checkpoint.to_json(),
        "output_dir": str(output_dir),
        "codec_dir": str(codec_dir.resolve()) if codec_dir is not None else None,
        "archived_metadata": archived,
        "removed_phdf": removed_phdf,
        "removed_raw": removed_raw,
        "codec": codec,
        "histories": histories,
        "removed_checkpoints": sorted(removed_checkpoints),
    }
    archive_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(archive_dir / "recovery.json", report)
    atomic_json(output_dir / "resume_recovery_latest.json", report)
    with (output_dir / "resume_recovery_history.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(report, sort_keys=True) + "\n")
    return report
