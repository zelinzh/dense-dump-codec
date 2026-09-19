#!/usr/bin/env python3
"""Decode one frame from a dense-dump codec sequence manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import decompress_keyframe, open_archive  # noqa: E402

from decode_dump_codec import (  # noqa: E402
    decode_archive_frames_to_arrays,
    decode_archive_frames_to_phdf,
    read_kharma_native_frame,
    read_metadata,
)
from prototype_dump_codec import phdf_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True, help="PHDF output sequence number.")
    parser.add_argument("--output-phdf", type=Path, required=True)
    parser.add_argument("--scheme", help="Codec scheme name; defaults to the first manifest scheme.")
    parser.add_argument("--datasets", help="Optional comma-separated subset to decode.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_scheme(manifest: dict[str, Any], requested: str | None) -> tuple[str, dict[str, Any]]:
    schemes = manifest["codec_schemes"]
    name = requested or next(iter(sorted(schemes)))
    if name not in schemes:
        raise KeyError(f"Unknown scheme {name!r}; available: {', '.join(sorted(schemes))}")
    return name, schemes[name]


def copy_keyframe(
    keyframe: Path,
    output_phdf: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    if output_phdf.exists():
        if not overwrite:
            raise FileExistsError(f"{output_phdf} already exists; pass --overwrite")
        output_phdf.unlink()
    output_phdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(keyframe, output_phdf)
    return {
        "output_phdf": str(output_phdf),
        "source_keyframe": str(keyframe),
        "exact_keyframe": True,
    }


_GOP_SEQUENCE_PATTERN = re.compile(r"^gop_(\d+)_(\d+)(?:_|\.)")


def archive_sequence_bounds(path: Path) -> tuple[int, int] | None:
    match = _GOP_SEQUENCE_PATTERN.match(path.name)
    if match is None:
        return None
    start_sequence, end_sequence = (int(value) for value in match.groups())
    if end_sequence <= start_sequence:
        return None
    return start_sequence, end_sequence


def read_archive_metadata_with_retry(
    archive_path: Path,
    *,
    attempts: int = 3,
    retry_seconds: float = 0.2,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            with open_archive(archive_path) as archive:
                return read_metadata(archive)
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(retry_seconds * (attempt + 1))
    raise AssertionError("unreachable")


def locate_archive(archive_paths: list[str], sequence: int) -> tuple[Path, int]:
    indexed_paths = [
        (Path(value), archive_sequence_bounds(Path(value))) for value in archive_paths
    ]
    candidates = [
        archive_path
        for archive_path, bounds in indexed_paths
        if bounds is None or bounds[0] < sequence < bounds[1]
    ]
    if not candidates:
        candidates = [archive_path for archive_path, _ in indexed_paths]
    for archive_path in candidates:
        metadata = read_archive_metadata_with_retry(archive_path)
        sequences = [int(item) for item in metadata.get("middle_sequences", [])]
        if sequence in sequences:
            return archive_path, sequences.index(sequence) + 1
    raise KeyError(f"Sequence {sequence} is not present in any codec GOP")


def group_archive_frames(
    archive_paths: list[str],
    sequences: list[int] | tuple[int, ...],
) -> dict[Path, dict[int, int]]:
    """Index many sequence numbers while opening each candidate GOP once."""
    remaining = set(map(int, sequences))
    if not remaining:
        return {}
    minimum = min(remaining)
    maximum = max(remaining)
    groups: dict[Path, dict[int, int]] = {}
    for value in archive_paths:
        archive_path = Path(value)
        bounds = archive_sequence_bounds(archive_path)
        if bounds is not None and (bounds[1] <= minimum or bounds[0] >= maximum):
            continue
        metadata = read_archive_metadata_with_retry(archive_path)
        rows = {
            frame_index: sequence
            for frame_index, sequence in enumerate(
                map(int, metadata.get("middle_sequences", [])), start=1
            )
            if sequence in remaining
        }
        if rows:
            groups[archive_path] = rows
            remaining.difference_update(rows.values())
            if not remaining:
                break
    if remaining:
        missing = ", ".join(map(str, sorted(remaining)))
        raise KeyError(f"Sequences are not present in any codec GOP: {missing}")
    return groups


def keyframe_archive_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["sequence"]): row
        for row in manifest.get("keyframe_storage", {}).get("archives", [])
    }


def retained_keyframe_map(manifest: dict[str, Any]) -> dict[int, Path]:
    retained: dict[int, tuple[int, Path]] = {}
    for row in manifest.get("retention", {}).get("files", []):
        target = Path(row["target"])
        if target.suffix != ".phdf":
            continue
        sequence = int(row["sequence"])
        priority = 0 if row.get("purpose") == "raw_baseline" else 1
        current = retained.get(sequence)
        if current is None or priority < current[0]:
            retained[sequence] = (priority, target)
    return {sequence: row[1] for sequence, row in retained.items()}


def existing_raw_keyframe(
    manifest: dict[str, Any],
    sequence: int,
    preferred_path: Path,
) -> Path | None:
    retained = retained_keyframe_map(manifest).get(sequence)
    if retained is not None and retained.is_file():
        return retained
    if preferred_path.is_file():
        return preferred_path
    for value in manifest["keyframes"]:
        candidate = Path(value)
        if phdf_sequence(candidate) == sequence and candidate.is_file():
            return candidate
    return None


def _materialize_keyframe(
    manifest: dict[str, Any],
    sequence: int,
    output_phdf: Path,
    *,
    overwrite: bool,
    temporary_path: Path,
    cache: dict[int, Path],
    active: set[int],
) -> dict[str, Any]:
    if sequence in cache:
        cached = cache[sequence]
        if cached.resolve() == output_phdf.resolve():
            return {"output_phdf": str(output_phdf), "exact_keyframe": True}
        return copy_keyframe(cached, output_phdf, overwrite=overwrite)
    if sequence in active:
        raise ValueError(f"Cyclic keyframe dependency at sequence {sequence}")
    active.add(sequence)
    try:
        retained_keyframe = retained_keyframe_map(manifest).get(sequence)
        if retained_keyframe is not None and retained_keyframe.is_file():
            result = copy_keyframe(
                retained_keyframe,
                output_phdf,
                overwrite=overwrite,
            )
            cache[sequence] = output_phdf
            return {
                **result,
                "source_retained_keyframe": str(retained_keyframe),
            }
        archives = keyframe_archive_map(manifest)
        if sequence in archives:
            row = archives[sequence]
            reference_paths = []
            for reference_sequence in row.get("reference_sequences", []):
                reference_sequence = int(reference_sequence)
                reference_output = temporary_path / f"keyframe_{reference_sequence:05d}.phdf"
                _materialize_keyframe(
                    manifest,
                    reference_sequence,
                    reference_output,
                    overwrite=True,
                    temporary_path=temporary_path,
                    cache=cache,
                    active=active,
                )
                reference_paths.append(cache[reference_sequence])
            archive_path = Path(row["archive_path"])
            result = decompress_keyframe(
                archive_path,
                output_phdf,
                overwrite=overwrite,
                reference_paths=reference_paths,
            )
            cache[sequence] = output_phdf
            return {
                **result,
                "output_phdf": str(output_phdf),
                "source_keyframe_archive": str(archive_path),
                "exact_keyframe": True,
            }
        for value in manifest["keyframes"]:
            keyframe = Path(value)
            if phdf_sequence(keyframe) == sequence:
                result = copy_keyframe(keyframe, output_phdf, overwrite=overwrite)
                cache[sequence] = output_phdf
                return result
        raise KeyError(f"Sequence {sequence} is not a keyframe")
    finally:
        active.remove(sequence)


def materialize_keyframe(
    manifest: dict[str, Any],
    sequence: int,
    output_phdf: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ddc_keyframe_dependencies_") as temporary:
        return _materialize_keyframe(
            manifest,
            sequence,
            output_phdf,
            overwrite=overwrite,
            temporary_path=Path(temporary),
            cache={},
            active=set(),
        )


def materialize_keyframes(
    manifest: dict[str, Any],
    outputs: dict[int, Path],
    *,
    overwrite: bool = False,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="ddc_keyframe_dependencies_") as temporary:
        temporary_path = Path(temporary)
        cache: dict[int, Path] = {}
        active: set[int] = set()
        for sequence, output_phdf in outputs.items():
            results[sequence] = _materialize_keyframe(
                manifest,
                sequence,
                output_phdf,
                overwrite=overwrite,
                temporary_path=temporary_path,
                cache=cache,
                active=active,
            )
    return results


def decode_sequence_frames(
    manifest_path: Path,
    outputs: dict[int, Path],
    *,
    requested_scheme: str | None = None,
    datasets: str | None = None,
    overwrite: bool = False,
) -> dict[int, dict[str, Any]]:
    if not outputs:
        return {}
    manifest = load_manifest(manifest_path)
    normalized_outputs = {
        int(sequence): Path(output_phdf) for sequence, output_phdf in outputs.items()
    }
    keyframe_sequences = {phdf_sequence(Path(value)) for value in manifest["keyframes"]}
    keyframe_outputs = {
        sequence: output_phdf
        for sequence, output_phdf in normalized_outputs.items()
        if sequence in keyframe_sequences
    }
    results: dict[int, dict[str, Any]] = {}
    if keyframe_outputs:
        keyframe_results = materialize_keyframes(
            manifest,
            keyframe_outputs,
            overwrite=overwrite,
        )
        for sequence, result in keyframe_results.items():
            result["sequence"] = sequence
            results[sequence] = result

    middle_outputs = {
        sequence: output_phdf
        for sequence, output_phdf in normalized_outputs.items()
        if sequence not in keyframe_sequences
    }
    if not middle_outputs:
        return results

    scheme_name, scheme = select_scheme(manifest, requested_scheme)
    indexed_groups = group_archive_frames(scheme["archive_paths"], tuple(middle_outputs))
    archive_groups = {
        archive_path: {
            frame_index: (sequence, middle_outputs[sequence])
            for frame_index, sequence in frame_rows.items()
        }
        for archive_path, frame_rows in indexed_groups.items()
    }

    archive_rows = keyframe_archive_map(manifest)
    with tempfile.TemporaryDirectory(prefix="ddc_keyframes_") as temporary:
        temporary_path = Path(temporary)
        keyframe_cache: dict[int, Path] = {}
        active: set[int] = set()
        for archive_path, frame_rows in archive_groups.items():
            metadata = read_archive_metadata_with_retry(archive_path)
            start_sequence = phdf_sequence(Path(metadata["start_file"]))
            end_sequence = phdf_sequence(Path(metadata["end_file"]))
            if start_sequence not in archive_rows and end_sequence not in archive_rows:
                start_phdf = existing_raw_keyframe(
                    manifest,
                    start_sequence,
                    Path(metadata["start_file"]),
                )
                end_phdf = existing_raw_keyframe(
                    manifest,
                    end_sequence,
                    Path(metadata["end_file"]),
                )
                if start_phdf is None or end_phdf is None:
                    raise FileNotFoundError(
                        "DDC GOP raw keyframes are unavailable in both the source "
                        "and retention paths"
                    )
            else:
                start_phdf = temporary_path / f"keyframe_{start_sequence:05d}.phdf"
                end_phdf = temporary_path / f"keyframe_{end_sequence:05d}.phdf"
                _materialize_keyframe(
                    manifest,
                    start_sequence,
                    start_phdf,
                    overwrite=True,
                    temporary_path=temporary_path,
                    cache=keyframe_cache,
                    active=active,
                )
                _materialize_keyframe(
                    manifest,
                    end_sequence,
                    end_phdf,
                    overwrite=True,
                    temporary_path=temporary_path,
                    cache=keyframe_cache,
                    active=active,
                )

            batch_results = decode_archive_frames_to_phdf(
                archive_path,
                {
                    frame_index: output_phdf
                    for frame_index, (_, output_phdf) in frame_rows.items()
                },
                datasets_value=datasets,
                start_phdf=start_phdf,
                end_phdf=end_phdf,
                overwrite=overwrite,
            )
            for frame_index, result in batch_results.items():
                sequence = frame_rows[frame_index][0]
                result.update(
                    {
                        "sequence": sequence,
                        "scheme": scheme_name,
                        "exact_keyframe": False,
                    }
                )
                results[sequence] = result
    return results


def decode_sequence_frames_to_arrays(
    manifest_path: Path,
    sequences: tuple[int, ...],
    *,
    requested_scheme: str | None = None,
    datasets: str = "prims.rho,prims.u,prims.uvec,prims.B",
    workers: int = 1,
    archive_cache_chunks: int = 128,
) -> dict[int, dict[str, Any]]:
    """Decode selected sequence numbers directly to KPolaris staged arrays."""
    if not sequences:
        return {}
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = load_manifest(manifest_path)
    requested = tuple(dict.fromkeys(int(sequence) for sequence in sequences))
    frame_times = {
        int(row["sequence"]): float(row["time"])
        for row in manifest.get("frames", [])
    }
    dataset_names = tuple(item.strip() for item in datasets.split(",") if item.strip())
    required = {"prims.rho", "prims.u", "prims.uvec", "prims.B"}
    if set(dataset_names) != required or len(dataset_names) != len(required):
        raise ValueError(
            "KPolaris native DDC input requires exactly prims.rho, prims.u, "
            "prims.uvec,prims.B"
        )
    keyframe_sequences = {phdf_sequence(Path(value)) for value in manifest["keyframes"]}
    results: dict[int, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="ddc_native_keyframes_") as temporary:
        temporary_path = Path(temporary)
        keyframe_cache: dict[int, Path] = {}
        active: set[int] = set()
        for sequence in requested:
            if sequence not in keyframe_sequences:
                continue
            preferred = next(
                Path(value)
                for value in manifest["keyframes"]
                if phdf_sequence(Path(value)) == sequence
            )
            source = existing_raw_keyframe(manifest, sequence, preferred)
            if source is None:
                source = temporary_path / f"keyframe_{sequence:05d}.phdf"
                _materialize_keyframe(
                    manifest,
                    sequence,
                    source,
                    overwrite=True,
                    temporary_path=temporary_path,
                    cache=keyframe_cache,
                    active=active,
                )
            results[sequence] = read_kharma_native_frame(
                source,
                dataset_names,
                sequence=sequence,
                time_value=frame_times.get(sequence),
            )

        middle = [sequence for sequence in requested if sequence not in keyframe_sequences]
        if not middle:
            return results
        scheme_name, scheme = select_scheme(manifest, requested_scheme)
        archive_groups = group_archive_frames(scheme["archive_paths"], middle)
        archive_jobs = []
        for archive_path, frame_rows in archive_groups.items():
            metadata = read_archive_metadata_with_retry(archive_path)
            start_sequence = phdf_sequence(Path(metadata["start_file"]))
            end_sequence = phdf_sequence(Path(metadata["end_file"]))
            start_phdf = existing_raw_keyframe(
                manifest, start_sequence, Path(metadata["start_file"])
            )
            end_phdf = existing_raw_keyframe(
                manifest, end_sequence, Path(metadata["end_file"])
            )
            if start_phdf is None:
                start_phdf = temporary_path / f"keyframe_{start_sequence:05d}.phdf"
                _materialize_keyframe(
                    manifest,
                    start_sequence,
                    start_phdf,
                    overwrite=True,
                    temporary_path=temporary_path,
                    cache=keyframe_cache,
                    active=active,
                )
            if end_phdf is None:
                end_phdf = temporary_path / f"keyframe_{end_sequence:05d}.phdf"
                _materialize_keyframe(
                    manifest,
                    end_sequence,
                    end_phdf,
                    overwrite=True,
                    temporary_path=temporary_path,
                    cache=keyframe_cache,
                    active=active,
                )
            archive_jobs.append((archive_path, frame_rows, start_phdf, end_phdf))

        def decode_group(job):
            archive_path, frame_rows, start_phdf, end_phdf = job
            decoded = decode_archive_frames_to_arrays(
                archive_path,
                tuple(frame_rows),
                datasets_value=datasets,
                start_phdf=start_phdf,
                end_phdf=end_phdf,
                cache_chunks=archive_cache_chunks,
            )
            return frame_rows, decoded

        executor = None
        if workers == 1 or len(archive_jobs) == 1:
            decoded_groups = map(decode_group, archive_jobs)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(workers, len(archive_jobs))
            )
            decoded_groups = executor.map(decode_group, archive_jobs)
        try:
            for frame_rows, decoded in decoded_groups:
                for frame_index, frame in decoded.items():
                    sequence = frame_rows[frame_index]
                    frame.update(
                        {
                            "sequence": sequence,
                            "time": frame_times.get(sequence, frame["time"]),
                            "scheme": scheme_name,
                        }
                    )
                    results[sequence] = frame
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    return results


def decode_sequence_frame(
    manifest_path: Path,
    sequence: int,
    output_phdf: Path,
    *,
    requested_scheme: str | None = None,
    datasets: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    return decode_sequence_frames(
        manifest_path,
        {sequence: output_phdf},
        requested_scheme=requested_scheme,
        datasets=datasets,
        overwrite=overwrite,
    )[sequence]


def main() -> int:
    args = parse_args()
    result = decode_sequence_frame(
        args.manifest,
        args.sequence,
        args.output_phdf,
        requested_scheme=args.scheme,
        datasets=args.datasets,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
