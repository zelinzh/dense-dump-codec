#!/usr/bin/env python3
"""Compare sparse primitive interpolation and DDC reconstruction against dense PHDF truth."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import open_archive as open_archive_unchecked  # noqa: E402

from decode_dump_codec import decode_dataset, read_metadata  # noqa: E402
from prototype_dump_codec import (  # noqa: E402
    DEFAULT_DATASETS,
    ErrorAccumulator,
    finalize_errors,
    parse_datasets,
    phdf_sequence,
    phdf_time,
    primitive_derived_fields,
    read_dataset as read_dataset_unchecked,
    sequence_files,
    update_channel_errors,
)


PRIMITIVE_DATASETS = ("prims.rho", "prims.u", "prims.uvec", "prims.B")


def repair_owner_readable(path: Path) -> None:
    path.chmod(0o644)


def open_archive(path: Path, *, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return open_archive_unchecked(path)
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            repair_owner_readable(path)
            time.sleep(0.2)
    raise AssertionError("unreachable")


def read_dataset(path: Path, name: str, *, attempts: int = 5) -> np.ndarray:
    for attempt in range(attempts):
        try:
            return read_dataset_unchecked(path, name)
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            repair_owner_readable(path)
            time.sleep(0.2)
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--codec-manifest", type=Path, required=True)
    parser.add_argument(
        "--raw-strides",
        default="5,10",
        help="Comma-separated raw cadence strides in dense-frame steps.",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--scheme", help="DDC scheme name; defaults to first sorted scheme.")
    parser.add_argument(
        "--raw-baseline-json",
        type=Path,
        help="Reuse raw cadence metrics from a fingerprint-matched comparison JSON.",
    )
    parser.add_argument(
        "--raw-workers",
        type=int,
        default=1,
        help="Independent processes used for raw-cadence evaluation.",
    )
    parser.add_argument(
        "--reuse-manifest-errors",
        action="store_true",
        help=(
            "Reuse codec errors recorded during encoding instead of decoding the same "
            "middle frames again."
        ),
    )
    parser.add_argument(
        "--allow-truth-subset",
        action="store_true",
        help=(
            "Evaluate a contiguous dense truth window covered by a larger complete "
            "DDC manifest."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def parse_strides(value: str) -> tuple[int, ...]:
    strides = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not strides or any(stride < 2 for stride in strides):
        raise ValueError("--raw-strides must contain integers of at least 2")
    return strides


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_index(files: list[Path]) -> dict[int, int]:
    return {phdf_sequence(path): index for index, path in enumerate(files)}


def evaluate_frame(
    target_values: dict[str, np.ndarray],
    predicted_values: dict[str, np.ndarray],
    errors: dict[str, ErrorAccumulator],
    derived_errors: dict[str, ErrorAccumulator],
) -> None:
    for dataset, target in target_values.items():
        update_channel_errors(errors, dataset, predicted_values[dataset], target)
    if set(PRIMITIVE_DATASETS).issubset(target_values):
        target_derived = primitive_derived_fields(target_values)
        predicted_derived = primitive_derived_fields(predicted_values)
        for name, target in target_derived.items():
            derived_errors.setdefault(name, ErrorAccumulator()).update(
                predicted_derived[name],
                target,
            )


def primitive_linear(start: np.ndarray, end: np.ndarray, tau: float) -> np.ndarray:
    return ((1.0 - tau) * start + tau * end).astype(np.float32, copy=False)


def raw_evaluation_states(strides: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    return {
        stride: {
            "common_errors": {},
            "common_derived_errors": {},
            "interior_errors": {},
            "interior_derived_errors": {},
            "interior_count": 0,
        }
        for stride in strides
    }


def merge_error_accumulators(
    target: dict[str, ErrorAccumulator],
    source: dict[str, ErrorAccumulator],
) -> None:
    for name, source_accumulator in source.items():
        target_accumulator = target.setdefault(name, ErrorAccumulator())
        target_accumulator.sse += source_accumulator.sse
        target_accumulator.sae += source_accumulator.sae
        target_accumulator.target_sse += source_accumulator.target_sse
        target_accumulator.target_abs += source_accumulator.target_abs
        target_accumulator.count += source_accumulator.count
        target_accumulator.min_value = min(
            target_accumulator.min_value,
            source_accumulator.min_value,
        )
        target_accumulator.max_abs_error = max(
            target_accumulator.max_abs_error,
            source_accumulator.max_abs_error,
        )


def merge_raw_evaluation_states(
    target: dict[int, dict[str, Any]],
    source: dict[int, dict[str, Any]],
) -> None:
    for stride, source_state in source.items():
        target_state = target[stride]
        target_state["interior_count"] += source_state["interior_count"]
        for key in (
            "common_errors",
            "common_derived_errors",
            "interior_errors",
            "interior_derived_errors",
        ):
            merge_error_accumulators(target_state[key], source_state[key])


def raw_evaluation_worker(
    file_values: tuple[str, ...],
    times: tuple[float, ...],
    evaluation_positions: tuple[int, ...],
    strides: tuple[int, ...],
    datasets: tuple[str, ...],
) -> dict[int, dict[str, Any]]:
    files = [Path(value) for value in file_values]
    time_values = list(times)
    states = raw_evaluation_states(strides)
    anchor_cache: dict[tuple[int, str], np.ndarray] = {}
    for position in evaluation_positions:
        target_values = {
            dataset: read_dataset(files[position], dataset) for dataset in datasets
        }
        update_raw_evaluations(
            files,
            time_values,
            position,
            target_values,
            strides,
            datasets,
            states,
            anchor_cache,
        )
    return states


def split_evaluation_positions(
    evaluation_positions: list[int],
    workers: int,
) -> list[tuple[int, ...]]:
    if workers < 1:
        raise ValueError("--raw-workers must be at least 1")
    chunk_size = max(1, math.ceil(len(evaluation_positions) / workers))
    return [
        tuple(evaluation_positions[start : start + chunk_size])
        for start in range(0, len(evaluation_positions), chunk_size)
    ]


def raw_anchor_positions(
    files: list[Path],
    evaluation_positions: list[int],
    stride: int,
) -> list[int]:
    if not evaluation_positions:
        return []
    last_evaluation = max(evaluation_positions)
    last_anchor = math.ceil(last_evaluation / stride) * stride
    if last_anchor >= len(files):
        raise ValueError(
            f"Raw stride {stride} requires a right cadence anchor at truth "
            f"position {last_anchor}, but only {len(files)} truth frames are available"
        )
    return list(range(0, last_anchor + 1, stride))


def update_raw_evaluations(
    files: list[Path],
    times: list[float],
    position: int,
    target_values: dict[str, np.ndarray],
    strides: tuple[int, ...],
    datasets: tuple[str, ...],
    states: dict[int, dict[str, Any]],
    anchor_cache: dict[tuple[int, str], np.ndarray],
) -> None:
    def anchor(anchor_position: int, dataset: str) -> np.ndarray:
        key = (anchor_position, dataset)
        if key not in anchor_cache:
            anchor_cache[key] = read_dataset(files[anchor_position], dataset)
        return anchor_cache[key]

    if position == len(files) - 1 or any(position % stride == 0 for stride in strides):
        for dataset, values in target_values.items():
            anchor_cache[(position, dataset)] = values
    for stride, state in states.items():
        lower = (position // stride) * stride
        upper = lower + stride
        is_anchor = position == lower or position == upper
        if is_anchor:
            predicted_values = target_values
        else:
            tau = (times[position] - times[lower]) / (times[upper] - times[lower])
            predicted_values = {
                dataset: primitive_linear(
                    anchor(lower, dataset),
                    anchor(upper, dataset),
                    tau,
                )
                for dataset in datasets
            }
        evaluate_frame(
            target_values,
            predicted_values,
            state["common_errors"],
            state["common_derived_errors"],
        )
        if not is_anchor:
            state["interior_count"] += 1
            evaluate_frame(
                target_values,
                predicted_values,
                state["interior_errors"],
                state["interior_derived_errors"],
            )
    oldest_required = min((position // stride) * stride for stride in strides)
    for key in tuple(anchor_cache):
        if key[0] < oldest_required:
            del anchor_cache[key]


def finalize_raw_evaluations(
    files: list[Path],
    times: list[float],
    evaluation_positions: list[int],
    states: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for stride, state in states.items():
        stored_positions = raw_anchor_positions(files, evaluation_positions, stride)
        summaries[f"stride_{stride}"] = {
            "stride": stride,
            "stored_frame_count": len(stored_positions),
            "stored_bytes": sum(files[position].stat().st_size for position in stored_positions),
            "stored_times": [times[position] for position in stored_positions],
            "common_frame_count": len(evaluation_positions),
            "interior_frame_count": state["interior_count"],
            "common_errors": finalize_errors(state["common_errors"]),
            "common_derived_errors": finalize_errors(state["common_derived_errors"]),
            "interior_errors": finalize_errors(state["interior_errors"]),
            "interior_derived_errors": finalize_errors(state["interior_derived_errors"]),
        }
    return summaries


def evaluate_raw_strides(
    files: list[Path],
    times: list[float],
    evaluation_positions: list[int],
    strides: tuple[int, ...],
    datasets: tuple[str, ...],
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    for stride in strides:
        raw_anchor_positions(files, evaluation_positions, stride)
    chunks = split_evaluation_positions(evaluation_positions, workers)
    if workers > 1 and len(chunks) > 1:
        states = raw_evaluation_states(strides)
        worker_args = [
            (
                tuple(str(path) for path in files),
                tuple(times),
                chunk,
                strides,
                datasets,
            )
            for chunk in chunks
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(chunks)),
            mp_context=context,
        ) as executor:
            for partial_states in executor.map(raw_evaluation_worker, *zip(*worker_args)):
                merge_raw_evaluation_states(states, partial_states)
        return finalize_raw_evaluations(files, times, evaluation_positions, states)

    states = raw_evaluation_states(strides)
    anchor_cache: dict[tuple[int, str], np.ndarray] = {}

    for position in evaluation_positions:
        target_values = {dataset: read_dataset(files[position], dataset) for dataset in datasets}
        update_raw_evaluations(
            files,
            times,
            position,
            target_values,
            strides,
            datasets,
            states,
            anchor_cache,
        )
    return finalize_raw_evaluations(files, times, evaluation_positions, states)


def evaluate_raw_stride(
    files: list[Path],
    times: list[float],
    evaluation_positions: list[int],
    stride: int,
    datasets: tuple[str, ...],
) -> dict[str, Any]:
    return evaluate_raw_strides(
        files,
        times,
        evaluation_positions,
        (stride,),
        datasets,
    )[f"stride_{stride}"]


def codec_errors_from_manifest(
    scheme: dict[str, Any],
    frame_count: int,
) -> dict[str, Any]:
    errors = scheme.get("errors")
    derived_errors = scheme.get("derived_errors")
    if not isinstance(errors, dict) or not errors:
        raise ValueError("Codec scheme does not contain reusable field errors")
    if not isinstance(derived_errors, dict):
        raise ValueError("Codec scheme does not contain reusable derived errors")
    for group_name, group in (("errors", errors), ("derived_errors", derived_errors)):
        invalid = [
            name
            for name, row in group.items()
            if not isinstance(row, dict)
            or int(row.get("count", 0)) <= 0
            or not math.isfinite(float(row.get("nrmse", math.nan)))
        ]
        if invalid:
            raise ValueError(
                f"Codec scheme contains invalid reusable {group_name}: {', '.join(invalid)}"
            )
    return {
        "frame_count": frame_count,
        "errors": errors,
        "derived_errors": derived_errors,
    }


def codec_frame_map(archive_paths: list[str]) -> dict[int, tuple[Path, int]]:
    frames: dict[int, tuple[Path, int]] = {}
    for value in archive_paths:
        path = Path(value)
        with open_archive(path) as archive:
            metadata = read_metadata(archive)
        for frame_index, sequence in enumerate(metadata["middle_sequences"], start=1):
            sequence = int(sequence)
            if sequence in frames:
                raise ValueError(f"Duplicate DDC sequence {sequence}")
            frames[sequence] = (path, frame_index)
    return frames


def validate_comparison_inputs(
    files: list[Path],
    times: list[float],
    manifest: dict[str, Any],
    frame_map: dict[int, tuple[Path, int]],
    *,
    allow_truth_subset: bool = False,
) -> None:
    sequences = [phdf_sequence(path) for path in files]
    expected_sequences = list(range(sequences[0], sequences[0] + len(sequences)))
    if sequences != expected_sequences:
        raise ValueError("Truth PHDF sequence numbers are not contiguous")
    if any(end <= start for start, end in zip(times, times[1:])):
        raise ValueError("Truth PHDF times are not strictly increasing")
    if manifest.get("format") != "dense_dump_codec_sequence_v1":
        raise ValueError("Codec manifest format is invalid")
    if manifest.get("complete") is not True:
        raise ValueError("Codec manifest is incomplete")
    keyframe_sequences = {phdf_sequence(Path(value)) for value in manifest["keyframes"]}
    covered_sequences = keyframe_sequences | set(frame_map)
    if allow_truth_subset:
        missing_sequences = sorted(set(sequences) - covered_sequences)
        if missing_sequences:
            raise ValueError(
                "Codec manifest does not cover truth subset sequences: "
                + ", ".join(str(value) for value in missing_sequences[:8])
            )
        if not set(sequences).intersection(frame_map):
            raise ValueError("Truth subset contains no DDC residual frames")
    else:
        if int(manifest.get("dense_frame_count", -1)) != len(files):
            raise ValueError("Codec manifest frame count does not match truth")
        if int(manifest.get("middle_frame_count", -1)) != len(frame_map):
            raise ValueError("Codec manifest middle-frame count is inconsistent")
        if covered_sequences != set(sequences):
            raise ValueError("Codec keyframes and residuals do not cover the truth sequence")


def codec_window_storage(
    frame_map: dict[int, tuple[Path, int]],
    truth_bytes: int,
) -> dict[str, Any]:
    archive_paths = sorted({path for path, _ in frame_map.values()})
    keyframe_paths: set[Path] = set()
    for archive_path in archive_paths:
        with open_archive(archive_path) as archive:
            metadata = read_metadata(archive)
        keyframe_paths.add(Path(metadata["start_file"]))
        keyframe_paths.add(Path(metadata["end_file"]))
    archive_bytes = sum(path.stat().st_size for path in archive_paths)
    keyframe_bytes = sum(path.stat().st_size for path in keyframe_paths)
    total_bytes = archive_bytes + keyframe_bytes
    return {
        "scope": "truth_window_dependencies",
        "archive_count": len(archive_paths),
        "archive_paths": [str(path) for path in archive_paths],
        "archive_bytes": archive_bytes,
        "keyframe_count": len(keyframe_paths),
        "keyframe_paths": [str(path) for path in sorted(keyframe_paths)],
        "keyframe_bytes": keyframe_bytes,
        "total_bytes": total_bytes,
        "ratio_vs_dense_truth": truth_bytes / max(total_bytes, 1),
    }


def codec_predictions(
    files: list[Path],
    sequence_to_position: dict[int, int],
    frame_map: dict[int, tuple[Path, int]],
    datasets: tuple[str, ...],
):
    by_archive: dict[Path, list[tuple[int, int]]] = {}
    for sequence, (archive_path, frame_index) in frame_map.items():
        by_archive.setdefault(archive_path, []).append((sequence, frame_index))

    for archive_path, rows in sorted(by_archive.items()):
        with open_archive(archive_path) as archive:
            metadata = read_metadata(archive)
            endpoints = {
                dataset: (
                    read_dataset(Path(metadata["start_file"]), dataset),
                    read_dataset(Path(metadata["end_file"]), dataset),
                )
                for dataset in datasets
            }
            for sequence, frame_index in sorted(rows):
                target_path = files[sequence_to_position[sequence]]
                target_values = {
                    dataset: read_dataset(target_path, dataset)
                    for dataset in datasets
                }
                predicted_values = {
                    dataset: decode_dataset(
                        archive,
                        metadata,
                        dataset,
                        frame_index,
                        start=endpoints[dataset][0],
                        end=endpoints[dataset][1],
                    )
                    for dataset in datasets
                }
                yield sequence_to_position[sequence], target_values, predicted_values


def evaluate_codec(
    files: list[Path],
    sequence_to_position: dict[int, int],
    frame_map: dict[int, tuple[Path, int]],
    datasets: tuple[str, ...],
) -> dict[str, Any]:
    errors: dict[str, ErrorAccumulator] = {}
    derived_errors: dict[str, ErrorAccumulator] = {}
    for _, target_values, predicted_values in codec_predictions(
        files,
        sequence_to_position,
        frame_map,
        datasets,
    ):
        evaluate_frame(
            target_values,
            predicted_values,
            errors,
            derived_errors,
        )
    return {
        "frame_count": len(frame_map),
        "errors": finalize_errors(errors),
        "derived_errors": finalize_errors(derived_errors),
    }


def evaluate_codec_and_raw(
    files: list[Path],
    times: list[float],
    sequence_to_position: dict[int, int],
    frame_map: dict[int, tuple[Path, int]],
    datasets: tuple[str, ...],
    strides: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evaluation_positions = sorted(
        sequence_to_position[sequence] for sequence in frame_map
    )
    for stride in strides:
        raw_anchor_positions(files, evaluation_positions, stride)
    errors: dict[str, ErrorAccumulator] = {}
    derived_errors: dict[str, ErrorAccumulator] = {}
    raw_states = raw_evaluation_states(strides)
    anchor_cache: dict[tuple[int, str], np.ndarray] = {}
    evaluation_positions: list[int] = []
    for position, target_values, predicted_values in codec_predictions(
        files,
        sequence_to_position,
        frame_map,
        datasets,
    ):
        evaluation_positions.append(position)
        evaluate_frame(
            target_values,
            predicted_values,
            errors,
            derived_errors,
        )
        update_raw_evaluations(
            files,
            times,
            position,
            target_values,
            strides,
            datasets,
            raw_states,
            anchor_cache,
        )
    if evaluation_positions != sorted(evaluation_positions):
        raise ValueError("Codec frames are not ordered by sequence")
    codec = {
        "frame_count": len(frame_map),
        "errors": finalize_errors(errors),
        "derived_errors": finalize_errors(derived_errors),
    }
    raw = finalize_raw_evaluations(files, times, evaluation_positions, raw_states)
    return codec, raw


def add_error_ratios(summary: dict[str, Any]) -> None:
    codec_errors = summary["codec"]["errors"]
    codec_derived = summary["codec"]["derived_errors"]
    for raw in summary["raw_cadences"].values():
        raw["codec_over_raw_common_nrmse"] = {
            name: codec_errors[name]["nrmse"] / max(row["nrmse"], 1.0e-30)
            for name, row in raw["common_errors"].items()
            if name in codec_errors
        }
        raw["codec_over_raw_common_derived_nrmse"] = {
            name: codec_derived[name]["nrmse"] / max(row["nrmse"], 1.0e-30)
            for name, row in raw["common_derived_errors"].items()
            if name in codec_derived
        }


def reusable_raw_cadences(
    path: Path,
    *,
    truth_dir: Path,
    files: list[Path],
    times: list[float],
    datasets: tuple[str, ...],
    evaluation_positions: list[int],
    strides: tuple[int, ...],
) -> dict[str, Any]:
    payload = load_manifest(path)
    expected_sequences = [phdf_sequence(files[position]) for position in evaluation_positions]
    checks = {
        "truth_dir": Path(payload.get("truth_dir", "")).resolve() == truth_dir.resolve(),
        "truth_frame_count": int(payload.get("truth_frame_count", -1)) == len(files),
        "truth_bytes": int(payload.get("truth_bytes", -1))
        == sum(file.stat().st_size for file in files),
        "truth_times": payload.get("truth_times") == times,
        "datasets": tuple(payload.get("datasets", ())) == datasets,
        "common_evaluation_positions": payload.get("common_evaluation_positions")
        == evaluation_positions,
        "common_evaluation_sequences": payload.get("common_evaluation_sequences")
        == expected_sequences,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(
            f"Raw baseline fingerprint mismatch in {path}: {', '.join(failed)}"
        )
    raw_cadences = payload.get("raw_cadences", {})
    expected_names = [f"stride_{stride}" for stride in strides]
    missing = [name for name in expected_names if name not in raw_cadences]
    if missing:
        raise ValueError(f"Raw baseline is missing cadences: {', '.join(missing)}")
    return {name: raw_cadences[name] for name in expected_names}


def main() -> int:
    args = parse_args()
    if args.raw_workers < 1:
        raise ValueError("--raw-workers must be at least 1")
    datasets = parse_datasets(args.datasets)
    strides = parse_strides(args.raw_strides)
    files = sequence_files(args.truth_dir)
    times = [phdf_time(path) for path in files]
    sequence_to_position = file_index(files)
    manifest = load_manifest(args.codec_manifest)
    scheme_name = args.scheme or next(iter(sorted(manifest["codec_schemes"])))
    scheme = manifest["codec_schemes"][scheme_name]
    full_frame_map = codec_frame_map(scheme["archive_paths"])
    validate_comparison_inputs(
        files,
        times,
        manifest,
        full_frame_map,
        allow_truth_subset=args.allow_truth_subset,
    )
    if args.allow_truth_subset:
        truth_sequences = set(sequence_to_position)
        frame_map = {
            sequence: row
            for sequence, row in full_frame_map.items()
            if sequence in truth_sequences
        }
    else:
        frame_map = full_frame_map
    missing = sorted(set(frame_map) - set(sequence_to_position))
    if missing:
        raise KeyError(f"DDC sequences missing from truth directory: {missing}")
    evaluation_positions = sorted(sequence_to_position[sequence] for sequence in frame_map)

    if args.raw_baseline_json:
        raw_cadences = reusable_raw_cadences(
            args.raw_baseline_json,
            truth_dir=args.truth_dir,
            files=files,
            times=times,
            datasets=datasets,
            evaluation_positions=evaluation_positions,
            strides=strides,
        )
        codec = (
            codec_errors_from_manifest(scheme, len(frame_map))
            if args.reuse_manifest_errors
            else evaluate_codec(files, sequence_to_position, frame_map, datasets)
        )
    elif args.reuse_manifest_errors:
        codec = codec_errors_from_manifest(scheme, len(frame_map))
        raw_cadences = evaluate_raw_strides(
            files,
            times,
            evaluation_positions,
            strides,
            datasets,
            workers=args.raw_workers,
        )
    elif args.raw_workers > 1:
        codec = evaluate_codec(files, sequence_to_position, frame_map, datasets)
        raw_cadences = evaluate_raw_strides(
            files,
            times,
            evaluation_positions,
            strides,
            datasets,
            workers=args.raw_workers,
        )
    else:
        codec, raw_cadences = evaluate_codec_and_raw(
            files,
            times,
            sequence_to_position,
            frame_map,
            datasets,
            strides,
        )
    summary = {
        "truth_dir": str(args.truth_dir),
        "truth_frame_count": len(files),
        "truth_bytes": sum(path.stat().st_size for path in files),
        "truth_times": times,
        "truth_dt_min": min(end - start for start, end in zip(times, times[1:])),
        "truth_dt_max": max(end - start for start, end in zip(times, times[1:])),
        "datasets": datasets,
        "common_evaluation_positions": evaluation_positions,
        "common_evaluation_sequences": [phdf_sequence(files[position]) for position in evaluation_positions],
        "codec_scheme": scheme_name,
        "codec_bits": int(scheme["bits"]),
        "codec_dataset_bits": dict(scheme.get("dataset_bits", {})),
        "codec_scale_percentile": float(scheme.get("scale_percentile", 100.0)),
        "codec_dataset_scale_percentiles": dict(
            scheme.get("dataset_scale_percentiles", {})
        ),
        "codec_keyframe_stride": int(manifest["keyframe_stride"]),
        "codec_archive_backend": scheme.get("archive_backend", "zip-deflate"),
        "codec_channel_chunk_frames": scheme.get("channel_chunk_frames"),
        "codec_archive_compression_level": scheme.get(
            "archive_compression_level"
        ),
        "codec_storage": {
            "scope": "full_manifest",
            "keyframe_bytes": manifest["storage"]["keyframe_bytes"],
            "archive_bytes": scheme["archive_size_bytes"],
            "total_bytes": scheme["total_with_keyframes_bytes"],
            "ratio_vs_dense_phdf": scheme["ratio_vs_dense_phdf"],
        },
        "codec_window_storage": codec_window_storage(
            frame_map,
            sum(path.stat().st_size for path in files),
        )
        if args.allow_truth_subset
        else None,
        "truth_subset": args.allow_truth_subset,
        "manifest_dense_frame_count": int(manifest["dense_frame_count"]),
        "codec_lossless_repack": {
            "backend": manifest.get("keyframe_storage", {}).get("backend"),
            "member_crc_verified": manifest.get("lossless_keyframe_repack", {}).get(
                "all_chunks_verified"
            ),
            "maximum_dependency_chain": manifest.get("keyframe_storage", {}).get(
                "maximum_dependency_chain"
            ),
        },
        "codec_error_source": "manifest" if args.reuse_manifest_errors else "decoded",
        "codec": codec,
        "raw_cadences": raw_cadences,
        "raw_baseline_source": str(args.raw_baseline_json.resolve())
        if args.raw_baseline_json
        else None,
    }
    add_error_ratios(summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"truth frames={len(files)} common={len(evaluation_positions)}")
    print(
        f"codec {scheme_name}: ratio={summary['codec_storage']['ratio_vs_dense_phdf']:.3f}x "
        f"rho={summary['codec']['errors'].get('rho', {}).get('nrmse', float('nan')):.4g} "
        f"B1={summary['codec']['errors'].get('B1', {}).get('nrmse', float('nan')):.4g}"
    )
    for name, raw in summary["raw_cadences"].items():
        print(
            f"{name}: frames={raw['stored_frame_count']} "
            f"rho={raw['common_errors'].get('rho', {}).get('nrmse', float('nan')):.4g} "
            f"B1={raw['common_errors'].get('B1', {}).get('nrmse', float('nan')):.4g} "
            f"codec/raw rho={raw['codec_over_raw_common_nrmse'].get('rho', float('nan')):.4g}"
        )
    print(f"summary: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
