#!/usr/bin/env python3
"""Prototype keyframe + quantized-residual compression for dense KHARMA PHDF dumps.

The codec keeps the first and last PHDF dumps in an interval as full keyframes.
Intermediate frames are represented as residuals from endpoint-linear prediction.
This script evaluates storage/error tradeoffs and writes compressed residual
archives for offline testing; it does not replace KHARMA/Parthenon output yet.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import (
    dequantize_residual,
    inverse_transformed,
    linear_predictor,
    pack_int4,
    quantize_residual,
    repack_zip_to_channel_bzip2,
    transformed,
)
from dense_dump_codec.tiled import dequantize_tiled, parse_dataset_tiles, quantize_tiled


PHDF_RE = re.compile(r"\.out0\.(\d+)\.phdf$")
DEFAULT_DATASETS = (
    "prims.rho",
    "prims.u",
    "prims.uvec",
    "prims.B",
    "cons.fB",
    "divB",
)
EPS = 1.0e-30


def require_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required; activate the project Spack environment") from exc
    return h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tile-shapes", default="",
                        help="Local scales, e.g. prims.u=8x16x32 in phi,theta,r order")
    parser.add_argument("--segment-dir", type=Path, required=True, help="Directory containing one dense PHDF sequence.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dense_dump_codec_smoke"))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--bits", default="4,8,16", help="Comma-separated quantization bit depths to test.")
    parser.add_argument(
        "--dataset-bits",
        default="",
        help="Optional DATASET=BITS overrides applied to each base --bits scheme.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("frame", "block-channel"),
        default="block-channel",
        help="Scale residuals globally per frame or independently per meshblock/channel.",
    )
    parser.add_argument(
        "--scale-percentile",
        type=float,
        default=100.0,
        help=(
            "Residual magnitude percentile used for each quantization scale. Values below 100 "
            "use a sparse exact exception channel unless --discard-outliers is set."
        ),
    )
    parser.add_argument(
        "--dataset-scale-percentiles",
        default="",
        help="Optional DATASET=PERCENTILE overrides applied to quantization scales.",
    )
    parser.add_argument(
        "--discard-outliers",
        action="store_true",
        help="Clip values outside the scale percentile instead of preserving them exactly.",
    )
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
    parser.add_argument("--start-sequence", type=int, help="Optional first PHDF sequence number in the GOP.")
    parser.add_argument("--end-sequence", type=int, help="Optional last PHDF sequence number in the GOP.")
    parser.add_argument("--max-middle-frames", type=int, default=0, help="Limit intermediate frames; 0 uses all.")
    parser.add_argument("--output-json", type=Path, help="Optional summary JSON path.")
    parser.add_argument("--skip-archives", action="store_true", help="Evaluate errors without writing residual archives.")
    return parser.parse_args()


def parse_bits(value: str) -> tuple[int, ...]:
    bits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    invalid = [item for item in bits if not 4 <= item <= 16]
    if invalid:
        raise ValueError(f"Supported --bits values are integers in [4,16], got {invalid}")
    return bits


def parse_datasets(value: str) -> tuple[str, ...]:
    datasets = tuple(item.strip() for item in value.split(",") if item.strip())
    if not datasets:
        raise ValueError("--datasets must include at least one dataset")
    return datasets


def parse_dataset_bits(value: str, datasets: tuple[str, ...]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if not value.strip():
        return overrides
    for item in value.split(","):
        dataset, separator, bits_value = item.partition("=")
        dataset = dataset.strip()
        if not separator or not dataset or not bits_value.strip():
            raise ValueError("--dataset-bits must use comma-separated DATASET=BITS assignments")
        if dataset not in datasets:
            raise ValueError(f"--dataset-bits contains unknown dataset {dataset!r}")
        if dataset in overrides:
            raise ValueError(f"duplicate --dataset-bits assignment for {dataset}")
        bits = int(bits_value)
        if not 4 <= bits <= 16:
            raise ValueError("--dataset-bits values must be in [4, 16]")
        overrides[dataset] = bits
    return overrides


def dataset_bits_for_scheme(
    datasets: tuple[str, ...],
    base_bits: int,
    overrides: dict[str, int],
) -> dict[str, int]:
    return {dataset: overrides.get(dataset, base_bits) for dataset in datasets}


def parse_dataset_scale_percentiles(
    value: str,
    datasets: tuple[str, ...],
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not value.strip():
        return overrides
    for item in value.split(","):
        dataset, separator, percentile_value = item.partition("=")
        dataset = dataset.strip()
        if not separator or not dataset or not percentile_value.strip():
            raise ValueError(
                "--dataset-scale-percentiles must use comma-separated "
                "DATASET=PERCENTILE assignments"
            )
        if dataset not in datasets:
            raise ValueError(
                f"--dataset-scale-percentiles contains unknown dataset {dataset!r}"
            )
        if dataset in overrides:
            raise ValueError(
                f"duplicate --dataset-scale-percentiles assignment for {dataset}"
            )
        percentile = float(percentile_value)
        if not 0.0 < percentile <= 100.0:
            raise ValueError("--dataset-scale-percentiles values must be in (0, 100]")
        overrides[dataset] = percentile
    return overrides


def dataset_scale_percentiles_for_scheme(
    datasets: tuple[str, ...],
    default_percentile: float,
    overrides: dict[str, float],
) -> dict[str, float]:
    return {
        dataset: overrides.get(dataset, default_percentile)
        for dataset in datasets
    }


def mixed_scheme_tag(base_bits: int, allocation: dict[str, int]) -> str:
    short_names = {
        "prims.rho": "rho",
        "prims.u": "u",
        "prims.uvec": "uvec",
        "prims.B": "B",
        "cons.fB": "fB",
        "divB": "divB",
    }
    overrides = [
        f"{short_names.get(dataset, dataset.replace('.', '_'))}{bits}"
        for dataset, bits in allocation.items()
        if bits != base_bits
    ]
    return f"q{base_bits}" if not overrides else f"q{base_bits}_" + "_".join(overrides)


def mixed_percentile_tag(
    default_percentile: float,
    allocation: dict[str, float],
) -> str:
    short_names = {
        "prims.rho": "rho",
        "prims.u": "u",
        "prims.uvec": "uvec",
        "prims.B": "B",
        "cons.fB": "fB",
        "divB": "divB",
    }
    overrides = [
        f"{short_names.get(dataset, dataset.replace('.', '_'))}{percentile_tag(value)}"
        for dataset, value in allocation.items()
        if value != default_percentile
    ]
    return "" if not overrides else "_" + "_".join(overrides)


def phdf_sequence(path: Path) -> int:
    match = PHDF_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer sequence number from {path}")
    return int(match.group(1))


def phdf_time(path: Path) -> float:
    h5py = require_h5py()
    with h5py.File(path, "r") as handle:
        if "Info" in handle and "Time" in handle["Info"].attrs:
            return float(handle["Info"].attrs["Time"])
        if "Time" in handle.attrs:
            return float(handle.attrs["Time"])
    raise KeyError(f"Cannot find PHDF time metadata in {path}")


def sequence_files(
    segment_dir: Path,
    start_sequence: int | None = None,
    end_sequence: int | None = None,
) -> list[Path]:
    files = sorted((path for path in segment_dir.glob("*.phdf") if path.is_file()), key=phdf_sequence)
    if start_sequence is not None:
        files = [path for path in files if phdf_sequence(path) >= start_sequence]
    if end_sequence is not None:
        files = [path for path in files if phdf_sequence(path) <= end_sequence]
    if len(files) < 3:
        raise ValueError(f"Need at least 3 PHDF files in {segment_dir}, found {len(files)}")
    return files


def read_dataset(path: Path, name: str) -> np.ndarray:
    h5py = require_h5py()
    with h5py.File(path, "r") as handle:
        if name not in handle:
            raise KeyError(f"{name!r} not found in {path}")
        return np.asarray(handle[name][...], dtype=np.float32)


def dataset_hdf5_info(path: Path, datasets: tuple[str, ...]) -> dict[str, Any]:
    h5py = require_h5py()
    info: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for name in datasets:
            if name not in handle:
                continue
            dataset = handle[name]
            info[name] = {
                "shape": tuple(int(value) for value in dataset.shape),
                "dtype": str(dataset.dtype),
                "chunks": tuple(int(value) for value in dataset.chunks) if dataset.chunks is not None else None,
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
                "shuffle": bool(dataset.shuffle),
                "scaleoffset": dataset.scaleoffset,
            }
    return info


def file_size(path: Path) -> int:
    return int(path.stat().st_size)


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


@dataclass
class ErrorAccumulator:
    sse: float = 0.0
    sae: float = 0.0
    target_sse: float = 0.0
    target_abs: float = 0.0
    count: int = 0
    min_value: float = math.inf
    max_abs_error: float = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred64 = np.asarray(pred, dtype=np.float64)
        target64 = np.asarray(target, dtype=np.float64)
        diff = pred64 - target64
        self.sse += float(np.square(diff).sum())
        self.sae += float(np.abs(diff).sum())
        self.target_sse += float(np.square(target64).sum())
        self.target_abs += float(np.abs(target64).sum())
        self.count += int(target64.size)
        self.min_value = min(self.min_value, float(np.nanmin(pred64)))
        self.max_abs_error = max(self.max_abs_error, float(np.nanmax(np.abs(diff))))

    def finalize(self) -> dict[str, float | int]:
        rmse = math.sqrt(self.sse / max(self.count, 1))
        target_rms = math.sqrt(self.target_sse / max(self.count, 1))
        return {
            "rmse": rmse,
            "target_rms": target_rms,
            "nrmse": rmse / max(target_rms, EPS),
            "rel_l1": self.sae / max(self.target_abs, EPS),
            "max_abs_error": self.max_abs_error,
            "min_reconstructed_value": self.min_value,
            "count": self.count,
            "sse": self.sse,
            "sae": self.sae,
            "target_sse": self.target_sse,
            "target_abs": self.target_abs,
        }


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_array_to_zip(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    archive.writestr(name, npy_bytes(array))


def archive_member_name(dataset: str, frame_index: int, suffix: str) -> str:
    safe_dataset = dataset.replace("/", "__").replace(".", "_")
    return f"frames/{frame_index:05d}/{safe_dataset}_{suffix}.npy"


def percentile_tag(value: float) -> str:
    return f"p{value:g}".replace(".", "p")


def channel_names(dataset: str) -> tuple[str, ...]:
    if dataset == "prims.rho":
        return ("rho",)
    if dataset == "prims.u":
        return ("u",)
    if dataset == "prims.uvec":
        return ("u1", "u2", "u3")
    if dataset == "prims.B":
        return ("B1", "B2", "B3")
    if dataset == "cons.fB":
        return ("fB1", "fB2", "fB3")
    if dataset == "divB":
        return ("divB",)
    return (dataset,)


def update_channel_errors(
    accumulators: dict[str, ErrorAccumulator],
    dataset: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> None:
    names = channel_names(dataset)
    if len(names) == 1:
        accumulators.setdefault(names[0], ErrorAccumulator()).update(pred, target)
        return
    for idx, name in enumerate(names):
        accumulators.setdefault(name, ErrorAccumulator()).update(pred[:, idx], target[:, idx])


def primitive_derived_fields(prims: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rho = np.maximum(prims["prims.rho"], EPS)
    u_internal = np.maximum(prims["prims.u"], EPS)
    uvec = prims["prims.uvec"]
    bvec = prims["prims.B"]
    velocity_sq = np.square(uvec, dtype=np.float64).sum(axis=1)
    b_sq = np.square(bvec, dtype=np.float64).sum(axis=1)
    b_sq_safe = np.maximum(b_sq, EPS)
    return {
        "velocity_sq": velocity_sq,
        "log_magnetic_pressure_proxy": np.log(np.maximum(0.5 * b_sq, EPS)),
        "log_magnetization_proxy": np.log(b_sq_safe) - np.log(rho),
        "log_plasma_beta_proxy": np.log(np.maximum(2.0 * u_internal, EPS)) - np.log(b_sq_safe),
    }


def finalize_errors(accumulators: dict[str, ErrorAccumulator]) -> dict[str, Any]:
    return {name: acc.finalize() for name, acc in sorted(accumulators.items())}


def evaluate_codec(args: argparse.Namespace) -> dict[str, Any]:
    wall_started = time.time()
    datasets = parse_datasets(args.datasets)
    dataset_tiles = parse_dataset_tiles(getattr(args, "dataset_tile_shapes", ""), datasets)
    if dataset_tiles and args.scale_mode != "block-channel":
        raise ValueError("Local tile scales require block-channel scaling")
    bits_values = parse_bits(args.bits)
    dataset_bit_overrides = parse_dataset_bits(
        getattr(args, "dataset_bits", ""),
        datasets,
    )
    dataset_percentile_overrides = parse_dataset_scale_percentiles(
        getattr(args, "dataset_scale_percentiles", ""),
        datasets,
    )
    dataset_scale_percentiles = dataset_scale_percentiles_for_scheme(
        datasets,
        args.scale_percentile,
        dataset_percentile_overrides,
    )
    allocations = {
        bits: dataset_bits_for_scheme(datasets, bits, dataset_bit_overrides)
        for bits in bits_values
    }
    archive_backend = getattr(args, "archive_backend", "zip-deflate")
    channel_chunk_frames = int(getattr(args, "channel_chunk_frames", 5))
    channel_compression_level = int(
        getattr(args, "channel_compression_level", 9)
    )
    archive_workers = int(getattr(args, "archive_workers", 1))
    if archive_backend not in {
        "zip-deflate",
        "channel-bzip2",
        "channel-bzip2-delta",
        "channel-bzip2-delta-shuffle",
        "channel-bzip2-adaptive",
    }:
        raise ValueError(f"Unsupported archive backend: {archive_backend}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sequence_files(
        args.segment_dir,
        getattr(args, "start_sequence", None),
        getattr(args, "end_sequence", None),
    )
    start_file = files[0]
    end_file = files[-1]
    start_seq = phdf_sequence(start_file)
    end_seq = phdf_sequence(end_file)
    middle_files = files[1:-1]
    if args.max_middle_frames > 0:
        middle_files = middle_files[: args.max_middle_frames]
    if not middle_files:
        raise ValueError("No middle frames selected")

    dataset_info = dataset_hdf5_info(start_file, datasets)
    original_selected_files = [start_file, *middle_files, end_file]
    original_size = sum(file_size(path) for path in original_selected_files)
    full_interval_original_size = sum(file_size(path) for path in files)
    key_size = file_size(start_file) + file_size(end_file)

    predictor_errors: dict[str, ErrorAccumulator] = {}
    codec_errors = {bits: {} for bits in bits_values}
    derived_predictor_errors: dict[str, ErrorAccumulator] = {}
    derived_codec_errors = {bits: {} for bits in bits_values}
    raw_residual_sse = {bits: 0.0 for bits in bits_values}
    raw_target_sse = {bits: 0.0 for bits in bits_values}
    exception_counts = {bits: 0 for bits in bits_values}

    archives: dict[int, zipfile.ZipFile] = {}
    archive_paths: dict[int, Path] = {}
    if not args.skip_archives:
        for bits in bits_values:
            scheme_tag = mixed_scheme_tag(bits, allocations[bits]) + mixed_percentile_tag(
                args.scale_percentile,
                dataset_scale_percentiles,
            )
            path = args.output_dir / (
                f"gop_{start_seq:05d}_{end_seq:05d}_linear_residual_{scheme_tag}_{args.scale_mode}_"
                f"{percentile_tag(args.scale_percentile)}.ddc"
            )
            archive_paths[bits] = path
            archives[bits] = zipfile.ZipFile(
                path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=args.compression_level,
            )

    try:
        middle_sequences = [phdf_sequence(path) for path in middle_files]
        start_time = phdf_time(start_file)
        end_time = phdf_time(end_file)
        middle_times = [phdf_time(path) for path in middle_files]
        if not start_time < end_time:
            raise ValueError(f"Non-increasing GOP endpoint times: {start_time} >= {end_time}")
        middle_taus = [
            (frame_time - start_time) / (end_time - start_time)
            for frame_time in middle_times
        ]
        if any(not 0.0 < tau < 1.0 for tau in middle_taus):
            raise ValueError(f"Middle PHDF times fall outside GOP endpoints: {middle_taus}")
        archive_metadata = {
            "codec": "ddc_endpoint_residual_v2",
            "format_version": 2,
            "segment_dir": str(args.segment_dir),
            "start_file": str(start_file),
            "end_file": str(end_file),
            "start_sequence": start_seq,
            "end_sequence": end_seq,
            "start_time": start_time,
            "end_time": end_time,
            "middle_files": [str(path) for path in middle_files],
            "middle_sequences": middle_sequences,
            "middle_times": middle_times,
            "middle_taus": middle_taus,
            "time_coordinate": "Info/Time",
            "datasets": datasets,
            "log_space_datasets": sorted({"prims.rho", "prims.u"}.intersection(datasets)),
            "scale_mode": args.scale_mode,
            "scale_percentile": args.scale_percentile,
            "dataset_scale_percentiles": dataset_scale_percentiles,
            "dataset_tile_shapes": dataset_tiles,
            "preserve_outliers": not args.discard_outliers,
            "compression": "zip_deflate",
            "compression_level": args.compression_level,
            "keyframes_stored_externally": True,
        }
        for bits, archive in archives.items():
            archive.writestr(
                "metadata.json",
                json.dumps(
                    archive_metadata
                    | {
                        "bits": bits,
                        "dataset_bits": allocations[bits],
                        "dataset_scale_percentiles": dataset_scale_percentiles,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )

        key_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for dataset in datasets:
            key_cache[dataset] = (read_dataset(start_file, dataset), read_dataset(end_file, dataset))

        for frame_idx, frame_path in enumerate(middle_files, start=1):
            tau = middle_taus[frame_idx - 1]
            pred_prims: dict[str, np.ndarray] = {}
            target_prims: dict[str, np.ndarray] = {}
            codec_prims: dict[int, dict[str, np.ndarray]] = {bits: {} for bits in bits_values}

            for dataset in datasets:
                start_array, end_array = key_cache[dataset]
                target = read_dataset(frame_path, dataset)
                pred, pred_t, _ = linear_predictor(start_array, end_array, tau, dataset)
                update_channel_errors(predictor_errors, dataset, pred, target)
                if dataset.startswith("prims."):
                    pred_prims[dataset] = pred
                    target_prims[dataset] = target

                target_t = transformed(target, dataset)
                residual = (target_t - pred_t).astype(np.float32, copy=False)
                raw_target_sse_all = float(np.square(target_t.astype(np.float64)).sum())

                for bits in bits_values:
                    dataset_bits = allocations[bits][dataset]
                    if dataset in dataset_tiles:
                        payload = quantize_tiled(
                            residual, dataset_tiles[dataset], dataset_bits,
                            scale_percentile=dataset_scale_percentiles[dataset],
                            preserve_outliers=not args.discard_outliers,
                        )
                        dequantized = dequantize_tiled(payload, dataset_tiles[dataset])
                    else:
                        payload = quantize_residual(
                            residual, dataset_bits, args.scale_mode,
                            scale_percentile=dataset_scale_percentiles[dataset],
                            preserve_outliers=not args.discard_outliers,
                        )
                        dequantized = dequantize_residual(payload)
                    recon = inverse_transformed(pred_t + dequantized, dataset)
                    update_channel_errors(codec_errors[bits], dataset, recon, target)
                    raw_residual_sse[bits] += float(np.square((dequantized - residual).astype(np.float64)).sum())
                    raw_target_sse[bits] += raw_target_sse_all
                    exception_counts[bits] += payload.exception_count
                    if dataset.startswith("prims."):
                        codec_prims[bits][dataset] = recon

                    if bits in archives:
                        archive = archives[bits]
                        if dataset_bits == 4:
                            stored_q = pack_int4(payload.values)
                            write_array_to_zip(archive, archive_member_name(dataset, frame_idx, "q4_packed"), stored_q)
                            write_array_to_zip(
                                archive,
                                archive_member_name(dataset, frame_idx, "q4_shape"),
                                np.asarray(payload.values.shape, dtype=np.int64),
                            )
                        else:
                            write_array_to_zip(
                                archive,
                                archive_member_name(dataset, frame_idx, f"q{dataset_bits}"),
                                payload.values,
                            )
                        write_array_to_zip(
                            archive,
                            archive_member_name(dataset, frame_idx, "scale"),
                            payload.scale,
                        )
                        if payload.exception_count:
                            write_array_to_zip(
                                archive,
                                archive_member_name(dataset, frame_idx, "exception_indices"),
                                payload.exception_indices,
                            )
                            write_array_to_zip(
                                archive,
                                archive_member_name(dataset, frame_idx, "exception_values"),
                                payload.exception_values,
                            )

            if {"prims.rho", "prims.u", "prims.uvec", "prims.B"}.issubset(pred_prims):
                target_derived = primitive_derived_fields(target_prims)
                pred_derived = primitive_derived_fields(pred_prims)
                for name, target_value in target_derived.items():
                    derived_predictor_errors.setdefault(name, ErrorAccumulator()).update(pred_derived[name], target_value)
                for bits in bits_values:
                    if {"prims.rho", "prims.u", "prims.uvec", "prims.B"}.issubset(codec_prims[bits]):
                        codec_derived = primitive_derived_fields(codec_prims[bits])
                        for name, target_value in target_derived.items():
                            derived_codec_errors[bits].setdefault(name, ErrorAccumulator()).update(codec_derived[name], target_value)
    finally:
        for archive in archives.values():
            archive.close()

    archive_repack_stats: dict[int, dict[str, Any]] = {}
    if archive_backend in {
        "channel-bzip2",
        "channel-bzip2-delta",
        "channel-bzip2-delta-shuffle",
        "channel-bzip2-adaptive",
    }:
        for bits, path in archive_paths.items():
            staged_path = path.with_name(f".{path.name}.{archive_backend}")
            staged_path.unlink(missing_ok=True)
            try:
                archive_repack_stats[bits] = repack_zip_to_channel_bzip2(
                    path,
                    staged_path,
                    chunk_frames=channel_chunk_frames,
                    compression_level=channel_compression_level,
                    workers=archive_workers,
                    temporal_delta=archive_backend == "channel-bzip2-delta",
                    temporal_delta_shuffle=(
                        archive_backend == "channel-bzip2-delta-shuffle"
                    ),
                    adaptive_temporal_order=(
                        archive_backend == "channel-bzip2-adaptive"
                    ),
                    metadata_updates={
                        "compression": archive_backend.replace("-", "_"),
                        "compression_level": channel_compression_level,
                        "channel_chunk_frames": channel_chunk_frames,
                    },
                    overwrite=True,
                )
                staged_path.replace(path)
            finally:
                staged_path.unlink(missing_ok=True)

    archive_sizes = {bits: file_size(path) for bits, path in archive_paths.items() if path.exists()}
    codec_summaries: dict[str, Any] = {}
    for bits in bits_values:
        archive_size = archive_sizes.get(bits, 0)
        total_size = key_size + archive_size
        residual_quant_nrmse = math.sqrt(raw_residual_sse[bits] / max(raw_target_sse[bits], EPS))
        scheme_tag = mixed_scheme_tag(bits, allocations[bits]) + mixed_percentile_tag(
            args.scale_percentile,
            dataset_scale_percentiles,
        )
        codec_summaries[f"linear_residual_{scheme_tag}_{args.scale_mode}"] = {
            "bits": bits,
            "dataset_tile_shapes": dataset_tiles,
            "dataset_bits": allocations[bits],
            "scale_percentile": args.scale_percentile,
            "dataset_scale_percentiles": dataset_scale_percentiles,
            "preserve_outliers": not args.discard_outliers,
            "archive_backend": archive_backend,
            "channel_chunk_frames": (
                channel_chunk_frames
                if archive_backend
                in {
                    "channel-bzip2",
                    "channel-bzip2-delta",
                    "channel-bzip2-delta-shuffle",
                    "channel-bzip2-adaptive",
                }
                else None
            ),
            "archive_compression_level": (
                channel_compression_level
                if archive_backend
                in {
                    "channel-bzip2",
                    "channel-bzip2-delta",
                    "channel-bzip2-delta-shuffle",
                    "channel-bzip2-adaptive",
                }
                else args.compression_level
            ),
            "archive_repack": archive_repack_stats.get(bits),
            "exception_count": exception_counts[bits],
            "archive_path": str(archive_paths.get(bits)) if bits in archive_paths else None,
            "archive_size_bytes": archive_size,
            "archive_size_mib": mib(archive_size),
            "estimated_total_with_keyframes_bytes": total_size,
            "estimated_total_with_keyframes_mib": mib(total_size),
            "ratio_vs_selected_dense_phdf": original_size / max(total_size, 1),
            "ratio_vs_middle_dense_phdf": sum(file_size(path) for path in middle_files) / max(archive_size, 1)
            if archive_size
            else None,
            "residual_quantization_nrmse_in_encoding_space": residual_quant_nrmse,
            "errors": finalize_errors(codec_errors[bits]),
            "derived_errors": finalize_errors(derived_codec_errors[bits]),
        }

    summary = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - wall_started,
        "segment_dir": str(args.segment_dir),
        "selected_files": [str(path) for path in original_selected_files],
        "full_interval_file_count": len(files),
        "selected_middle_file_count": len(middle_files),
        "datasets": datasets,
        "dataset_hdf5_info": dataset_info,
        "storage": {
            "selected_dense_phdf_bytes": original_size,
            "selected_dense_phdf_mib": mib(original_size),
            "full_interval_dense_phdf_bytes": full_interval_original_size,
            "full_interval_dense_phdf_mib": mib(full_interval_original_size),
            "selected_keyframe_bytes": key_size,
            "selected_keyframe_mib": mib(key_size),
            "keyframe_only_ratio_vs_selected_dense_phdf": original_size / max(key_size, 1),
        },
        "linear_predictor": {
            "estimated_total_with_keyframes_bytes": key_size,
            "estimated_total_with_keyframes_mib": mib(key_size),
            "ratio_vs_selected_dense_phdf": original_size / max(key_size, 1),
            "errors": finalize_errors(predictor_errors),
            "derived_errors": finalize_errors(derived_predictor_errors),
        },
        "codec_schemes": codec_summaries,
    }
    output_json = args.output_json or (
        args.output_dir / f"gop_{start_seq:05d}_{end_seq:05d}_codec_summary.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["output_json"] = str(output_json)
    return summary


def print_compact_summary(summary: dict[str, Any]) -> None:
    print(f"segment: {summary['segment_dir']}")
    print(f"frames: {summary['selected_middle_file_count']} middle + 2 key")
    storage = summary["storage"]
    print(
        "dense_selected={:.1f} MiB keyframes={:.1f} MiB key_only_ratio={:.2f}x".format(
            storage["selected_dense_phdf_mib"],
            storage["selected_keyframe_mib"],
            storage["keyframe_only_ratio_vs_selected_dense_phdf"],
        )
    )
    linear = summary["linear_predictor"]
    rho = linear["errors"].get("rho", {}).get("nrmse")
    b1 = linear["errors"].get("B1", {}).get("nrmse")
    fb1 = linear["errors"].get("fB1", {}).get("nrmse")
    print(f"linear key-only nrmse: rho={rho:.4g} B1={b1:.4g} fB1={fb1:.4g}")
    for name, scheme in summary["codec_schemes"].items():
        errors = scheme["errors"]
        print(
            "{}: total={:.1f} MiB ratio={:.2f}x archive={:.1f} MiB q_nrmse={:.3g} "
            "rho={:.3g} B1={:.3g} fB1={:.3g}".format(
                name,
                scheme["estimated_total_with_keyframes_mib"],
                scheme["ratio_vs_selected_dense_phdf"],
                scheme["archive_size_mib"],
                scheme["residual_quantization_nrmse_in_encoding_space"],
                errors.get("rho", {}).get("nrmse", float("nan")),
                errors.get("B1", {}).get("nrmse", float("nan")),
                errors.get("fB1", {}).get("nrmse", float("nan")),
            )
        )
    print(f"summary: {summary['output_json']}")


def main() -> None:
    args = parse_args()
    summary = evaluate_codec(args)
    print_compact_summary(summary)


if __name__ == "__main__":
    main()
