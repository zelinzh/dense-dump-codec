#!/usr/bin/env python3
"""Decode a prototype dump-codec residual archive into an approximate PHDF dump."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dense_dump_codec import (  # noqa: E402
    QuantizedResidual,
    dequantize_residual,
    inverse_transformed,
    linear_predictor,
    open_archive,
    transformed,
    unpack_int4,
)
from prototype_dump_codec import (  # noqa: E402
    archive_member_name,
    parse_datasets,
    read_dataset,
    require_h5py,
)
from dense_dump_codec.tiled import dequantize_tiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Residual archive written by prototype_dump_codec.py.")
    parser.add_argument("--frame-index", type=int, required=True, help="1-based middle frame index inside the archive.")
    parser.add_argument("--output-phdf", type=Path, required=True)
    parser.add_argument(
        "--template-phdf",
        type=Path,
        help="PHDF to copy as structure template. Defaults to the GOP start keyframe.",
    )
    parser.add_argument("--datasets", help="Comma-separated datasets to decode; defaults to archive metadata.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def np_load_member(archive: Any, name: str) -> np.ndarray:
    with archive.open(name, "r") as handle:
        return np.load(BytesIO(handle.read()), allow_pickle=False)


def read_metadata(archive: Any) -> dict[str, Any]:
    with archive.open("metadata.json", "r") as handle:
        return json.loads(handle.read().decode("utf-8"))


def read_quantized(archive: Any, dataset: str, frame_index: int, bits: int) -> np.ndarray:
    if bits == 4:
        packed = np_load_member(archive, archive_member_name(dataset, frame_index, "q4_packed"))
        shape = tuple(int(value) for value in np_load_member(archive, archive_member_name(dataset, frame_index, "q4_shape")))
        return unpack_int4(packed, shape)
    return np_load_member(archive, archive_member_name(dataset, frame_index, f"q{bits}"))


def dataset_bits(metadata: dict[str, Any], dataset: str) -> int:
    allocation = metadata.get("dataset_bits", {})
    return int(allocation.get(dataset, metadata["bits"]))


def decode_dataset(
    archive: Any,
    metadata: dict[str, Any],
    dataset: str,
    frame_index: int,
    *,
    start: np.ndarray | None = None,
    end: np.ndarray | None = None,
    start_transformed: np.ndarray | None = None,
    end_transformed: np.ndarray | None = None,
) -> np.ndarray:
    bits = dataset_bits(metadata, dataset)
    if start is None:
        start = read_dataset(Path(metadata["start_file"]), dataset)
    if end is None:
        end = read_dataset(Path(metadata["end_file"]), dataset)
    tau = frame_tau(archive, metadata, (dataset,), frame_index)
    if start_transformed is None or end_transformed is None:
        _, pred_t, _ = linear_predictor(start, end, tau, dataset)
    else:
        pred_t = (
            (1.0 - tau) * start_transformed + tau * end_transformed
        ).astype(np.float32, copy=False)
    quantized = read_quantized(archive, dataset, frame_index, bits)
    scale = np_load_member(archive, archive_member_name(dataset, frame_index, "scale"))
    indices_name = archive_member_name(dataset, frame_index, "exception_indices")
    values_name = archive_member_name(dataset, frame_index, "exception_values")
    if indices_name in archive.namelist():
        exception_indices = np_load_member(archive, indices_name)
        exception_values = np_load_member(archive, values_name)
    else:
        exception_indices = np.empty(0, dtype=np.uint32)
        exception_values = np.empty(0, dtype=np.float32)
    payload = QuantizedResidual(
            values=quantized,
            scale=scale,
            exception_indices=exception_indices,
            exception_values=exception_values,
    )
    tiles = metadata.get("dataset_tile_shapes", {}).get(dataset)
    residual = dequantize_tiled(payload, tiles) if tiles else dequantize_residual(payload)
    return inverse_transformed(pred_t + residual, dataset)


def interpolated_attr(metadata: dict[str, Any], group: str, attr: str, tau: float) -> float | int | None:
    h5py = require_h5py()
    start_path = Path(metadata["start_file"])
    end_path = Path(metadata["end_file"])
    with h5py.File(start_path, "r") as start_handle, h5py.File(end_path, "r") as end_handle:
        if group not in start_handle or group not in end_handle:
            return None
        start_group = start_handle[group]
        end_group = end_handle[group]
        if attr not in start_group.attrs or attr not in end_group.attrs:
            return None
        start_value = start_group.attrs[attr]
        end_value = end_group.attrs[attr]
    return interpolate_attr_values(start_value, end_value, tau)


def interpolate_attr_values(
    start_value: Any,
    end_value: Any,
    tau: float,
) -> float | int | None:
    if np.issubdtype(np.asarray(start_value).dtype, np.integer):
        return int(round((1.0 - tau) * int(start_value) + tau * int(end_value)))
    if np.issubdtype(np.asarray(start_value).dtype, np.floating):
        return float((1.0 - tau) * float(start_value) + tau * float(end_value))
    return None


def frame_tau(archive: Any, metadata: dict[str, Any], datasets: tuple[str, ...], frame_index: int) -> float:
    middle_taus = metadata.get("middle_taus")
    if middle_taus is not None:
        if not 1 <= frame_index <= len(middle_taus):
            raise IndexError(f"frame index {frame_index} is outside [1, {len(middle_taus)}]")
        return float(middle_taus[frame_index - 1])
    for dataset in datasets:
        try:
            return float(np_load_member(archive, archive_member_name(dataset, frame_index, "tau")))
        except KeyError:
            continue
    raise KeyError(f"No tau member found for frame {frame_index}")


def default_template(metadata: dict[str, Any], frame_index: int) -> Path:
    return Path(metadata["start_file"])


def read_kharma_native_metadata(handle: Any) -> dict[str, Any]:
    required = ("Info", "Input", "Blocks/loc.lx123")
    missing = [name for name in required if name not in handle]
    if missing:
        raise KeyError("KHARMA native DDC input requires " + ", ".join(missing))
    info = handle["Info"]
    input_group = handle["Input"]
    if "NumMeshBlocks" not in info.attrs or "MeshBlockSize" not in info.attrs:
        raise KeyError("KHARMA Info requires NumMeshBlocks and MeshBlockSize")
    if "File" not in input_group.attrs:
        raise KeyError("KHARMA Input requires File")
    par_text = input_group.attrs["File"]
    if isinstance(par_text, bytes):
        par_text = par_text.decode("utf-8")
    meshblock_size = np.asarray(info.attrs["MeshBlockSize"], dtype=np.int64).reshape(-1)
    if meshblock_size.size < 3:
        raise ValueError("KHARMA MeshBlockSize requires at least three entries")
    block_order = np.ascontiguousarray(handle["Blocks/loc.lx123"][...], dtype="<i8")
    num_meshblocks = int(info.attrs["NumMeshBlocks"])
    if block_order.size != num_meshblocks * 3:
        raise ValueError("KHARMA block-order size does not match NumMeshBlocks")
    return {
        "par_text": str(par_text),
        "num_meshblocks": num_meshblocks,
        "meshblock_size": tuple(int(value) for value in meshblock_size[:3]),
        "block_order": block_order,
    }


def read_kharma_native_frame(
    path: Path,
    datasets: tuple[str, ...],
    *,
    sequence: int,
    time_value: float | None = None,
) -> dict[str, Any]:
    h5py = require_h5py()
    with h5py.File(path, "r") as handle:
        native = read_kharma_native_metadata(handle)
        arrays = {}
        for dataset in datasets:
            if dataset not in handle:
                raise KeyError(f"{dataset!r} not found in {path}")
            arrays[dataset] = np.ascontiguousarray(handle[dataset][...], dtype="<f4")
        if time_value is None:
            info = handle["Info"]
            if "Time" in info.attrs:
                time_value = float(info.attrs["Time"])
            elif "time" in info.attrs:
                time_value = float(info.attrs["time"])
            else:
                raise KeyError(f"KHARMA Info time is missing in {path}")
    return {
        **native,
        "sequence": int(sequence),
        "time": float(time_value),
        "datasets": arrays,
        "source_phdf": str(path),
        "exact_keyframe": True,
    }


def decode_archive_frames_to_arrays(
    archive_path: Path,
    frame_indices: tuple[int, ...],
    *,
    datasets_value: str | None = None,
    start_phdf: Path | None = None,
    end_phdf: Path | None = None,
    cache_chunks: int = 128,
) -> dict[int, dict[str, Any]]:
    """Decode GOP frames directly into contiguous KHARMA primitive arrays."""
    if not frame_indices:
        return {}
    results: dict[int, dict[str, Any]] = {}
    with open_archive(archive_path, cache_chunks=cache_chunks) as archive:
        metadata = read_metadata(archive)
        if start_phdf is not None:
            metadata["start_file"] = str(start_phdf)
        if end_phdf is not None:
            metadata["end_file"] = str(end_phdf)
        datasets = parse_datasets(datasets_value) if datasets_value else tuple(metadata["datasets"])
        h5py = require_h5py()
        with (
            h5py.File(Path(metadata["start_file"]), "r") as start_handle,
            h5py.File(Path(metadata["end_file"]), "r") as end_handle,
        ):
            native = read_kharma_native_metadata(start_handle)
            anchor_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for dataset in datasets:
                if dataset not in start_handle or dataset not in end_handle:
                    raise KeyError(f"{dataset!r} is missing from a GOP native-read anchor")
                anchor_arrays[dataset] = (
                    np.asarray(start_handle[dataset][...], dtype=np.float32),
                    np.asarray(end_handle[dataset][...], dtype=np.float32),
                )
            anchor_transformed = {
                dataset: (
                    transformed(anchors[0], dataset),
                    transformed(anchors[1], dataset),
                )
                for dataset, anchors in anchor_arrays.items()
            }
            for frame_index in frame_indices:
                tau = frame_tau(archive, metadata, datasets, int(frame_index))
                arrays = {
                    dataset: np.ascontiguousarray(
                        decode_dataset(
                            archive,
                            metadata,
                            dataset,
                            int(frame_index),
                            start=anchor_arrays[dataset][0],
                            end=anchor_arrays[dataset][1],
                            start_transformed=anchor_transformed[dataset][0],
                            end_transformed=anchor_transformed[dataset][1],
                        ),
                        dtype="<f4",
                    )
                    for dataset in datasets
                }
                start_time = end_time = None
                for attribute in ("Time", "time"):
                    if attribute in start_handle["Info"].attrs:
                        start_time = float(start_handle["Info"].attrs[attribute])
                    if attribute in end_handle["Info"].attrs:
                        end_time = float(end_handle["Info"].attrs[attribute])
                if start_time is None or end_time is None:
                    raise KeyError("KHARMA GOP anchors require Info Time/time")
                results[int(frame_index)] = {
                    **native,
                    "time": float((1.0 - tau) * start_time + tau * end_time),
                    "tau": float(tau),
                    "datasets": arrays,
                    "archive": str(archive_path),
                    "frame_index": int(frame_index),
                    "exact_keyframe": False,
                }
    return results


def decode_archive_frames_to_phdf(
    archive_path: Path,
    frame_outputs: dict[int, Path],
    *,
    template_phdf: Path | None = None,
    datasets_value: str | None = None,
    start_phdf: Path | None = None,
    end_phdf: Path | None = None,
    overwrite: bool = False,
) -> dict[int, dict[str, Any]]:
    if not frame_outputs:
        return {}
    outputs = {int(frame_index): Path(output) for frame_index, output in frame_outputs.items()}
    for output_phdf in outputs.values():
        if output_phdf.exists():
            if not overwrite:
                raise FileExistsError(f"{output_phdf} already exists; pass --overwrite")
            output_phdf.unlink()
        output_phdf.parent.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, Any]] = {}
    with open_archive(archive_path) as archive:
        metadata = read_metadata(archive)
        if start_phdf is not None:
            metadata["start_file"] = str(start_phdf)
        if end_phdf is not None:
            metadata["end_file"] = str(end_phdf)
        datasets = parse_datasets(datasets_value) if datasets_value else tuple(metadata["datasets"])
        first_frame_index = next(iter(outputs))
        template = template_phdf or default_template(metadata, first_frame_index)
        h5py = require_h5py()
        with (
            h5py.File(Path(metadata["start_file"]), "r") as start_handle,
            h5py.File(Path(metadata["end_file"]), "r") as end_handle,
        ):
            anchor_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for dataset in datasets:
                if dataset not in start_handle:
                    raise KeyError(f"{dataset!r} not found in GOP start {metadata['start_file']}")
                if dataset not in end_handle:
                    raise KeyError(f"{dataset!r} not found in GOP end {metadata['end_file']}")
                anchor_arrays[dataset] = (
                    start_handle[dataset][...],
                    end_handle[dataset][...],
                )

            for frame_index, output_phdf in outputs.items():
                tau = frame_tau(archive, metadata, datasets, frame_index)
                shutil.copy2(template, output_phdf)
                with h5py.File(output_phdf, "r+") as handle:
                    handle.attrs["dump_codec"] = str(
                        metadata.get("codec", "linear_residual_quantized_v1")
                    )
                    handle.attrs["dump_codec_format_version"] = int(
                        metadata.get("format_version", 1)
                    )
                    handle.attrs["dump_codec_archive"] = str(archive_path)
                    handle.attrs["dump_codec_frame_index"] = int(frame_index)
                    handle.attrs["dump_codec_bits"] = int(metadata["bits"])
                    handle.attrs["dump_codec_dataset_bits"] = json.dumps(
                        metadata.get("dataset_bits", {}),
                        sort_keys=True,
                    )
                    handle.attrs["dump_codec_tau"] = float(tau)
                    for group, attrs in {
                        "Info": ("Time", "NCycle"),
                        "Params": ("Globals/time",),
                    }.items():
                        if group in handle:
                            for attr in attrs:
                                if (
                                    group not in start_handle
                                    or group not in end_handle
                                    or attr not in start_handle[group].attrs
                                    or attr not in end_handle[group].attrs
                                ):
                                    continue
                                value = interpolate_attr_values(
                                    start_handle[group].attrs[attr],
                                    end_handle[group].attrs[attr],
                                    tau,
                                )
                                if value is not None:
                                    handle[group].attrs[attr] = value
                    for dataset in datasets:
                        if dataset not in handle:
                            raise KeyError(f"{dataset!r} not found in template {template}")
                        start, end = anchor_arrays[dataset]
                        reconstructed = decode_dataset(
                            archive,
                            metadata,
                            dataset,
                            frame_index,
                            start=start,
                            end=end,
                        )
                        if handle[dataset].shape != reconstructed.shape:
                            raise ValueError(
                                f"Shape mismatch for {dataset}: template "
                                f"{handle[dataset].shape}, decoded {reconstructed.shape}"
                            )
                        handle[dataset][...] = reconstructed.astype(
                            handle[dataset].dtype,
                            copy=False,
                        )
                results[frame_index] = {
                    "output_phdf": str(output_phdf),
                    "template_phdf": str(template),
                    "datasets": datasets,
                    "archive": str(archive_path),
                    "frame_index": frame_index,
                }
    return results


def decode_archive_to_phdf(
    archive_path: Path,
    frame_index: int,
    output_phdf: Path,
    *,
    template_phdf: Path | None = None,
    datasets_value: str | None = None,
    start_phdf: Path | None = None,
    end_phdf: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    return decode_archive_frames_to_phdf(
        archive_path,
        {frame_index: output_phdf},
        template_phdf=template_phdf,
        datasets_value=datasets_value,
        start_phdf=start_phdf,
        end_phdf=end_phdf,
        overwrite=overwrite,
    )[frame_index]


def main() -> None:
    args = parse_args()
    result = decode_archive_to_phdf(
        args.archive,
        args.frame_index,
        args.output_phdf,
        template_phdf=args.template_phdf,
        datasets_value=args.datasets,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
