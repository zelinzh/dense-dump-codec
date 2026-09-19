"""Lossless independent and temporally differenced PHDF keyframe archives."""

from __future__ import annotations

import bz2
import concurrent.futures
import contextlib
import json
import lzma
import shutil
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


KEYFRAME_BZIP2_SHUFFLE_FORMAT = "dense_dump_codec_keyframe_bzip2_shuffle_v1"
KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT = (
    "dense_dump_codec_keyframe_temporal_bzip2_shuffle_v1"
)
KEYFRAME_XZ_SHUFFLE_FORMAT = "dense_dump_codec_keyframe_xz_shuffle_v1"
KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT = (
    "dense_dump_codec_keyframe_temporal_xz_shuffle_v1"
)
KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT = (
    "dense_dump_codec_keyframe_temporal_xz_zigzag_shuffle_v1"
)
KEYFRAME_METADATA_MEMBER = "keyframe.json"
KEYFRAME_TEMPLATE_MEMBER = "template.phdf"
KEYFRAME_SIDECAR_GROUP = "__dense_dump_codec_keyframe__"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def _copy_attributes(source: h5py.AttributeManager, output: h5py.AttributeManager) -> None:
    for key, value in source.items():
        output[key] = value


def _byte_shuffle(payload: bytes, itemsize: int, *, inverse: bool = False) -> bytes:
    if itemsize <= 1:
        return payload
    values = np.frombuffer(payload, dtype=np.uint8)
    if values.size % itemsize:
        raise ValueError("Payload size is not divisible by the dtype itemsize")
    if inverse:
        return values.reshape(itemsize, -1).T.copy().reshape(-1).tobytes()
    return values.reshape(-1, itemsize).T.copy().reshape(-1).tobytes()


def _slice_row(slices: tuple[slice, ...]) -> list[list[int]]:
    return [[int(item.start), int(item.stop)] for item in slices]


def _row_slices(row: list[list[int]]) -> tuple[slice, ...]:
    return tuple(slice(int(start), int(stop)) for start, stop in row)


def _chunk_slices(dataset: h5py.Dataset) -> Iterable[tuple[slice, ...]]:
    if dataset.chunks is None:
        yield tuple(slice(0, size) for size in dataset.shape)
        return
    yield from dataset.iter_chunks()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _compression_options(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _dataset_metadata(name: str, dataset: h5py.Dataset, sidecar: h5py.Group) -> dict[str, Any]:
    attribute_group = sidecar.create_group(f"{len(sidecar):05d}")
    _copy_attributes(dataset.attrs, attribute_group.attrs)
    return {
        "name": name,
        "attribute_group": attribute_group.name.removeprefix(f"/{KEYFRAME_SIDECAR_GROUP}/"),
        "dtype": dataset.dtype.str,
        "shape": list(dataset.shape),
        "maxshape": list(dataset.maxshape),
        "chunks": list(dataset.chunks) if dataset.chunks is not None else None,
        "compression": dataset.compression,
        "compression_opts": _json_value(dataset.compression_opts),
        "shuffle": bool(dataset.shuffle),
        "fletcher32": bool(dataset.fletcher32),
        "scaleoffset": _json_value(dataset.scaleoffset),
        "fillvalue": _json_value(dataset.fillvalue),
        "chunks_data": [],
    }


def _create_template(
    source_path: Path,
    template_path: Path,
    datasets: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected = set(datasets)
    rows: dict[str, dict[str, Any]] = {}
    with h5py.File(source_path, "r") as source, h5py.File(template_path, "w") as output:
        if KEYFRAME_SIDECAR_GROUP in source:
            raise ValueError(f"Source already contains reserved group {KEYFRAME_SIDECAR_GROUP!r}")
        _copy_attributes(source.attrs, output.attrs)
        sidecar = output.create_group(KEYFRAME_SIDECAR_GROUP)

        def copy_object(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                group = output.require_group(name)
                _copy_attributes(obj.attrs, group.attrs)
                return
            if name in selected:
                rows[name] = _dataset_metadata(name, obj, sidecar)
                return
            parent_name, _, leaf = name.rpartition("/")
            parent = output.require_group(parent_name) if parent_name else output
            source.copy(name, parent, name=leaf)

        source.visititems(copy_object)
    missing = sorted(selected - rows.keys())
    if missing:
        raise KeyError(f"Datasets are missing from {source_path}: {missing}")
    return [rows[name] for name in datasets]


@dataclass(frozen=True)
class _ChunkTask:
    member: str
    slices: tuple[slice, ...]
    dtype: np.dtype[Any]
    payload: bytes
    target_crc32: int


def _compress_payload(payload: bytes, compression: str, compression_level: int) -> bytes:
    if compression == "bzip2":
        return bz2.compress(payload, compresslevel=compression_level)
    if compression == "xz":
        return lzma.compress(
            payload,
            format=lzma.FORMAT_XZ,
            preset=compression_level | lzma.PRESET_EXTREME,
        )
    raise ValueError(f"Unsupported keyframe compression {compression!r}")


def _decompress_payload(payload: bytes, compression: str) -> bytes:
    if compression == "bzip2":
        return bz2.decompress(payload)
    if compression == "xz":
        return lzma.decompress(payload, format=lzma.FORMAT_XZ)
    raise ValueError(f"Unsupported keyframe compression {compression!r}")


def _compress_chunk(
    task: _ChunkTask,
    compression: str,
    compression_level: int,
    temporal_zigzag: bool,
) -> tuple[_ChunkTask, bytes]:
    payload = (
        _temporal_zigzag_encode(task.payload, task.dtype)
        if temporal_zigzag
        else task.payload
    )
    shuffled = _byte_shuffle(payload, task.dtype.itemsize)
    return task, _compress_payload(shuffled, compression, compression_level)


def _chunk_tasks(
    source_path: Path,
    rows: list[dict[str, Any]],
    reference_paths: tuple[Path, ...],
    temporal_order: int,
    member_extension: str,
) -> Iterable[_ChunkTask]:
    with contextlib.ExitStack() as stack:
        source = stack.enter_context(h5py.File(source_path, "r"))
        references = [
            stack.enter_context(h5py.File(reference_path, "r"))
            for reference_path in reference_paths
        ]
        for dataset_index, row in enumerate(rows):
            dataset = source[row["name"]]
            reference_datasets = [reference[row["name"]] for reference in references]
            for reference_dataset in reference_datasets:
                if reference_dataset.shape != dataset.shape:
                    raise ValueError(f"Shape mismatch for reference dataset {row['name']}")
                if reference_dataset.dtype != dataset.dtype:
                    raise ValueError(f"Dtype mismatch for reference dataset {row['name']}")
            for chunk_index, slices in enumerate(_chunk_slices(dataset)):
                member = (
                    f"datasets/{dataset_index:05d}/{chunk_index:05d}.{member_extension}"
                )
                target = np.asarray(dataset[slices])
                target_payload = target.tobytes(order="C")
                payload = _temporal_encode(
                    target,
                    [np.asarray(reference[slices]) for reference in reference_datasets],
                    temporal_order,
                )
                yield _ChunkTask(
                    member,
                    slices,
                    dataset.dtype,
                    payload,
                    zlib.crc32(target_payload) & 0xFFFFFFFF,
                )


def _unsigned_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    if dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(f"Temporal transform does not support dtype {dtype}")
    return np.dtype(f"{dtype.byteorder}u{dtype.itemsize}")


def _temporal_encode(
    target: np.ndarray[Any, Any],
    references: list[np.ndarray[Any, Any]],
    temporal_order: int,
) -> bytes:
    if temporal_order == 0:
        return target.tobytes(order="C")
    unsigned = _unsigned_dtype(target.dtype)
    encoded = np.ascontiguousarray(target).view(unsigned).reshape(-1).copy()
    previous = np.ascontiguousarray(references[0]).view(unsigned).reshape(-1)
    np.subtract(encoded, previous, out=encoded)
    if temporal_order == 2:
        previous_delta = previous.copy()
        previous_previous = (
            np.ascontiguousarray(references[1]).view(unsigned).reshape(-1)
        )
        np.subtract(previous_delta, previous_previous, out=previous_delta)
        np.subtract(encoded, previous_delta, out=encoded)
    return encoded.tobytes(order="C")


def _temporal_decode(
    encoded_payload: bytes,
    dtype: np.dtype[Any],
    references: list[np.ndarray[Any, Any]],
    temporal_order: int,
) -> bytes:
    if temporal_order == 0:
        return encoded_payload
    unsigned = _unsigned_dtype(dtype)
    decoded = np.frombuffer(encoded_payload, dtype=unsigned).copy()
    previous = np.ascontiguousarray(references[0]).view(unsigned).reshape(-1)
    if temporal_order == 2:
        previous_delta = previous.copy()
        previous_previous = (
            np.ascontiguousarray(references[1]).view(unsigned).reshape(-1)
        )
        np.subtract(previous_delta, previous_previous, out=previous_delta)
        np.add(decoded, previous_delta, out=decoded)
    np.add(decoded, previous, out=decoded)
    return decoded.tobytes(order="C")


def _temporal_zigzag_encode(payload: bytes, dtype: np.dtype[Any]) -> bytes:
    unsigned = _unsigned_dtype(dtype)
    values = np.frombuffer(payload, dtype=unsigned).copy()
    sign = values >> (dtype.itemsize * 8 - 1)
    sign_mask = np.zeros_like(sign)
    np.subtract(sign_mask, sign, out=sign_mask)
    encoded = values << 1
    np.bitwise_xor(encoded, sign_mask, out=encoded)
    return encoded.tobytes(order="C")


def _temporal_zigzag_decode(payload: bytes, dtype: np.dtype[Any]) -> bytes:
    unsigned = _unsigned_dtype(dtype)
    values = np.frombuffer(payload, dtype=unsigned).copy()
    low_bit = values & 1
    sign_mask = np.zeros_like(low_bit)
    np.subtract(sign_mask, low_bit, out=sign_mask)
    decoded = values >> 1
    np.bitwise_xor(decoded, sign_mask, out=decoded)
    return decoded.tobytes(order="C")


def _resolved_references(
    reference_paths: Iterable[Path | str],
    temporal_order: int,
) -> tuple[Path, ...]:
    if temporal_order not in (0, 1, 2):
        raise ValueError("temporal_order must be 0, 1, or 2")
    references = tuple(Path(path).resolve() for path in reference_paths)
    if len(references) != temporal_order:
        raise ValueError(
            f"temporal_order={temporal_order} requires {temporal_order} reference paths"
        )
    return references


def compress_keyframe(
    source_path: Path | str,
    output_path: Path | str,
    datasets: Iterable[str],
    *,
    compression_level: int = 9,
    workers: int = 1,
    overwrite: bool = False,
    reference_paths: Iterable[Path | str] = (),
    temporal_order: int = 0,
    compression: str = "bzip2",
    temporal_zigzag: bool = False,
) -> dict[str, Any]:
    """Compress selected PHDF datasets exactly into a byte-shuffled archive."""

    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    references = _resolved_references(reference_paths, temporal_order)
    dataset_names = tuple(dict.fromkeys(datasets))
    if source_path == output_path:
        raise ValueError("Source and output paths must differ")
    if not dataset_names:
        raise ValueError("At least one dataset is required")
    if not 1 <= compression_level <= 9:
        raise ValueError("compression_level must be in [1, 9]")
    if compression not in ("bzip2", "xz"):
        raise ValueError("compression must be 'bzip2' or 'xz'")
    if temporal_zigzag and (temporal_order == 0 or compression != "xz"):
        raise ValueError("temporal_zigzag requires temporal_order > 0 and xz compression")
    if workers < 1:
        raise ValueError("workers must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_dir:
            template_path = Path(temporary_dir) / KEYFRAME_TEMPLATE_MEMBER
            rows = _create_template(source_path, template_path, dataset_names)
            formats = {
                ("bzip2", False): KEYFRAME_BZIP2_SHUFFLE_FORMAT,
                ("bzip2", True): KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT,
                ("xz", False): KEYFRAME_XZ_SHUFFLE_FORMAT,
                ("xz", True): KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT,
            }
            archive_format = (
                KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT
                if temporal_zigzag
                else formats[(compression, bool(temporal_order))]
            )
            metadata = {
                "format": archive_format,
                "source_name": source_path.name,
                "source_mode": source_path.stat().st_mode & 0o777,
                "compression": compression,
                "compression_level": compression_level,
                "preconditioner": "hdf5-chunk-byte-shuffle-v1",
                "temporal_preconditioner": (
                    f"unsigned-modular-difference-order-{temporal_order}-v1"
                    if temporal_order
                    else "none"
                ),
                "temporal_order": temporal_order,
                "temporal_zigzag": temporal_zigzag,
                "reference_names": [path.name for path in references],
                "datasets": rows,
            }
            with zipfile.ZipFile(temporary_output, "w", allowZip64=True) as archive:
                archive.writestr(_zip_info(KEYFRAME_TEMPLATE_MEMBER), template_path.read_bytes())
                tasks = _chunk_tasks(
                    source_path,
                    rows,
                    references,
                    temporal_order,
                    "bz2" if compression == "bzip2" else "xz",
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    results = executor.map(
                        lambda task: _compress_chunk(
                            task, compression, compression_level, temporal_zigzag
                        ),
                        tasks,
                    )
                    row_by_prefix = {
                        f"datasets/{index:05d}/": row for index, row in enumerate(rows)
                    }
                    for task, compressed in results:
                        archive.writestr(_zip_info(task.member), compressed)
                        row = next(
                            value
                            for prefix, value in row_by_prefix.items()
                            if task.member.startswith(prefix)
                        )
                        row["chunks_data"].append(
                            {
                                "member": task.member,
                                "slices": _slice_row(task.slices),
                                "raw_bytes": len(task.payload),
                                "compressed_bytes": len(compressed),
                                "crc32": task.target_crc32,
                                "encoded_crc32": zlib.crc32(task.payload) & 0xFFFFFFFF,
                            }
                        )
                archive.writestr(
                    _zip_info(KEYFRAME_METADATA_MEMBER),
                    json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n",
                )
        temporary_output.replace(output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    source_bytes = source_path.stat().st_size
    output_bytes = output_path.stat().st_size
    return {
        "format": archive_format,
        "source": str(source_path),
        "output": str(output_path),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "saving_vs_source_fraction": 1.0 - output_bytes / source_bytes,
        "dataset_count": len(rows),
        "chunk_count": sum(len(row["chunks_data"]) for row in rows),
        "compression_level": compression_level,
        "compression": compression,
        "temporal_order": temporal_order,
        "temporal_zigzag": temporal_zigzag,
        "reference_paths": [str(path) for path in references],
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _creation_kwargs(row: dict[str, Any], attribute_group: h5py.Group) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"maxshape": tuple(row["maxshape"])}
    if row["chunks"] is not None:
        kwargs["chunks"] = tuple(row["chunks"])
    if row["compression"] is not None:
        kwargs["compression"] = row["compression"]
        kwargs["compression_opts"] = _compression_options(row["compression_opts"])
    if row["shuffle"]:
        kwargs["shuffle"] = True
    if row["fletcher32"]:
        kwargs["fletcher32"] = True
    if row["scaleoffset"] is not None:
        kwargs["scaleoffset"] = row["scaleoffset"]
    if row.get("fillvalue") is not None:
        kwargs["fillvalue"] = row["fillvalue"]
    elif "fillvalue" in attribute_group.attrs:
        kwargs["fillvalue"] = attribute_group.attrs["fillvalue"]
    return kwargs


def decompress_keyframe(
    archive_path: Path | str,
    output_path: Path | str,
    *,
    overwrite: bool = False,
    reference_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    """Reconstruct a PHDF keyframe and validate every decoded chunk."""

    archive_path = Path(archive_path).resolve()
    output_path = Path(output_path).resolve()
    if archive_path == output_path:
        raise ValueError("Archive and output paths must differ")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Archive CRC failure at {bad_member}")
            metadata = json.loads(archive.read(KEYFRAME_METADATA_MEMBER))
            archive_format = metadata.get("format")
            if archive_format not in (
                KEYFRAME_BZIP2_SHUFFLE_FORMAT,
                KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT,
                KEYFRAME_XZ_SHUFFLE_FORMAT,
                KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT,
                KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT,
            ):
                raise ValueError(f"Unsupported keyframe format {metadata.get('format')!r}")
            temporal_order = int(metadata.get("temporal_order", 0))
            temporal_zigzag = bool(metadata.get("temporal_zigzag", False))
            compression = str(metadata.get("compression", "bzip2"))
            references = _resolved_references(reference_paths, temporal_order)
            with archive.open(KEYFRAME_TEMPLATE_MEMBER) as source, temporary_output.open("wb") as output:
                shutil.copyfileobj(source, output)
            with contextlib.ExitStack() as stack:
                output = stack.enter_context(h5py.File(temporary_output, "r+"))
                reference_files = [
                    stack.enter_context(h5py.File(path, "r")) for path in references
                ]
                sidecar = output[KEYFRAME_SIDECAR_GROUP]
                for row in metadata["datasets"]:
                    attribute_group = sidecar[row["attribute_group"]]
                    parent_name, _, leaf = row["name"].rpartition("/")
                    parent = output.require_group(parent_name) if parent_name else output
                    dataset = parent.create_dataset(
                        leaf,
                        shape=tuple(row["shape"]),
                        dtype=np.dtype(row["dtype"]),
                        **_creation_kwargs(row, attribute_group),
                    )
                    for key, value in attribute_group.attrs.items():
                        if key != "fillvalue" or "fillvalue" in row:
                            dataset.attrs[key] = value
                    for chunk in row["chunks_data"]:
                        compressed = archive.read(chunk["member"])
                        shuffled = _decompress_payload(compressed, compression)
                        encoded_payload = _byte_shuffle(
                            shuffled, dataset.dtype.itemsize, inverse=True
                        )
                        if temporal_zigzag:
                            encoded_payload = _temporal_zigzag_decode(
                                encoded_payload, dataset.dtype
                            )
                        if len(encoded_payload) != int(chunk["raw_bytes"]):
                            raise ValueError(f"Size mismatch for {chunk['member']}")
                        encoded_crc32 = int(chunk.get("encoded_crc32", chunk["crc32"]))
                        if zlib.crc32(encoded_payload) & 0xFFFFFFFF != encoded_crc32:
                            raise ValueError(f"Encoded CRC failure for {chunk['member']}")
                        slices = _row_slices(chunk["slices"])
                        shape = tuple(item.stop - item.start for item in slices)
                        payload = _temporal_decode(
                            encoded_payload,
                            dataset.dtype,
                            [reference[row["name"]][slices] for reference in reference_files],
                            temporal_order,
                        )
                        if zlib.crc32(payload) & 0xFFFFFFFF != int(chunk["crc32"]):
                            raise ValueError(f"CRC failure for {chunk['member']}")
                        dataset[slices] = np.frombuffer(payload, dtype=dataset.dtype).reshape(shape)
                del output[KEYFRAME_SIDECAR_GROUP]
        temporary_output.replace(output_path)
        try:
            output_path.chmod(int(metadata.get("source_mode", 0o644)))
        except PermissionError:
            pass
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
    return {
        "format": archive_format,
        "archive": str(archive_path),
        "output": str(output_path),
        "archive_bytes": archive_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "dataset_count": len(metadata["datasets"]),
        "chunk_count": sum(len(row["chunks_data"]) for row in metadata["datasets"]),
        "temporal_order": temporal_order,
        "temporal_zigzag": temporal_zigzag,
        "compression": compression,
        "elapsed_seconds": time.perf_counter() - started,
    }


def validate_keyframe_archive(
    archive_path: Path | str,
    *,
    reference_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    """Validate the container, compressed streams, decoded sizes, and raw CRCs."""

    archive_path = Path(archive_path).resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Archive CRC failure at {bad_member}")
        metadata = json.loads(archive.read(KEYFRAME_METADATA_MEMBER))
        archive_format = metadata.get("format")
        if archive_format not in (
            KEYFRAME_BZIP2_SHUFFLE_FORMAT,
            KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT,
            KEYFRAME_XZ_SHUFFLE_FORMAT,
            KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT,
            KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT,
        ):
            raise ValueError(f"Unsupported keyframe format {metadata.get('format')!r}")
        temporal_order = int(metadata.get("temporal_order", 0))
        temporal_zigzag = bool(metadata.get("temporal_zigzag", False))
        compression = str(metadata.get("compression", "bzip2"))
        references = tuple(Path(path).resolve() for path in reference_paths)
        if references and len(references) != temporal_order:
            raise ValueError(
                f"temporal_order={temporal_order} requires {temporal_order} reference paths"
            )
        if temporal_order == 0 and references:
            raise ValueError("Independent archives do not accept reference paths")
        chunk_count = 0
        with contextlib.ExitStack() as stack:
            reference_files = [
                stack.enter_context(h5py.File(path, "r")) for path in references
            ]
            for row in metadata["datasets"]:
                dtype = np.dtype(row["dtype"])
                for chunk in row["chunks_data"]:
                    shuffled = _decompress_payload(
                        archive.read(chunk["member"]), compression
                    )
                    encoded_payload = _byte_shuffle(shuffled, dtype.itemsize, inverse=True)
                    if temporal_zigzag:
                        encoded_payload = _temporal_zigzag_decode(encoded_payload, dtype)
                    if len(encoded_payload) != int(chunk["raw_bytes"]):
                        raise ValueError(f"Size mismatch for {chunk['member']}")
                    encoded_crc32 = int(chunk.get("encoded_crc32", chunk["crc32"]))
                    if zlib.crc32(encoded_payload) & 0xFFFFFFFF != encoded_crc32:
                        raise ValueError(f"Encoded CRC failure for {chunk['member']}")
                    if temporal_order == 0 or reference_files:
                        slices = _row_slices(chunk["slices"])
                        payload = _temporal_decode(
                            encoded_payload,
                            dtype,
                            [reference[row["name"]][slices] for reference in reference_files],
                            temporal_order,
                        )
                        if zlib.crc32(payload) & 0xFFFFFFFF != int(chunk["crc32"]):
                            raise ValueError(f"CRC failure for {chunk['member']}")
                    chunk_count += 1
    return {
        "format": archive_format,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "dataset_count": len(metadata["datasets"]),
        "chunk_count": chunk_count,
        "temporal_order": temporal_order,
        "temporal_zigzag": temporal_zigzag,
        "target_crc_verified": temporal_order == 0 or bool(references),
        "compression": compression,
    }
