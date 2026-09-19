"""Random-access containers for dense-dump codec residual members."""

from __future__ import annotations

import bz2
import concurrent.futures
import json
import shutil
import tarfile
import tempfile
import time
import zipfile
import zlib
from collections import OrderedDict, defaultdict, deque
from contextlib import AbstractContextManager
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


CHANNEL_BZIP2_FORMAT = "dense_dump_codec_channel_bzip2_v1"
CHANNEL_BZIP2_DELTA_FORMAT = "dense_dump_codec_channel_bzip2_delta_v1"
CHANNEL_BZIP2_DELTA_SHUFFLE_FORMAT = (
    "dense_dump_codec_channel_bzip2_delta_shuffle_v1"
)
CHANNEL_BZIP2_ADAPTIVE_FORMAT = "dense_dump_codec_channel_bzip2_adaptive_v1"
TEMPORAL_DELTA_PRECONDITIONER = "temporal-npy-delta-xor-v1"
TEMPORAL_DELTA_SHUFFLE_PRECONDITIONER = (
    "temporal-npy-delta-zigzag-byte-shuffle-v1"
)
TEMPORAL_DELTA2_QUANT_SHUFFLE_PRECONDITIONER = (
    "temporal-npy-delta2-quant-zigzag-byte-shuffle-v1"
)
ADAPTIVE_TEMPORAL_PRECONDITIONER = (
    "adaptive-min-temporal-npy-delta1-delta2-quant-zigzag-byte-shuffle-v1"
)
CHANNEL_BZIP2_FORMATS = {
    CHANNEL_BZIP2_FORMAT,
    CHANNEL_BZIP2_DELTA_FORMAT,
    CHANNEL_BZIP2_DELTA_SHUFFLE_FORMAT,
    CHANNEL_BZIP2_ADAPTIVE_FORMAT,
}
CONTAINER_MEMBER = "container.json"
CHANNEL_DEFLATE_FORMAT = "dense_dump_codec_channel_deflate_v1"
CHANNEL_FORMATS = CHANNEL_BZIP2_FORMATS | {CHANNEL_DEFLATE_FORMAT}
INDEX_MEMBER = "index.json"
RESERVED_MEMBERS = {CONTAINER_MEMBER, INDEX_MEMBER}


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def _member_group(name: str, chunk_frames: int) -> str | None:
    parts = Path(name).parts
    if len(parts) < 3 or parts[0] != "frames":
        return None
    frame_offset = max(int(parts[1]) - 1, 0)
    return f"{parts[-1]}|chunk={frame_offset // chunk_frames:05d}"


def _tar_info(info: zipfile.ZipInfo) -> tarfile.TarInfo:
    tar_info = tarfile.TarInfo(info.filename)
    tar_info.mtime = 0
    tar_info.mode = 0o755 if info.is_dir() else 0o644
    tar_info.uid = 0
    tar_info.gid = 0
    tar_info.uname = ""
    tar_info.gname = ""
    if info.is_dir():
        tar_info.type = tarfile.DIRTYPE
    else:
        tar_info.size = info.file_size
    return tar_info


def _npy_layout(payload: bytes) -> tuple[int, np.dtype[Any], tuple[int, ...]] | None:
    try:
        stream = BytesIO(payload)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            array = np.load(BytesIO(payload), allow_pickle=False)
            shape = array.shape
            dtype = array.dtype
            body_offset = len(payload) - array.nbytes
            return body_offset, dtype, shape
    except (EOFError, ValueError):
        return None
    body_offset = stream.tell()
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if dtype.hasobject or body_offset + expected_bytes != len(payload):
        return None
    return body_offset, dtype, shape


def _unsigned_dtype(dtype: np.dtype[Any]) -> np.dtype[Any] | None:
    if dtype.itemsize not in {1, 2, 4, 8}:
        return None
    byteorder = dtype.byteorder if dtype.byteorder in {"<", ">"} else "="
    return np.dtype(f"{byteorder}u{dtype.itemsize}")


def _temporal_npy_transform(
    payload: bytes,
    previous_payload: bytes | None,
    *,
    inverse: bool,
) -> bytes:
    if previous_payload is None:
        return payload
    layout = _npy_layout(payload)
    previous_layout = _npy_layout(previous_payload)
    if layout is None or previous_layout is None:
        return payload
    body_offset, dtype, shape = layout
    previous_offset, previous_dtype, previous_shape = previous_layout
    if (
        payload[:body_offset] != previous_payload[:previous_offset]
        or dtype != previous_dtype
        or shape != previous_shape
    ):
        return payload
    body = payload[body_offset:]
    previous_body = previous_payload[previous_offset:]
    if dtype.kind in {"i", "u"}:
        unsigned_dtype = _unsigned_dtype(dtype)
        if unsigned_dtype is None:
            return payload
        values = np.frombuffer(body, dtype=unsigned_dtype)
        previous_values = np.frombuffer(previous_body, dtype=unsigned_dtype)
        operation = np.add if inverse else np.subtract
        transformed = operation(values, previous_values, dtype=unsigned_dtype)
        transformed_body = transformed.tobytes()
    elif dtype.kind in {"b", "f", "c"}:
        transformed_body = np.bitwise_xor(
            np.frombuffer(body, dtype=np.uint8),
            np.frombuffer(previous_body, dtype=np.uint8),
        ).tobytes()
    else:
        return payload
    return payload[:body_offset] + transformed_body


def _byte_shuffle(body: bytes, itemsize: int, *, inverse: bool) -> bytes:
    if itemsize <= 1:
        return body
    values = np.frombuffer(body, dtype=np.uint8)
    if values.size % itemsize:
        raise ValueError("NPY body size is not divisible by dtype itemsize")
    if inverse:
        shuffled = values.reshape(itemsize, -1)
        return shuffled.T.copy().reshape(-1).tobytes()
    unshuffled = values.reshape(-1, itemsize)
    return unshuffled.T.copy().reshape(-1).tobytes()


def _zigzag_unsigned(
    values: np.ndarray,
    dtype: np.dtype[Any],
    *,
    inverse: bool,
) -> np.ndarray:
    if inverse:
        sign_mask = np.subtract(
            np.zeros((), dtype=dtype),
            np.bitwise_and(values, 1),
            dtype=dtype,
        )
        return np.bitwise_xor(np.right_shift(values, 1), sign_mask)
    sign = np.right_shift(values, dtype.itemsize * 8 - 1)
    sign_mask = np.subtract(np.zeros((), dtype=dtype), sign, dtype=dtype)
    return np.bitwise_xor(np.left_shift(values, 1, dtype=dtype), sign_mask)


def _temporal_npy_delta_shuffle_transform(
    payload: bytes,
    previous_payload: bytes | None,
    *,
    inverse: bool,
) -> bytes:
    layout = _npy_layout(payload)
    if layout is None:
        return payload
    body_offset, dtype, shape = layout
    body = _byte_shuffle(payload[body_offset:], dtype.itemsize, inverse=inverse)
    unshuffled = payload[:body_offset] + body
    previous_layout = (
        _npy_layout(previous_payload) if previous_payload is not None else None
    )
    temporal_match = (
        previous_payload is not None
        and previous_layout is not None
        and payload[:body_offset] == previous_payload[: previous_layout[0]]
        and dtype == previous_layout[1]
        and shape == previous_layout[2]
    )
    if inverse:
        if temporal_match and dtype.kind in {"i", "u"}:
            unsigned_dtype = _unsigned_dtype(dtype)
            if unsigned_dtype is not None:
                values = np.frombuffer(body, dtype=unsigned_dtype)
                body = _zigzag_unsigned(
                    values,
                    unsigned_dtype,
                    inverse=True,
                ).tobytes()
                unshuffled = payload[:body_offset] + body
        return _temporal_npy_transform(
            unshuffled,
            previous_payload,
            inverse=True,
        )

    transformed = _temporal_npy_transform(
        payload,
        previous_payload,
        inverse=False,
    )
    transformed_body = transformed[body_offset:]
    if temporal_match and dtype.kind in {"i", "u"}:
        unsigned_dtype = _unsigned_dtype(dtype)
        if unsigned_dtype is not None:
            values = np.frombuffer(transformed_body, dtype=unsigned_dtype)
            transformed_body = _zigzag_unsigned(
                values,
                unsigned_dtype,
                inverse=False,
            ).tobytes()
    transformed_body = _byte_shuffle(
        transformed_body,
        dtype.itemsize,
        inverse=False,
    )
    return transformed[:body_offset] + transformed_body


def _is_quantized_member(name: str | None) -> bool:
    if name is None:
        return False
    basename = Path(name).name
    return (
        basename.endswith(".npy")
        and "_q" in basename
        and not basename.endswith("_shape.npy")
    )


def _temporal_npy_delta2_quant_shuffle_transform(
    payload: bytes,
    previous_payload: bytes | None,
    previous_previous_payload: bytes | None,
    member_name: str | None,
    *,
    inverse: bool,
) -> bytes:
    if (
        previous_payload is None
        or previous_previous_payload is None
        or not _is_quantized_member(member_name)
    ):
        return _temporal_npy_delta_shuffle_transform(
            payload,
            previous_payload,
            inverse=inverse,
        )
    layout = _npy_layout(payload)
    previous_layout = _npy_layout(previous_payload)
    previous_previous_layout = _npy_layout(previous_previous_payload)
    if (
        layout is None
        or previous_layout is None
        or previous_previous_layout is None
    ):
        return _temporal_npy_delta_shuffle_transform(
            payload,
            previous_payload,
            inverse=inverse,
        )
    body_offset, dtype, shape = layout
    previous_offset, previous_dtype, previous_shape = previous_layout
    previous_previous_offset, previous_previous_dtype, previous_previous_shape = (
        previous_previous_layout
    )
    if (
        payload[:body_offset] != previous_payload[:previous_offset]
        or payload[:body_offset]
        != previous_previous_payload[:previous_previous_offset]
        or dtype != previous_dtype
        or dtype != previous_previous_dtype
        or shape != previous_shape
        or shape != previous_previous_shape
        or dtype.kind not in {"i", "u"}
    ):
        return _temporal_npy_delta_shuffle_transform(
            payload,
            previous_payload,
            inverse=inverse,
        )
    unsigned_dtype = _unsigned_dtype(dtype)
    if unsigned_dtype is None:
        return _temporal_npy_delta_shuffle_transform(
            payload,
            previous_payload,
            inverse=inverse,
        )
    body = payload[body_offset:]
    if inverse:
        body = _byte_shuffle(body, dtype.itemsize, inverse=True)
        second_difference = _zigzag_unsigned(
            np.frombuffer(body, dtype=unsigned_dtype),
            unsigned_dtype,
            inverse=True,
        )
        previous_values = np.frombuffer(
            previous_payload[previous_offset:],
            dtype=unsigned_dtype,
        )
        previous_previous_values = np.frombuffer(
            previous_previous_payload[previous_previous_offset:],
            dtype=unsigned_dtype,
        )
        values = np.subtract(
            np.add(
                second_difference,
                np.add(previous_values, previous_values, dtype=unsigned_dtype),
                dtype=unsigned_dtype,
            ),
            previous_previous_values,
            dtype=unsigned_dtype,
        )
        return payload[:body_offset] + values.tobytes()

    values = np.frombuffer(body, dtype=unsigned_dtype)
    previous_values = np.frombuffer(
        previous_payload[previous_offset:],
        dtype=unsigned_dtype,
    )
    previous_previous_values = np.frombuffer(
        previous_previous_payload[previous_previous_offset:],
        dtype=unsigned_dtype,
    )
    second_difference = np.add(
        np.subtract(values, previous_values, dtype=unsigned_dtype),
        np.subtract(
            previous_previous_values,
            previous_values,
            dtype=unsigned_dtype,
        ),
        dtype=unsigned_dtype,
    )
    encoded = _zigzag_unsigned(
        second_difference,
        unsigned_dtype,
        inverse=False,
    )
    encoded_body = _byte_shuffle(
        encoded.tobytes(),
        dtype.itemsize,
        inverse=False,
    )
    return payload[:body_offset] + encoded_body


def _apply_preconditioner(
    payload: bytes,
    previous_payload: bytes | None,
    preconditioner: str | None,
    *,
    inverse: bool,
    previous_previous_payload: bytes | None = None,
    member_name: str | None = None,
) -> bytes:
    if preconditioner == TEMPORAL_DELTA_PRECONDITIONER:
        return _temporal_npy_transform(
            payload,
            previous_payload,
            inverse=inverse,
        )
    if preconditioner == TEMPORAL_DELTA_SHUFFLE_PRECONDITIONER:
        return _temporal_npy_delta_shuffle_transform(
            payload,
            previous_payload,
            inverse=inverse,
        )
    if preconditioner == TEMPORAL_DELTA2_QUANT_SHUFFLE_PRECONDITIONER:
        return _temporal_npy_delta2_quant_shuffle_transform(
            payload,
            previous_payload,
            previous_previous_payload,
            member_name,
            inverse=inverse,
        )
    return payload


def _write_compressed_chunk(
    source_path: Path,
    infos: list[zipfile.ZipInfo],
    output_path: Path,
    compression_level: int,
    preconditioner: str | None,
) -> int:
    with open_archive(source_path, "r") as source:
        with bz2.BZ2File(output_path, "wb", compresslevel=compression_level) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                previous_payloads: deque[bytes] = deque(maxlen=2)
                for info in infos:
                    tar_info = _tar_info(info)
                    if info.is_dir():
                        archive.addfile(tar_info)
                    else:
                        payload = source.read(info.filename)
                        transformed = _apply_preconditioner(
                            payload,
                            previous_payloads[-1] if previous_payloads else None,
                            preconditioner,
                            inverse=False,
                            previous_previous_payload=(
                                previous_payloads[-2]
                                if len(previous_payloads) > 1
                                else None
                            ),
                            member_name=info.filename,
                        )
                        archive.addfile(tar_info, BytesIO(transformed))
                        previous_payloads.append(payload)
    return output_path.stat().st_size


def _patched_metadata(payload: bytes, updates: dict[str, Any] | None) -> bytes:
    if not updates:
        return payload
    metadata = json.loads(payload.decode("utf-8"))
    metadata.update(updates)
    return (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")


def repack_zip_to_channel_bzip2(
    source_path: Path,
    output_path: Path,
    *,
    chunk_frames: int = 5,
    compression_level: int = 9,
    workers: int = 1,
    temporal_delta: bool = False,
    temporal_delta_shuffle: bool = False,
    adaptive_temporal_order: bool = False,
    metadata_updates: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Repack a logical DDC archive into indexed channel chunks."""

    started = time.perf_counter()
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    if not 1 <= compression_level <= 9:
        raise ValueError("bzip2 compression_level must be in [1, 9]")
    if workers < 1:
        raise ValueError("workers must be positive")
    if sum((temporal_delta, temporal_delta_shuffle, adaptive_temporal_order)) > 1:
        raise ValueError(
            "temporal preconditioners are mutually exclusive"
        )
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError("source and output paths must differ")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if adaptive_temporal_order:
        preconditioner = ADAPTIVE_TEMPORAL_PRECONDITIONER
        container_format = CHANNEL_BZIP2_ADAPTIVE_FORMAT
    elif temporal_delta_shuffle:
        preconditioner = TEMPORAL_DELTA_SHUFFLE_PRECONDITIONER
        container_format = CHANNEL_BZIP2_DELTA_SHUFFLE_FORMAT
    elif temporal_delta:
        preconditioner = TEMPORAL_DELTA_PRECONDITIONER
        container_format = CHANNEL_BZIP2_DELTA_FORMAT
    else:
        preconditioner = None
        container_format = CHANNEL_BZIP2_FORMAT

    with open_archive(source_path, "r") as source:
        if isinstance(source, zipfile.ZipFile):
            infos = sorted(source.infolist(), key=lambda item: item.filename)
        else:
            infos = []
            for name in source.namelist():
                row = source.index["members"][name]
                info = zipfile.ZipInfo(name)
                info.file_size = int(row["size"])
                info.CRC = int(row["crc32"])
                infos.append(info)
        uncompressed_member_bytes = sum(
            info.file_size for info in infos if not info.is_dir()
        )
        infos_by_name = {info.filename: info for info in infos}
        grouped: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        direct_infos: list[zipfile.ZipInfo] = []
        for info in infos:
            if info.is_dir():
                continue
            group = _member_group(info.filename, chunk_frames)
            if group is None:
                direct_infos.append(info)
            else:
                grouped[group].append(info)
        direct_payloads = {
            info.filename: _patched_metadata(
                source.read(info.filename),
                metadata_updates if info.filename == "metadata.json" else None,
            )
            for info in direct_infos
        }

    group_rows = sorted(grouped.items())
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.chunks.",
        dir=output_path.parent,
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)

        def compress_indexed_group(
            indexed_group: tuple[int, tuple[str, list[zipfile.ZipInfo]]],
        ) -> dict[str, Any]:
            group_index, (group_name, group_infos) = indexed_group
            chunk_name = f"chunks/{group_index:05d}.tar.bz2"
            chunk_path = temporary_path / f"{group_index:05d}.tar.bz2"
            selected_preconditioner = preconditioner
            first_order_size: int | None = None
            second_order_size: int | None = None
            if adaptive_temporal_order:
                first_order_path = chunk_path.with_suffix(".delta1.tar.bz2")
                first_order_size = _write_compressed_chunk(
                    source_path,
                    group_infos,
                    first_order_path,
                    compression_level,
                    TEMPORAL_DELTA_SHUFFLE_PRECONDITIONER,
                )
                selected_preconditioner = TEMPORAL_DELTA_SHUFFLE_PRECONDITIONER
                selected_path = first_order_path
                if any(_is_quantized_member(info.filename) for info in group_infos):
                    second_order_path = chunk_path.with_suffix(".delta2.tar.bz2")
                    second_order_size = _write_compressed_chunk(
                        source_path,
                        group_infos,
                        second_order_path,
                        compression_level,
                        TEMPORAL_DELTA2_QUANT_SHUFFLE_PRECONDITIONER,
                    )
                    if second_order_size < first_order_size:
                        selected_preconditioner = (
                            TEMPORAL_DELTA2_QUANT_SHUFFLE_PRECONDITIONER
                        )
                        selected_path = second_order_path
                selected_path.replace(chunk_path)
                compressed_size = chunk_path.stat().st_size
            else:
                compressed_size = _write_compressed_chunk(
                    source_path,
                    group_infos,
                    chunk_path,
                    compression_level,
                    preconditioner,
                )
            return {
                "group": group_name,
                "name": chunk_name,
                "path": chunk_path,
                "compressed_size": compressed_size,
                "preconditioner": selected_preconditioner,
                "first_order_size": first_order_size,
                "second_order_size": second_order_size,
                "uncompressed_member_bytes": sum(
                    info.file_size for info in group_infos
                ),
                "members": [info.filename for info in group_infos],
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(compress_indexed_group, enumerate(group_rows)))

        index_members: dict[str, dict[str, Any]] = {}
        for info in direct_infos:
            if info.is_dir():
                continue
            payload = direct_payloads[info.filename]
            index_members[info.filename] = {
                "storage": "direct",
                "size": len(payload),
                "crc32": zlib.crc32(payload) & 0xFFFFFFFF,
            }
        for chunk in chunks:
            for member_name in chunk["members"]:
                info = infos_by_name[member_name]
                index_members[member_name] = {
                    "storage": "chunk",
                    "chunk": chunk["name"],
                    "size": info.file_size,
                    "crc32": info.CRC,
                }

        container = {
            "format": container_format,
            "compression": "bzip2",
            "compression_level": compression_level,
            "preconditioner": preconditioner,
            "chunk_frames": chunk_frames,
            "member_count": len(index_members),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    key: row[key]
                    for key in (
                        "group",
                        "name",
                        "compressed_size",
                        "preconditioner",
                        "uncompressed_member_bytes",
                        "members",
                    )
                }
                for row in chunks
            ],
        }
        index = {
            "format": container_format,
            "members": index_members,
        }
        temporary_output = output_path.with_name(f".{output_path.name}.tmp")
        temporary_output.unlink(missing_ok=True)
        try:
            with zipfile.ZipFile(temporary_output, "w", compression=zipfile.ZIP_STORED) as output:
                output.writestr(
                    _zip_info(CONTAINER_MEMBER),
                    json.dumps(container, indent=2, sort_keys=True) + "\n",
                )
                output.writestr(
                    _zip_info(INDEX_MEMBER),
                    json.dumps(index, indent=2, sort_keys=True) + "\n",
                )
                for name in sorted(direct_payloads):
                    output.writestr(_zip_info(name), direct_payloads[name])
                for chunk in chunks:
                    with output.open(_zip_info(chunk["name"]), "w") as destination:
                        with Path(chunk["path"]).open("rb") as source_handle:
                            shutil.copyfileobj(source_handle, destination, length=8 * 1024 * 1024)
            if output_path.exists():
                output_path.unlink()
            temporary_output.replace(output_path)
        finally:
            temporary_output.unlink(missing_ok=True)

    elapsed_seconds = time.perf_counter() - started
    source_bytes = source_path.stat().st_size
    output_bytes = output_path.stat().st_size
    first_order_stream_bytes = sum(
        int(chunk["first_order_size"])
        for chunk in chunks
        if chunk["first_order_size"] is not None
    )
    selected_stream_bytes = sum(int(chunk["compressed_size"]) for chunk in chunks)
    second_order_chunk_count = sum(
        chunk["preconditioner"]
        == TEMPORAL_DELTA2_QUANT_SHUFFLE_PRECONDITIONER
        for chunk in chunks
    )
    return {
        "format": container_format,
        "source": str(source_path),
        "output": str(output_path),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "uncompressed_member_bytes": uncompressed_member_bytes,
        "saving_vs_source_fraction": 1.0 - output_bytes / source_bytes,
        "elapsed_seconds": elapsed_seconds,
        "logical_input_mib_per_second": (
            uncompressed_member_bytes / 1048576.0 / elapsed_seconds
        ),
        "chunk_frames": chunk_frames,
        "compression_level": compression_level,
        "preconditioner": preconditioner,
        "adaptive_candidate_chunk_count": (
            sum(chunk["second_order_size"] is not None for chunk in chunks)
            if adaptive_temporal_order
            else 0
        ),
        "adaptive_second_order_chunk_count": second_order_chunk_count,
        "adaptive_first_order_stream_bytes": (
            first_order_stream_bytes if adaptive_temporal_order else None
        ),
        "adaptive_selected_stream_bytes": (
            selected_stream_bytes if adaptive_temporal_order else None
        ),
        "adaptive_saving_vs_first_order_fraction": (
            1.0 - selected_stream_bytes / first_order_stream_bytes
            if adaptive_temporal_order and first_order_stream_bytes
            else None
        ),
        "workers": workers,
        "chunk_count": len(chunks),
        "member_count": len(index_members),
    }


class ChannelBzip2Archive(AbstractContextManager["ChannelBzip2Archive"]):
    """Read an indexed channel-bzip2 DDC container like ``ZipFile``."""

    def __init__(self, path: Path, *, cache_chunks: int = 32) -> None:
        if cache_chunks < 0:
            raise ValueError("cache_chunks must be non-negative")
        self.path = Path(path)
        self._outer = zipfile.ZipFile(self.path, "r")
        try:
            self.container = json.loads(self._outer.read(CONTAINER_MEMBER))
            self.index = json.loads(self._outer.read(INDEX_MEMBER))
        except BaseException:
            self._outer.close()
            raise
        if self.container.get("format") not in CHANNEL_FORMATS:
            self._outer.close()
            raise ValueError(f"Unsupported channel container format in {self.path}")
        if self.index.get("format") != self.container.get("format"):
            self._outer.close()
            raise ValueError(f"Invalid channel container index in {self.path}")
        self._members: dict[str, dict[str, Any]] = self.index["members"]
        self._chunk_preconditioners = {
            row["name"]: row.get(
                "preconditioner",
                self.container.get("preconditioner"),
            )
            for row in self.container.get("chunks", [])
        }
        self._cache_chunks = cache_chunks
        self._cache: OrderedDict[str, dict[str, bytes]] = OrderedDict()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._cache.clear()
        self._outer.close()

    def namelist(self) -> list[str]:
        return sorted(self._members)

    def _chunk_members(self, chunk_name: str) -> dict[str, bytes]:
        cached = self._cache.pop(chunk_name, None)
        if cached is not None:
            self._cache[chunk_name] = cached
            return cached
        compressed = self._outer.read(chunk_name)
        if self.container["format"] == CHANNEL_DEFLATE_FORMAT:
            payload = zlib.decompress(compressed)
        else:
            payload = bz2.decompress(compressed)
        members: dict[str, bytes] = {}
        with tarfile.open(fileobj=BytesIO(payload), mode="r:") as archive:
            previous_payloads: deque[bytes] = deque(maxlen=2)
            for info in archive:
                if not info.isfile():
                    continue
                handle = archive.extractfile(info)
                if handle is None:
                    raise ValueError(f"Cannot read {info.name} from {chunk_name}")
                member_payload = handle.read()
                member_payload = _apply_preconditioner(
                    member_payload,
                    previous_payloads[-1] if previous_payloads else None,
                    self._chunk_preconditioners.get(
                        chunk_name,
                        self.container.get("preconditioner"),
                    ),
                    inverse=True,
                    previous_previous_payload=(
                        previous_payloads[-2]
                        if len(previous_payloads) > 1
                        else None
                    ),
                    member_name=info.name,
                )
                members[info.name] = member_payload
                previous_payloads.append(member_payload)
        if self._cache_chunks:
            self._cache[chunk_name] = members
            while len(self._cache) > self._cache_chunks:
                self._cache.popitem(last=False)
        return members

    def read(self, name: str) -> bytes:
        row = self._members[name]
        if row["storage"] == "direct":
            payload = self._outer.read(name)
        else:
            payload = self._chunk_members(row["chunk"])[name]
        self._validate_payload(name, payload)
        return payload

    def _validate_payload(self, name: str, payload: bytes) -> None:
        row = self._members[name]
        if len(payload) != int(row["size"]):
            raise ValueError(f"Size mismatch for {name} in {self.path}")
        if zlib.crc32(payload) & 0xFFFFFFFF != int(row["crc32"]):
            raise ValueError(f"CRC failure for {name} in {self.path}")

    def open(self, name: str, mode: str = "r") -> BinaryIO:
        if mode not in {"r", "rb"}:
            raise ValueError("ChannelBzip2Archive is read-only")
        return BytesIO(self.read(name))

    def testzip(self) -> str | None:
        bad_outer = self._outer.testzip()
        if bad_outer is not None:
            return bad_outer
        direct_names = sorted(
            name
            for name, row in self._members.items()
            if row["storage"] == "direct"
        )
        for name in direct_names:
            try:
                self.read(name)
            except (KeyError, OSError, EOFError, ValueError, tarfile.TarError):
                return name
        for chunk in self.container.get("chunks", []):
            members = list(chunk.get("members", []))
            try:
                payloads = self._chunk_members(chunk["name"])
                for name in members:
                    self._validate_payload(name, payloads[name])
            except (KeyError, OSError, EOFError, ValueError, tarfile.TarError):
                return members[0] if members else chunk.get("name")
        return None


def open_archive(
    path: Path | str,
    mode: str = "r",
    *,
    cache_chunks: int = 32,
) -> zipfile.ZipFile | ChannelBzip2Archive:
    """Open either a legacy Deflate archive or a channel-bzip2 container."""

    if mode != "r":
        raise ValueError("open_archive currently supports read mode only")
    archive = zipfile.ZipFile(path, "r")
    try:
        if CONTAINER_MEMBER not in archive.namelist():
            return archive
        container = json.loads(archive.read(CONTAINER_MEMBER))
        if container.get("format") not in CHANNEL_FORMATS:
            return archive
    except BaseException:
        archive.close()
        raise
    archive.close()
    return ChannelBzip2Archive(Path(path), cache_chunks=cache_chunks)
