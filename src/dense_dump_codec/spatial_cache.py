"""Lossless, radial-slab indexed working copies of immutable quantized DDC data."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .archive import open_archive
from .fast_blocks import compress, decompress, decompress_view


FORMAT = "ddc_spatial_working_lz4_v1"
HEADER = struct.Struct("<8sQI")
MAGIC = b"DDCSW01\n"
DATASETS = ("prims.rho", "prims.u", "prims.uvec", "prims.B")


def source_identity(source):
    source = Path(source).resolve()
    stat = source.stat()
    return dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, device=stat.st_dev, inode=stat.st_ino)


def cache_path(source, directory):
    token = hashlib.sha256(json.dumps(source_identity(source), sort_keys=True).encode()).hexdigest()
    return Path(directory) / f"{token}.ddcw"


def _array_record(values, member, temporary):
    values = np.ascontiguousarray(values)
    raw = values.tobytes()
    compressed = compress(raw)
    checksum = zlib.crc32(raw)
    if decompress(compressed, len(raw), checksum) != raw:
        raise ValueError("Working block round-trip differs")
    (temporary / member).write_bytes(compressed)
    return dict(member=member, shape=values.shape, dtype=values.dtype.str,
                bytes=len(raw), crc32=checksum, compressed_bytes=len(compressed))


def prepare_spatial_cache(source, directory, *, codec, slab_cells=32, workers=4):
    """Preserve codes/scales/exceptions exactly, validate, then publish without overwrite."""
    source, directory = Path(source), Path(directory)
    if slab_cells < 1 or workers < 1:
        raise ValueError("Slab length and workers must be positive")
    identity = source_identity(source)
    output = cache_path(source, directory)
    if output.exists():
        raise FileExistsError(output)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for payload in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(payload)
    with open_archive(source) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    if not metadata.get("middle_taus"):
        raise ValueError("Spatial cache requires explicit physical-time interpolation weights")
    with tempfile.TemporaryDirectory(prefix=".ddcw_build_", dir=directory) as temporary:
        temporary = Path(temporary)

        def channel(task):
            dataset, first, last = task
            rows = {}
            with open_archive(source, cache_chunks=8) as archive:
                bits = codec.dataset_bits(metadata, dataset)
                tiles = metadata.get("dataset_tile_shapes", {}).get(dataset)
                if tiles and slab_cells % tiles[-1]:
                    raise ValueError("Spatial slabs must align with compact scale tiles")
                for frame in range(first, last):
                    member = lambda suffix: codec.archive_member_name(dataset, frame, suffix)
                    codes = codec.read_quantized(archive, dataset, frame, bits)
                    scale = codec.np_load_member(archive, member("scale"))
                    if member("exception_indices") in archive.namelist():
                        indices = codec.np_load_member(archive, member("exception_indices")).astype("i8")
                        values = codec.np_load_member(archive, member("exception_values"))
                    else:
                        indices, values = np.empty(0, dtype="i8"), np.empty(0, dtype="f4")
                    if (indices.ndim != 1 or values.shape != indices.shape
                            or np.any(indices < 0) or np.any(indices >= codes.size)):
                        raise ValueError("Invalid source exception indices")
                    radial = codes.shape[-1]
                    rows[str(frame)] = dict(shape=codes.shape, tile_shape=tiles, slabs=[])
                    for begin in range(0, radial, slab_cells):
                        end = min(begin + slab_cells, radial)
                        selected = (indices % radial >= begin) & (indices % radial < end)
                        local_indices = (indices[selected] // radial) * (end - begin) + (
                            indices[selected] % radial - begin)
                        factors = scale[..., begin // tiles[-1]:end // tiles[-1]] if tiles else scale
                        arrays = dict(codes=codes[..., begin:end], scale=factors,
                                      indices=local_indices, values=values[selected])
                        records = {name: _array_record(array,
                            f"{dataset}_{frame:03d}_{begin:05d}_{name}.lz4", temporary)
                            for name, array in arrays.items()}
                        rows[str(frame)]["slabs"].append(dict(begin=begin, end=end, arrays=records))
            return dataset, rows

        count = len(metadata["middle_taus"])
        tasks = [(dataset, first, min(first + 5, count + 1))
                 for dataset in DATASETS for first in range(1, count + 1, 5)]
        channels = {dataset: {} for dataset in DATASETS}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for dataset, rows in pool.map(channel, tasks):
                channels[dataset].update(rows)
        index = dict(format=FORMAT, source=identity, source_sha256=digest.hexdigest(),
                     slab_cells=slab_cells, channels=channels, metadata=metadata,
                     re_quantized=False)
        candidate = temporary / "candidate.ddcw"
        blocks, offset = [], 0
        for frames in channels.values():
            for frame in frames.values():
                for slab in frame["slabs"]:
                    for record in slab["arrays"].values():
                        record["offset"] = offset
                        offset += record["compressed_bytes"]
                        blocks.append(temporary / record["member"])
        payload = json.dumps(index, sort_keys=True).encode()
        with candidate.open("wb") as handle:
            handle.write(HEADER.pack(MAGIC, len(payload), zlib.crc32(payload)))
            handle.write(payload)
            for block in blocks:
                with block.open("rb") as source_block:
                    shutil.copyfileobj(source_block, handle, 4 * 1024 * 1024)
        with SpatialWorkingArchive(candidate, source=source) as archive:
            archive.validate_all()
        if identity != source_identity(source):
            raise ValueError("Source changed while the working cache was built")
        output.hardlink_to(candidate)
    return dict(source=str(source), output=str(output), source_bytes=identity["size"],
                output_bytes=output.stat().st_size, elapsed_seconds=time.perf_counter() - started,
                source_sha256=digest.hexdigest(), re_quantized=False, all_blocks_crc_verified=True,
                workers=workers)


@functools.lru_cache(maxsize=16)
def _load_index(path, size, mtime_ns, ctime_ns):
    with Path(path).open("rb") as handle:
        header = handle.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("Truncated spatial cache header")
        magic, length, checksum = HEADER.unpack(header)
        if magic != MAGIC or length > 16 * 1024 * 1024 or length + HEADER.size > size:
            raise ValueError("Invalid spatial cache header")
        payload = handle.read(length)
    if zlib.crc32(payload) != checksum:
        raise ValueError("Spatial index CRC mismatch")
    return json.loads(payload), HEADER.size + length


class SpatialWorkingArchive:
    """Read only intersecting spatial slabs; no archive-wide entropy stream is decoded."""

    def __init__(self, path, *, source=None):
        path = Path(path).resolve()
        self.descriptor = os.open(path, os.O_RDONLY)
        try:
            stat = os.fstat(self.descriptor)
            self.index, self.data_offset = _load_index(
                str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if self.index["format"] != FORMAT:
                raise ValueError("Unknown spatial working format")
            if source is not None and self.index["source"] != source_identity(source):
                raise ValueError("Stale spatial working cache")
        except BaseException:
            os.close(self.descriptor)
            raise
        self.container = {"chunk_frames": 1}
        self.blocks_read, self.compressed_bytes_read = 0, 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.close(self.descriptor)

    def _array(self, record):
        dtype = np.dtype(record["dtype"])
        shape = tuple(record["shape"])
        if (dtype.hasobject or dtype.kind not in "biufc" or any(length < 0 for length in shape)
                or math.prod(shape) * dtype.itemsize != record["bytes"]):
            raise ValueError("Invalid spatial array descriptor")
        compressed_size = record["compressed_bytes"]
        if not 0 < compressed_size <= 512 * 1024 * 1024 or record["offset"] < 0:
            raise ValueError("Invalid compressed spatial block extent")
        payload = os.pread(self.descriptor, compressed_size, self.data_offset + record["offset"])
        if len(payload) != compressed_size:
            raise ValueError("Truncated spatial block")
        raw = decompress_view(payload, record["bytes"], record["crc32"])
        self.blocks_read += 1
        self.compressed_bytes_read += len(payload)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def validate_all(self):
        for frames in self.index["channels"].values():
            for frame in frames.values():
                for slab in frame["slabs"]:
                    for record in slab["arrays"].values():
                        self._array(record)

    def quantized_region(self, dataset, frame_index, stop=None):
        row = self.index["channels"][dataset][str(frame_index)]
        shape = tuple(row["shape"])
        stop = shape[-1] if stop is None else stop
        if not 0 < stop <= shape[-1]:
            raise ValueError("Spatial region outside cached source")
        tiles = row["tile_shape"]
        if tiles and stop % tiles[-1]:
            raise ValueError("Spatial region must align with scale tiles")
        codes = np.empty((*shape[:-1], stop), dtype=row["slabs"][0]["arrays"]["codes"]["dtype"])
        scales, exceptions, exception_values = [], [], []
        cursor = 0
        for slab in row["slabs"]:
            begin, end = slab["begin"], slab["end"]
            if begin >= stop:
                break
            if begin != cursor or end <= begin:
                raise ValueError("Noncontiguous spatial slab index")
            arrays = {name: self._array(record) for name, record in slab["arrays"].items()}
            retained_end = min(stop, end)
            codes[..., begin:retained_end] = arrays["codes"][..., :retained_end - begin]
            scales.append(arrays["scale"][..., :(retained_end - begin) // tiles[-1]]
                          if tiles else arrays["scale"])
            indices = arrays["indices"]
            width = end - begin
            if (indices.ndim != 1 or indices.dtype.kind not in "iu"
                    or indices.shape != arrays["values"].shape
                    or np.any(indices < 0) or np.any(indices >= arrays["codes"].size)):
                raise ValueError("Invalid cached exception indices")
            selected = indices % width < retained_end - begin
            exceptions.append(indices[selected] // width * stop + indices[selected] % width + begin)
            exception_values.append(arrays["values"][selected])
            cursor = end
        if cursor < stop:
            raise ValueError("Incomplete spatial cache coverage")
        scale = np.concatenate(scales, axis=-1) if tiles else scales[0]
        if not tiles and any(not np.array_equal(scale, other) for other in scales[1:]):
            raise ValueError("Inconsistent native-block scales in spatial cache")
        return (codes, scale, np.concatenate(exceptions), np.concatenate(exception_values), tiles)
