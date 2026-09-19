"""Allocation-bounded reconstruction with the reference float32 operation order."""

from __future__ import annotations

import math
import threading
from io import BytesIO

import numpy as np

from .core import LOG_SPACE_DATASETS, inverse_transformed
from .region import radial_exceptions


def read_array_member(archive, name):
    """Return a read-only numeric NPY view owning its validated archive bytes."""
    payload = archive.read(name)
    stream = BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        result = np.load(BytesIO(payload), allow_pickle=False)
        result.flags.writeable = False
        return result
    if dtype.hasobject:
        raise ValueError("Object arrays are not supported in DDC")
    count = math.prod(shape)
    offset = stream.tell()
    if any(length < 0 for length in shape) or dtype.itemsize == 0:
        raise ValueError("Invalid DDC array shape or dtype")
    if count * dtype.itemsize != len(payload) - offset:
        raise ValueError("Invalid or truncated DDC NPY payload")
    return np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(
        shape, order="F" if fortran_order else "C"
    )


def tile_view(values, tile_shape):
    """View native (..., phi, theta, radius) arrays as compact spatial tiles."""
    tiles = tuple(int(length) for length in tile_shape)
    if len(tiles) != 3 or min(tiles) <= 0 or values.ndim < 3:
        raise ValueError("Tile shape must contain three positive spatial dimensions")
    spatial = values.shape[-3:]
    if any(length % tile for length, tile in zip(spatial, tiles)):
        raise ValueError("Tile shape must divide the spatial array shape")
    leading = values.shape[:-3]
    interleaved = tuple(part for length, tile in zip(spatial, tiles)
                        for part in (length // tile, tile))
    offset = len(leading)
    order = (*range(offset), offset, offset + 2, offset + 4,
             offset + 1, offset + 3, offset + 5)
    return values.reshape(*leading, *interleaved).transpose(order)


def scaled_residual(codes, scale, *, tile_shape=None, out=None):
    """Apply native or compact tile scales without expanding a scale grid."""
    if out is None:
        out = np.empty(codes.shape, dtype=np.float32)
    if tile_shape is None:
        source, target, factors = codes, out, scale
    else:
        source, target = tile_view(codes, tile_shape), tile_view(out, tile_shape)
        if scale.shape != source.shape[:-3]:
            raise ValueError("Compact tile scales do not match the code shape")
        factors = scale[..., None, None, None]
    if scale.dtype == np.dtype("float32") and codes.dtype in (np.dtype("int8"),
                                                              np.dtype("int16")):
        np.multiply(source, factors, out=target, dtype=np.float32)
    else:
        target[...] = (source.astype(np.float32) * factors).astype(np.float32)
    return out


class Reconstruction:
    """Reuse thread-local scratch; every returned frame owns its output storage."""

    def __init__(self, codec, mode="auto", radial_stop=None, predictor_library=None):
        if mode not in {"auto", "numpy", "reference"}:
            raise ValueError("Unknown reconstruction mode")
        self.codec = codec
        self.mode = "numpy" if mode == "auto" else mode
        if radial_stop is not None and self.mode != "numpy":
            raise ValueError("Radial ROI requires numpy reconstruction")
        self.radial_stop = radial_stop
        self.predictor = None
        if predictor_library is not None:
            if self.mode != "numpy":
                raise ValueError("Native prediction requires numpy reconstruction")
            from .predict_kernel import FusedPredictor
            self.predictor = FusedPredictor(predictor_library)
        self._local = threading.local()

    def decode(self, archive, metadata, dataset, frame_index, *, start, end,
               start_transformed, end_transformed):
        if self.mode == "reference":
            if dataset in metadata.get("dataset_tile_shapes", {}):
                tau = self.codec.frame_tau(archive, metadata, (dataset,), frame_index)
                predicted = ((1.0 - tau) * start_transformed + tau * end_transformed).astype(
                    np.float32, copy=False)
                bits = self.codec.dataset_bits(metadata, dataset)
                codes = self.codec.read_quantized(archive, dataset, frame_index, bits)
                member = lambda suffix: self.codec.archive_member_name(dataset, frame_index, suffix)
                scale = self.codec.np_load_member(archive, member("scale"))
                residual = scaled_residual(codes, scale,
                                           tile_shape=metadata["dataset_tile_shapes"][dataset])
                if member("exception_indices") in archive.namelist():
                    indices = self.codec.np_load_member(archive, member("exception_indices"))
                    values = self.codec.np_load_member(archive, member("exception_values"))
                    residual.reshape(-1)[indices.astype(np.int64)] = values
                return inverse_transformed(predicted + residual, dataset)
            return self.codec.decode_dataset(
                archive, metadata, dataset, frame_index, start=start, end=end,
                start_transformed=start_transformed, end_transformed=end_transformed
            )
        if (start_transformed.dtype != np.dtype("float32")
                or end_transformed.dtype != np.dtype("float32")
                or start_transformed.shape != end_transformed.shape):
            if self.radial_stop is not None:
                raise ValueError("Radial ROI requires matched float32 anchors")
            return self.codec.decode_dataset(
                archive, metadata, dataset, frame_index, start=start, end=end,
                start_transformed=start_transformed, end_transformed=end_transformed
            )
        predictor = self.predictor
        if not start_transformed.flags.c_contiguous or not end_transformed.flags.c_contiguous:
            predictor = None
        result = np.empty(start_transformed.shape, dtype=np.float32)
        buffer = getattr(self._local, "buffer", None)
        if buffer is None or buffer.size < result.size:
            buffer = np.empty(result.size, dtype=np.float32)
            self._local.buffer = buffer
        scratch = buffer[:result.size].reshape(result.shape)
        tau = self.codec.frame_tau(archive, metadata, (dataset,), frame_index)
        if predictor is None:
            np.multiply(start_transformed, 1.0 - tau, out=result)
            np.multiply(end_transformed, tau, out=scratch)
            np.add(result, scratch, out=result)
        if hasattr(archive, "quantized_region"):
            codes, scale, indices, values, tile_shape = archive.quantized_region(
                dataset, frame_index, self.radial_stop)
            if codes.shape != result.shape:
                raise ValueError("Spatial codes and anchors differ in shape")
            scaled_residual(codes, scale, tile_shape=tile_shape, out=scratch)
            scratch.reshape(-1)[indices.astype(np.int64)] = values
        else:
            member = lambda suffix: self.codec.archive_member_name(dataset, frame_index, suffix)
            bits = self.codec.dataset_bits(metadata, dataset)
            if bits == 4:
                packed = read_array_member(archive, member("q4_packed"))
                shape = tuple(int(length) for length in read_array_member(archive, member("q4_shape")))
                codes = self.codec.unpack_int4(packed, shape)
            else:
                codes = read_array_member(archive, member(f"q{bits}"))
            source_cells = codes.shape[-1]
            if self.radial_stop is not None:
                codes = codes[..., :self.radial_stop]
            if codes.shape != result.shape:
                raise ValueError("DDC quantized array and anchor shape differ")
            scale = read_array_member(archive, member("scale"))
            tile_shape = metadata.get("dataset_tile_shapes", {}).get(dataset)
            if self.radial_stop is not None and tile_shape is not None:
                if self.radial_stop % tile_shape[-1]:
                    raise ValueError("Radial ROI must align with compact scale tiles")
                scale = scale[..., :self.radial_stop // tile_shape[-1]]
            scaled_residual(codes, scale, tile_shape=tile_shape, out=scratch)
            if member("exception_indices") in archive.namelist():
                indices = read_array_member(archive, member("exception_indices"))
                values = read_array_member(archive, member("exception_values"))
                if self.radial_stop is not None:
                    indices, values = radial_exceptions(indices, values, source_cells,
                                                         self.radial_stop)
                scratch.reshape(-1)[indices.astype(np.int64)] = values
        if predictor is None:
            np.add(result, scratch, out=result)
        else:
            predictor(start_transformed, end_transformed, scratch, tau, out=result)
        if dataset in LOG_SPACE_DATASETS:
            np.exp(result, out=result, dtype=np.float32)
        return result
