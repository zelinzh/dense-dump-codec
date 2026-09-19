"""Local residual scales in native array order, compatible with existing archives."""

from __future__ import annotations

import numpy as np

from .core import QuantizedResidual, quantize_residual
from .reconstruction import scaled_residual, tile_view


def native_view(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    leading = len(shape) - 3
    order = (*range(leading), leading, leading + 3, leading + 1,
             leading + 4, leading + 2, leading + 5)
    return values.transpose(order).reshape(shape)


def quantize_tiled(residual, tile_shape, bits=8, *, scale_percentile=99.9,
                   preserve_outliers=True):
    residual = np.asarray(residual, dtype=np.float32)
    tiled = tile_view(residual, tile_shape)
    payload = quantize_residual(
        tiled, bits, "block-channel", scale_percentile=scale_percentile,
        preserve_outliers=preserve_outliers,
    )
    marked = np.zeros(tiled.shape, dtype=bool)
    marked.reshape(-1)[payload.exception_indices] = True
    indices = np.flatnonzero(native_view(marked, residual.shape))
    index_dtype = np.uint32 if residual.size <= np.iinfo(np.uint32).max else np.uint64
    indices = indices.astype(index_dtype, copy=False)
    return QuantizedResidual(
        native_view(payload.values, residual.shape), payload.scale[..., 0, 0, 0].copy(),
        indices, residual.reshape(-1)[indices].copy(),
    )


def dequantize_tiled(payload, tile_shape):
    residual = scaled_residual(payload.values, payload.scale, tile_shape=tile_shape)
    residual.reshape(-1)[payload.exception_indices] = payload.exception_values
    return residual


def parse_dataset_tiles(value: str, datasets) -> dict[str, tuple[int, int, int]]:
    result = {}
    for assignment in filter(None, (item.strip() for item in value.split(","))):
        dataset, separator, dimensions = assignment.partition("=")
        if not separator or dataset not in datasets or dataset in result:
            raise ValueError("Tile assignments require unique selected DATASET=PHIxTHETAxR names")
        shape = tuple(int(part) for part in dimensions.split("x"))
        if len(shape) != 3 or min(shape) <= 0:
            raise ValueError("Tile dimensions must be three positive integers")
        result[dataset] = shape
    return result
