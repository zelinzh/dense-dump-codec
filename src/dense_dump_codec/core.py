"""Numerical core for endpoint-predicted dense dump compression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-30
LOG_SPACE_DATASETS = frozenset({"prims.rho", "prims.u"})


@dataclass(frozen=True)
class QuantizedResidual:
    """A block-scaled residual plus optional unquantized float32 sparse exceptions."""

    values: np.ndarray
    scale: np.ndarray
    exception_indices: np.ndarray
    exception_values: np.ndarray

    @property
    def exception_count(self) -> int:
        return int(self.exception_indices.size)


def transformed(array: np.ndarray, dataset: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if dataset in LOG_SPACE_DATASETS:
        return np.log(np.maximum(values, EPS), dtype=np.float32)
    return values


def inverse_transformed(array: np.ndarray, dataset: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if dataset in LOG_SPACE_DATASETS:
        return np.exp(values, dtype=np.float32)
    return values


def linear_predictor(
    start: np.ndarray,
    end: np.ndarray,
    tau: float,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_transformed = transformed(start, dataset)
    end_transformed = transformed(end, dataset)
    predicted_transformed = (
        (1.0 - tau) * start_transformed + tau * end_transformed
    ).astype(np.float32, copy=False)
    predicted = inverse_transformed(predicted_transformed, dataset)
    return predicted, predicted_transformed, start_transformed


def spatial_axes(array: np.ndarray) -> tuple[int, ...]:
    if array.ndim < 3:
        return tuple(range(array.ndim))
    return tuple(range(array.ndim - 3, array.ndim))


def _validate_quantization(bits: int, scale_mode: str, scale_percentile: float) -> None:
    if not 4 <= bits <= 16:
        raise ValueError(f"bits must be in [4, 16], got {bits}")
    if scale_mode not in {"frame", "block-channel"}:
        raise ValueError(f"Unknown scale mode {scale_mode!r}")
    if not 0.0 < scale_percentile <= 100.0:
        raise ValueError(f"scale_percentile must be in (0, 100], got {scale_percentile}")


def residual_scale(
    residual: np.ndarray,
    bits: int,
    scale_mode: str,
    scale_percentile: float = 100.0,
) -> tuple[np.ndarray, int]:
    _validate_quantization(bits, scale_mode, scale_percentile)
    qmax = (1 << (bits - 1)) - 1
    absolute = np.abs(np.asarray(residual, dtype=np.float32))
    axes = None if scale_mode == "frame" else spatial_axes(absolute)
    if scale_percentile == 100.0:
        threshold = np.nanmax(absolute, axis=axes, keepdims=axes is not None)
    else:
        threshold = np.nanpercentile(
            absolute,
            scale_percentile,
            axis=axes,
            keepdims=axes is not None,
        )
    scale = np.asarray(threshold / qmax, dtype=np.float32)
    scale = np.maximum(scale, np.asarray(EPS, dtype=np.float32))
    return scale, qmax


def quantize_residual(
    residual: np.ndarray,
    bits: int,
    scale_mode: str,
    *,
    scale_percentile: float = 100.0,
    preserve_outliers: bool = True,
) -> QuantizedResidual:
    residual_values = np.asarray(residual, dtype=np.float32)
    if not np.isfinite(residual_values).all():
        raise ValueError("residual contains non-finite values")
    scale, qmax = residual_scale(
        residual_values,
        bits,
        scale_mode,
        scale_percentile,
    )
    scaled = residual_values / scale
    quantized = np.rint(scaled).clip(-qmax, qmax)
    storage_dtype = np.int8 if bits <= 8 else np.int16
    quantized = quantized.astype(storage_dtype, copy=False)

    if preserve_outliers and scale_percentile < 100.0:
        clipped = np.abs(scaled) > qmax
        flat_indices = np.flatnonzero(clipped)
        index_dtype = np.uint32 if residual_values.size <= np.iinfo(np.uint32).max else np.uint64
        exception_indices = flat_indices.astype(index_dtype, copy=False)
        exception_values = residual_values.reshape(-1)[flat_indices].astype(np.float32, copy=False)
    else:
        exception_indices = np.empty(0, dtype=np.uint32)
        exception_values = np.empty(0, dtype=np.float32)

    return QuantizedResidual(
        values=quantized,
        scale=scale,
        exception_indices=exception_indices,
        exception_values=exception_values,
    )


def dequantize_residual(payload: QuantizedResidual) -> np.ndarray:
    residual = (payload.values.astype(np.float32) * payload.scale).astype(
        np.float32,
        copy=False,
    )
    if payload.exception_count:
        residual = residual.copy()
        residual.reshape(-1)[payload.exception_indices.astype(np.int64)] = payload.exception_values
    return residual


def pack_int4(quantized: np.ndarray) -> np.ndarray:
    flat = np.asarray(quantized, dtype=np.int8).reshape(-1)
    if np.any(flat < -7) or np.any(flat > 7):
        raise ValueError("Signed symmetric int4 values must lie in [-7, 7]")
    encoded = (flat + 8).astype(np.uint8, copy=False)
    if encoded.size % 2:
        encoded = np.concatenate([encoded, np.zeros(1, dtype=np.uint8)])
    return (encoded[0::2] | (encoded[1::2] << np.uint8(4))).astype(np.uint8, copy=False)


def unpack_int4(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    encoded = np.empty(size + size % 2, dtype=np.uint8)
    packed_values = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if packed_values.size * 2 < size:
        raise ValueError(f"Packed int4 payload is too short for shape {shape}")
    encoded[0::2] = packed_values & np.uint8(0x0F)
    encoded[1::2] = packed_values >> np.uint8(4)
    return (encoded[:size].astype(np.int16) - 8).astype(np.int8).reshape(shape)
