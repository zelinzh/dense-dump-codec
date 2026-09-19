"""Predictive codec primitives for dense simulation dump sequences."""

__version__ = "1.0.0"

from .core import (
    EPS,
    LOG_SPACE_DATASETS,
    QuantizedResidual,
    dequantize_residual,
    inverse_transformed,
    linear_predictor,
    pack_int4,
    quantize_residual,
    transformed,
    unpack_int4,
)
from .archive import (
    CHANNEL_BZIP2_ADAPTIVE_FORMAT,
    CHANNEL_BZIP2_DELTA_SHUFFLE_FORMAT,
    CHANNEL_BZIP2_DELTA_FORMAT,
    CHANNEL_BZIP2_FORMAT,
    ChannelBzip2Archive,
    open_archive,
    repack_zip_to_channel_bzip2,
)
from .keyframe import (
    KEYFRAME_BZIP2_SHUFFLE_FORMAT,
    KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT,
    KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT,
    KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT,
    KEYFRAME_XZ_SHUFFLE_FORMAT,
    compress_keyframe,
    decompress_keyframe,
    validate_keyframe_archive,
)
from .sequence import DDCFrameMaterializer, DenseSequenceIndex, FrameBracket, FrameRecord

__all__ = [
    "__version__",
    "EPS",
    "LOG_SPACE_DATASETS",
    "QuantizedResidual",
    "dequantize_residual",
    "inverse_transformed",
    "linear_predictor",
    "pack_int4",
    "quantize_residual",
    "transformed",
    "unpack_int4",
    "CHANNEL_BZIP2_FORMAT",
    "CHANNEL_BZIP2_ADAPTIVE_FORMAT",
    "CHANNEL_BZIP2_DELTA_FORMAT",
    "CHANNEL_BZIP2_DELTA_SHUFFLE_FORMAT",
    "ChannelBzip2Archive",
    "open_archive",
    "repack_zip_to_channel_bzip2",
    "KEYFRAME_BZIP2_SHUFFLE_FORMAT",
    "KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT",
    "KEYFRAME_XZ_SHUFFLE_FORMAT",
    "KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT",
    "KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT",
    "compress_keyframe",
    "decompress_keyframe",
    "validate_keyframe_archive",
    "DDCFrameMaterializer",
    "DenseSequenceIndex",
    "FrameBracket",
    "FrameRecord",
]
