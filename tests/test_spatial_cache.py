import json
from pathlib import Path

import numpy as np
import pytest

from dense_dump_codec.fast_blocks import compress, decompress
from dense_dump_codec.native import NativeSequenceDecoder
from dense_dump_codec.spatial_cache import (
    SpatialWorkingArchive, cache_path, prepare_spatial_cache,
)
from scripts import decode_dump_codec as codec
from test_native_runtime import sequence_manifest


def test_lossless_blocks_and_corruption():
    import zlib
    for payload in (b"", b"abc", bytes(range(256)) * 100):
        packed = compress(payload)
        assert decompress(packed, len(payload), zlib.crc32(payload)) == payload
        with pytest.raises(ValueError):
            decompress(packed, len(payload), zlib.crc32(payload) ^ 1)
        with pytest.raises(ValueError):
            decompress(packed[:-1], len(payload), zlib.crc32(payload))


def test_working_cache_is_exact_and_skips_unneeded_slabs(sequence_manifest, tmp_path):
    manifest = json.loads(sequence_manifest.read_text())
    paths = next(iter(manifest["codec_schemes"].values()))["archive_paths"]
    working = tmp_path / "working"
    for source in paths:
        result = prepare_spatial_cache(source, working, codec=codec, slab_cells=1, workers=2)
        assert result["all_blocks_crc_verified"] and not result["re_quantized"]
        with pytest.raises(FileExistsError):
            prepare_spatial_cache(source, working, codec=codec)
    reference = NativeSequenceDecoder(sequence_manifest)(range(9))
    actual = NativeSequenceDecoder(sequence_manifest, working_cache=working)(range(9))
    for sequence in range(9):
        for name, values in reference[sequence]["datasets"].items():
            assert values.tobytes() == actual[sequence]["datasets"][name].tobytes()
    path = cache_path(paths[0], working)
    with SpatialWorkingArchive(path) as archive:
        full = archive.quantized_region("prims.u", 1)
        all_reads = archive.blocks_read
    with SpatialWorkingArchive(path) as archive:
        partial = archive.quantized_region("prims.u", 1, 1)
        assert archive.blocks_read * 2 == all_reads
    assert partial[0].tobytes() == full[0][..., :1].tobytes()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    broken = tmp_path / "broken.ddcw"
    broken.write_bytes(data)
    with SpatialWorkingArchive(broken) as archive, pytest.raises(ValueError):
        archive.validate_all()
    broken.write_bytes(data[:10])
    with pytest.raises(ValueError, match="header"):
        SpatialWorkingArchive(broken)
    original = Path(paths[0])
    original.touch()
    assert cache_path(original, working) != path
    with pytest.raises(ValueError, match="Stale"):
        SpatialWorkingArchive(path, source=original)
    fallback = NativeSequenceDecoder(sequence_manifest, working_cache=working)((1,))[1]
    assert fallback["datasets"]["prims.u"].tobytes() == reference[1]["datasets"]["prims.u"].tobytes()
