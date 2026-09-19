import gc
import zlib

import numpy as np
import pytest

from dense_dump_codec.fast_blocks import compress, decompress, decompress_view


@pytest.mark.parametrize("payload", [b"", b"ab", bytes(range(256)) * 4096])
def test_owned_readonly_block_view(payload):
    packed = compress(payload)
    view = decompress_view(packed, len(payload), zlib.crc32(payload))
    assert view.readonly
    array = np.frombuffer(view, dtype=np.uint8)
    del packed, view
    gc.collect()
    assert array.tobytes() == payload
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)
    assert type(decompress(compress(payload), len(payload), zlib.crc32(payload))) is bytes


def test_view_rejects_invalid_size_and_corruption():
    payload = b"numeric block" * 20
    packed = compress(payload)
    for size in (-1, 2 ** 30, len(payload) - 1, len(payload) + 1):
        with pytest.raises(ValueError):
            decompress_view(packed, size, zlib.crc32(payload))
    with pytest.raises(ValueError, match="CRC"):
        decompress_view(packed, len(payload), 0)
    with pytest.raises(ValueError):
        decompress_view(packed[:-1], len(payload), zlib.crc32(payload))
