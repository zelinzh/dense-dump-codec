"""Optional system-lib LZ4 blocks with explicit bounds and decoded-byte CRCs."""

import ctypes
import ctypes.util
import functools
import zlib

import numpy as np


MAX_BLOCK_BYTES = 256 * 1024 * 1024


@functools.lru_cache(maxsize=1)
def library():
    name = ctypes.util.find_library("lz4")
    if not name:
        raise RuntimeError("Spatial working caches require the optional system liblz4")
    handle = ctypes.CDLL(name)
    handle.LZ4_compressBound.argtypes = [ctypes.c_int]
    handle.LZ4_compressBound.restype = ctypes.c_int
    for function in (handle.LZ4_compress_default, handle.LZ4_decompress_safe):
        function.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        function.restype = ctypes.c_int
    return handle


def compress(payload):
    if len(payload) > MAX_BLOCK_BYTES:
        raise ValueError("Working block exceeds allocation limit")
    handle = library()
    output = ctypes.create_string_buffer(handle.LZ4_compressBound(len(payload)))
    size = handle.LZ4_compress_default(payload, output, len(payload), len(output))
    if size <= 0:
        raise ValueError("LZ4 block compression failed")
    return output.raw[:size]


def decompress(payload, size, crc):
    return bytes(decompress_view(payload, size, crc))


def decompress_view(payload, size, crc):
    """Decode into owned uninitialized storage and expose a read-only byte view."""
    if not 0 <= size <= MAX_BLOCK_BYTES or len(payload) > MAX_BLOCK_BYTES * 2:
        raise ValueError("Invalid working block size")
    output = np.empty(size, dtype=np.uint8)
    actual = library().LZ4_decompress_safe(payload, output.ctypes.data, len(payload), size)
    if actual != size:
        raise ValueError("Invalid or truncated LZ4 working block")
    output.flags.writeable = False
    raw = memoryview(output)
    if zlib.crc32(raw) != crc:
        raise ValueError("Working block decoded CRC mismatch")
    return raw
