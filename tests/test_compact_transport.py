import json
import struct

import numpy as np
import pytest

from dense_dump_codec.compact import CompactSequenceDecoder, compact_wire_parts
from dense_dump_codec.native import NativeSequenceDecoder
from dense_dump_codec.reconstruction import scaled_residual
from dense_dump_codec.spatial_cache import prepare_spatial_cache
from scripts import decode_dump_codec as codec
from test_native_runtime import sequence_manifest


def restore(frame):
    if frame['exact_keyframe']:
        return frame['anchors'][frame['start']][1]
    first = frame['anchors'][frame['start']][2]
    last = frame['anchors'][frame['end']][2]
    result = {}
    for name, channel in frame['channels'].items():
        codes, scales, indices, values, tiles = channel
        residual = scaled_residual(codes, scales, tile_shape=tiles)
        residual.reshape(-1)[indices] = values
        decoded = ((1-frame['tau'])*first[name] + frame['tau']*last[name]) + residual
        if name in ('prims.rho', 'prims.u'):
            decoded = np.exp(decoded, dtype=np.float32)
        result[name] = decoded
    return result


def test_compact_wire_roundtrip_and_anchor_reuse(sequence_manifest, tmp_path):
    manifest = json.loads(sequence_manifest.read_text())
    paths = next(iter(manifest['codec_schemes'].values()))['archive_paths']
    working = tmp_path / 'working'
    for source in paths:
        prepare_spatial_cache(source, working, codec=codec, slab_cells=1, workers=2)
    frames = CompactSequenceDecoder(sequence_manifest, working_cache=working)(range(9))
    truth = NativeSequenceDecoder(sequence_manifest)(range(9))
    for sequence, frame in frames.items():
        for name, values in restore(frame).items():
            assert values.tobytes() == truth[sequence]['datasets'][name].tobytes()
        header, pieces = compact_wire_parts(frame)
        assert header.startswith(b'STAGEQ1 ')
        assert int(header.split()[-1]) == sum(piece.nbytes for piece in pieces)
        metadata = struct.unpack('<7Qd', pieces[2][:64])
        assert metadata[0] == 1 and metadata[4] == len(frame['anchors'])
        _, reused = compact_wire_parts(frame, tuple(frame['anchors']))
        assert sum(piece.nbytes for piece in reused) < sum(piece.nbytes for piece in pieces)
        for name, channel in frame['channels'].items():
            indices = channel[2]
            assert np.all(indices[1:] > indices[:-1])
    published = {}
    assert CompactSequenceDecoder(sequence_manifest, working_cache=working)(range(9),
        on_frame=lambda sequence, frame: published.setdefault(sequence, frame)) == {}
    assert set(published) == set(range(9))


def test_compact_requires_valid_working_cache(sequence_manifest, tmp_path):
    with pytest.raises(ValueError, match='working cache'):
        CompactSequenceDecoder(sequence_manifest)((1,))
    with pytest.raises(ValueError, match='Missing valid'):
        CompactSequenceDecoder(sequence_manifest, working_cache=tmp_path)((1,))
