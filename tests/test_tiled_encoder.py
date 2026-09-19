import sys

import numpy as np
import pytest

from dense_dump_codec import dequantize_residual, quantize_residual
from dense_dump_codec.tiled import dequantize_tiled, parse_dataset_tiles, quantize_tiled
from scripts import decode_dump_codec


@pytest.mark.parametrize("bits", [4, 7, 8, 16])
@pytest.mark.parametrize("vector", [False, True])
def test_tiles_match_independent_research_formula(bits, vector):
    random = np.random.default_rng(123)
    leading = (2, 3) if vector else (2,)
    residual = random.normal(size=(*leading, 8, 16, 32)).astype("f4")
    residual.reshape(-1)[[0, 997, -1]] = [0, -200, 900]
    offset = len(leading)
    forward = (*range(offset), offset, offset + 2, offset + 4,
               offset + 1, offset + 3, offset + 5)
    backward = (*range(offset), offset, offset + 3, offset + 1,
                offset + 4, offset + 2, offset + 5)
    arranged = residual.reshape(*leading, 2, 4, 2, 8, 2, 16).transpose(forward)
    reference = quantize_residual(arranged, bits, "block-channel", scale_percentile=99.9)
    expected = dequantize_residual(reference).transpose(backward).reshape(residual.shape)
    actual = quantize_tiled(residual, (4, 8, 16), bits)
    np.testing.assert_array_equal(dequantize_tiled(actual, (4, 8, 16)), expected)
    np.testing.assert_array_equal(actual.values, reference.values.transpose(backward).reshape(residual.shape))
    np.testing.assert_array_equal(actual.scale, reference.scale[..., 0, 0, 0])
    np.testing.assert_array_equal(actual.exception_values, residual.reshape(-1)[actual.exception_indices])


def test_zero_percentile_and_invalid_tiles():
    residual = np.zeros((1, 8, 16, 32), dtype="f4")
    residual[0, 0, 0, 0] = 1
    payload = quantize_tiled(residual, (8, 16, 32))
    np.testing.assert_array_equal(dequantize_tiled(payload, (8, 16, 32)), residual)
    with pytest.raises(ValueError, match="divide"):
        quantize_tiled(residual, (3, 16, 32))
    for value in ("prims.u=0x16x32", "unknown=8x16x32", "prims.u=8x16", "prims.u=8x16x32,prims.u=4x8x16"):
        with pytest.raises(ValueError):
            parse_dataset_tiles(value, ("prims.u",))


def test_file_decoder_recognizes_local_scale_metadata(tmp_path):
    import io
    import zipfile
    from dense_dump_codec import open_archive
    from scripts.prototype_dump_codec import archive_member_name

    residual = np.random.default_rng(10).normal(size=(1, 8, 16, 32)).astype("f4")
    payload = quantize_tiled(residual, (4, 8, 16))
    archive_path = tmp_path / "tiles.ddc"
    arrays = {"q8": payload.values, "scale": payload.scale,
              "exception_indices": payload.exception_indices, "exception_values": payload.exception_values}
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, values in arrays.items():
            stream = io.BytesIO()
            np.save(stream, values, allow_pickle=False)
            archive.writestr(archive_member_name("prims.u", 1, name), stream.getvalue())
    metadata = {"bits": 8, "middle_taus": [0.5], "dataset_tile_shapes": {"prims.u": [4, 8, 16]}}
    with open_archive(archive_path) as archive:
        result = decode_dump_codec.decode_dataset(
            archive, metadata, "prims.u", 1, start=np.ones_like(residual), end=np.ones_like(residual))
    np.testing.assert_array_equal(result, np.exp(dequantize_tiled(payload, (4, 8, 16)), dtype=np.float32))


def test_watcher_configuration_records_local_tiles(monkeypatch):
    from scripts import watch_dense_codec
    monkeypatch.setattr(sys, "argv", ["watch", "--segment-dir", "/data/raw", "--output-dir", "/data/ddc",
                                     "--expected-frame-count", "51", "--keyframe-stride", "25",
                                     "--dataset-tile-shapes", "prims.u=8x16x32"])
    arguments = watch_dense_codec.parse_args()
    configured = watch_dense_codec.configuration(arguments)
    assert configured["dataset_tile_shapes"] == {"prims.u": [8, 16, 32]}
    different = dict(configured, dataset_tile_shapes={"prims.u": [4, 8, 16]})
    assert watch_dense_codec.safe_configuration_extension(configured, different) is None


def test_sequence_rejects_uncovered_single_interval_tail():
    from scripts.encode_dense_sequence import build_gop_ranges
    with pytest.raises(ValueError, match="no intermediate frame"):
        build_gop_ranges(27, 25, True)
