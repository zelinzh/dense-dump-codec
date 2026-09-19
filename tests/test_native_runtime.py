from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dense_dump_codec import pack_int4
from dense_dump_codec.archive import open_archive
from dense_dump_codec.native import NativeSequenceDecoder, native_capabilities
from dense_dump_codec.reconstruction import Reconstruction, read_array_member, scaled_residual, tile_view
from dense_dump_codec.streaming import StreamingFrameService
from dense_dump_codec.working_set import transcode_working_archive
from scripts import decode_dump_codec as codec
from scripts.decode_dense_sequence import decode_sequence_frames_to_arrays
from test_native_ddc_arrays import DATASETS, _write_kharma_frame


ROOT = Path(__file__).resolve().parents[1]


def array_archive(arrays):
    members = {}
    for name, values in arrays.items():
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        members[name] = stream.getvalue()
    return SimpleNamespace(read=lambda name: members[name], namelist=lambda: list(members),
                           open=lambda name, mode: io.BytesIO(members[name]), members=members)


@pytest.mark.parametrize("bits", [4, 5, 7, 8, 16])
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("scale_dtype", [np.float32, np.float64])
def test_reconstruction_is_bitwise_and_independently_owned(bits, dataset, scale_dtype):
    rng = np.random.default_rng(415)
    shape = (2, 3, 5)
    start, end = (rng.uniform(-5, 5, shape).astype("f4") for _ in range(2))
    codes = rng.integers(-7, 8, shape, dtype=np.int8 if bits <= 8 else np.int16)
    scale = np.array([0.001, 0.007], dtype=scale_dtype).reshape(2, 1, 1)
    arrays = {"scale": scale, "exception_indices": np.array([0, 7, 29], dtype="u4"),
              "exception_values": np.array([-0.3, 0, 0.4], dtype="f4")}
    if bits == 4:
        arrays.update(q4_packed=pack_int4(codes), q4_shape=np.array(shape, dtype="i8"))
    else:
        arrays[f"q{bits}"] = codes
    archive = array_archive({codec.archive_member_name(dataset, 1, name): values
                             for name, values in arrays.items()})
    metadata = {"bits": bits, "middle_taus": [0.3125]}
    kwargs = dict(start=start, end=end, start_transformed=start, end_transformed=end)
    expected = codec.decode_dataset(archive, metadata, dataset, 1, **kwargs)
    decoder = Reconstruction(codec)
    actual = decoder.decode(archive, metadata, dataset, 1, **kwargs)
    assert actual.tobytes() == expected.tobytes()
    later = decoder.decode(archive, metadata, dataset, 1, **kwargs)
    assert not np.shares_memory(actual, later)
    assert actual.tobytes() == expected.tobytes()


def test_member_views_own_bytes_and_reject_malformed_payloads():
    for values in (np.asfortranarray(np.arange(24, dtype=">f4").reshape(2, 3, 4)),
                   np.array(4, dtype="i8"), np.empty((0, 3), dtype="f4")):
        archive = array_archive({"values": values})
        actual = read_array_member(archive, "values")
        archive.members.clear()
        assert actual.tobytes() == values.tobytes()
        assert actual.dtype == values.dtype
        assert not actual.flags.writeable
    archive = array_archive({"values": np.arange(10, dtype="f4")})
    archive.members["values"] = archive.members["values"][:-1]
    with pytest.raises(ValueError, match="truncated"):
        read_array_member(archive, "values")
    stream = io.BytesIO()
    np.save(stream, np.array([object()], dtype=object))
    archive.members["values"] = stream.getvalue()
    with pytest.raises(ValueError, match="Object"):
        read_array_member(archive, "values")


def test_compact_scales_match_experimental_tile_formula():
    rng = np.random.default_rng(210)
    shape, tiles = (2, 8, 16, 32), (4, 8, 16)
    codes = rng.integers(-127, 128, shape, dtype=np.int8)
    scale = rng.uniform(0.001, 0.01, (2, 2, 2, 2)).astype("f4")
    tiled = tile_view(codes, tiles).astype(np.float32) * scale[..., None, None, None]
    expected = tiled.transpose(0, 1, 4, 2, 5, 3, 6).reshape(shape)
    np.testing.assert_array_equal(scaled_residual(codes, scale, tile_shape=tiles), expected)
    with pytest.raises(ValueError, match="divide"):
        scaled_residual(codes, scale, tile_shape=(3, 8, 16))
    with pytest.raises(ValueError, match="scales"):
        scaled_residual(codes, scale[..., 0], tile_shape=tiles)
    arrays = {"q8": codes, "scale": scale, "exception_indices": np.array([7], dtype="u4"),
              "exception_values": np.array([0.032], dtype="f4")}
    archive = array_archive({codec.archive_member_name("prims.u", 1, name): values
                             for name, values in arrays.items()})
    start = rng.normal(size=shape).astype("f4")
    end = rng.normal(size=shape).astype("f4")
    metadata = {"bits": 8, "middle_taus": [0.3125],
                "dataset_tile_shapes": {"prims.u": tiles}}
    expected.reshape(-1)[7] = np.float32(0.032)
    expected = np.exp((0.6875 * start + 0.3125 * end).astype("f4") + expected, dtype="f4")
    for mode in ("reference", "numpy"):
        actual = Reconstruction(codec, mode).decode(
            archive, metadata, "prims.u", 1, start=start, end=end,
            start_transformed=start, end_transformed=end)
        assert actual.tobytes() == expected.tobytes()


@pytest.fixture
def sequence_manifest(tmp_path):
    segment, output = tmp_path / "segment", tmp_path / "codec"
    segment.mkdir()
    for sequence in range(9):
        _write_kharma_frame(segment / f"tiny.out0.{sequence:05d}.phdf", sequence)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/encode_dense_sequence.py"),
        "--segment-dir", str(segment), "--output-dir", str(output),
        "--keyframe-stride", "4", "--datasets", ",".join(DATASETS), "--bits", "8",
        "--archive-backend", "channel-bzip2-adaptive", "--keyframe-backend", "raw",
        "--delete-middle-frames",
    ], check=True, capture_output=True, text=True)
    return output / "sequence_manifest.json"


def test_native_parallel_callback_matches_legacy(sequence_manifest):
    expected = decode_sequence_frames_to_arrays(sequence_manifest, tuple(range(9)))
    decoder = NativeSequenceDecoder(sequence_manifest, workers=3, maximum_batch_frames=9)
    actual = {}
    assert decoder((8, 7, 0, 1, 2, 3, 4, 5, 6, 1), on_frame=lambda seq, frame:
                   actual.update({seq: frame})) == {}
    for sequence, frame in expected.items():
        for name in DATASETS:
            assert actual[sequence]["datasets"][name].tobytes() == frame["datasets"][name].tobytes()
        assert actual[sequence]["time"] == frame["time"]
        np.testing.assert_array_equal(actual[sequence]["block_order"], frame["block_order"])
    again = decoder((0,))[0]
    assert actual[0]["datasets"]["prims.rho"] is again["datasets"]["prims.rho"]
    assert not again["datasets"]["prims.rho"].flags.writeable
    assert not again["block_order"].flags.writeable
    assert len(decoder._anchors) <= 4
    with pytest.raises(KeyError):
        decoder((99,))
    with pytest.raises(ValueError, match="batch"):
        NativeSequenceDecoder(sequence_manifest, maximum_batch_frames=2)((0, 1, 2))
    assert native_capabilities()["api_version"] == 1


def test_lossless_working_backend_preserves_all_members(sequence_manifest, tmp_path):
    manifest = json.loads(sequence_manifest.read_text())
    source = Path(next(iter(manifest["codec_schemes"].values()))["archive_paths"][0])
    output = tmp_path / "working.ddc"
    result = transcode_working_archive(source, output, workers=2)
    assert result["logical_crcs_verified"] and not result["re_quantized"]
    with open_archive(source) as original, open_archive(output) as working:
        assert working.testzip() is None
        assert original.namelist() == working.namelist()
        for name in original.namelist():
            assert original.read(name) == working.read(name)
    with pytest.raises(FileExistsError):
        transcode_working_archive(source, output)
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    kwargs = dict(start=np.ones((1, 1, 1, 1, 2), dtype="f4"),
                  end=np.ones((1, 1, 1, 1, 2), dtype="f4"))
    with open_archive(source) as original, open_archive(output) as working:
        first = codec.decode_dataset(original, metadata, "prims.u", 1, **kwargs)
        second = codec.decode_dataset(working, metadata, "prims.u", 1, **kwargs)
        assert first.tobytes() == second.tobytes()


def frame_index():
    return SimpleNamespace(frames=[SimpleNamespace(sequence=sequence) for sequence in range(12)])


def test_stream_publishes_before_batch_completion_and_can_seek():
    published, release = threading.Event(), threading.Event()

    def decode(sequences, *, on_frame):
        on_frame(sequences[0], {"sequence": sequences[0]})
        published.set()
        if not release.wait(5):
            raise RuntimeError("test timed out")
        for sequence in sequences[1:]:
            on_frame(sequence, {"sequence": sequence})

    service = StreamingFrameService(frame_index(), "unused", decode,
                                    maximum_cache_files=4, prefetch_files=4)
    results = []
    reader = threading.Thread(target=lambda: results.append(service.stage_sequence(2)))
    try:
        reader.start()
        assert published.wait(2)
        reader.join(1)
        assert not reader.is_alive()
        assert results == [{"sequence": 2}]
        release.set()
        for sequence in range(3, 12):
            assert service.stage_sequence(sequence)["sequence"] == sequence
        assert service.stage_sequence(0)["sequence"] == 0
        assert service.statistics()["native_peak_cache_frames"] <= 4
    finally:
        release.set()
        service.close()
        reader.join(2)


@pytest.mark.parametrize("failure", ["raise", "missing", "duplicate", "mismatch"])
def test_stream_errors_wake_waiters(failure):
    def decode(sequences, *, on_frame):
        if failure == "raise":
            raise ValueError("injected failure")
        if failure == "duplicate":
            on_frame(sequences[-1], {"sequence": sequences[-1]})
            on_frame(sequences[-1], {"sequence": sequences[-1]})
        if failure == "mismatch":
            on_frame(sequences[0], {"sequence": -1})

    service = StreamingFrameService(frame_index(), "unused", decode,
                                    maximum_cache_files=2, prefetch_files=2)
    try:
        with pytest.raises(RuntimeError, match="streaming decode failed"):
            service.stage_sequence(0)
    finally:
        service.close()
    with pytest.raises(RuntimeError):
        service.stage_sequence(0)
