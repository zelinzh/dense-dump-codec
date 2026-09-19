import threading

import numpy as np
from pathlib import Path

from dense_dump_codec.kpolaris_service import (
    DDCFrameService,
    DDCUnixServer,
    request_ddc_frame,
    request_ddc_stage,
    shutdown_ddc_server,
    warm_ddc_sequences,
)
from dense_dump_codec.sequence import DenseSequenceIndex, FrameRecord


def _index(count: int = 7) -> DenseSequenceIndex:
    return DenseSequenceIndex(
        FrameRecord(sequence, 10.0 + 0.1 * sequence, f"frame{sequence}")
        for sequence in range(count)
    )


def test_service_batch_decodes_and_prunes_atomically(tmp_path: Path) -> None:
    calls: list[list[int]] = []

    def decode(outputs: dict[int, Path]) -> dict[int, dict]:
        calls.append(sorted(outputs))
        for sequence, output in outputs.items():
            assert output.name.endswith(".partial")
            output.write_text(str(sequence))
        return {sequence: {"sequence": sequence} for sequence in outputs}

    service = DDCFrameService(
        _index(), tmp_path, decode, maximum_cache_files=3
    )
    first = service.materialize_sequence(1)
    assert calls == [[1, 2, 3]]
    assert first["batch_decoded_sequences"] == [1, 2, 3]
    assert (tmp_path / "ddc_frame_00001.phdf").read_text() == "1"
    assert not list(tmp_path.glob("*.partial"))

    hit = service.materialize_sequence(2)
    assert hit["cache_hit"]
    assert calls == [[1, 2, 3], [4]]

    service.materialize_sequence(4)
    assert calls[-1] == [5, 6]
    assert sorted(path.name for path in tmp_path.glob("ddc_frame_*.phdf")) == [
        "ddc_frame_00004.phdf",
        "ddc_frame_00005.phdf",
        "ddc_frame_00006.phdf",
    ]
    stats = service.statistics()
    assert stats["requests"] == 3
    assert stats["batch_decode_calls"] == 3
    assert stats["decoded_frames"] == 6


def test_unix_service_materializes_and_shuts_down(tmp_path: Path) -> None:
    def decode(outputs: dict[int, Path]) -> dict[int, dict]:
        for sequence, output in outputs.items():
            output.write_text(str(sequence))
        return {sequence: {} for sequence in outputs}

    service = DDCFrameService(
        _index(4), tmp_path / "cache", decode, maximum_cache_files=3
    )
    socket_path = tmp_path / "ddc.sock"
    server = DDCUnixServer(socket_path, service)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        path = request_ddc_frame(socket_path, 1, timeout=2.0)
        assert path.read_text() == "1"
        shutdown_ddc_server(socket_path, timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()


def _native_frame(sequence: int) -> dict:
    scalar = np.asarray([sequence + 0.25, sequence + 0.75], dtype=np.float32)
    vector = np.arange(6, dtype=np.float32) + np.float32(sequence)
    return {
        "sequence": sequence,
        "time": 10.0 + 0.1 * sequence,
        "num_meshblocks": 1,
        "meshblock_size": (2, 1, 1),
        "par_text": "<parthenon/mesh>\nnx1=2\nnx2=1\nnx3=1\n",
        "block_order": np.asarray([[0, 0, 0]], dtype=np.int64),
        "datasets": {
            "prims.rho": scalar,
            "prims.u": scalar + 1.0,
            "prims.uvec": vector,
            "prims.B": vector + 2.0,
        },
    }


def test_native_stage_batches_arrays_in_memory(tmp_path: Path) -> None:
    calls: list[tuple[int, ...]] = []

    def decode(outputs: dict[int, Path]) -> dict[int, dict]:
        raise AssertionError("native staging must not materialize PHDF")

    def decode_staged(sequences: tuple[int, ...]) -> dict[int, dict]:
        calls.append(sequences)
        return {sequence: _native_frame(sequence) for sequence in sequences}

    service = DDCFrameService(
        _index(5),
        tmp_path,
        decode,
        decode_staged_frames=decode_staged,
        maximum_cache_files=3,
    )
    assert service.stage_sequence(1)["sequence"] == 1
    assert calls == [(1, 2, 3)]
    assert service.stage_sequence(2)["sequence"] == 2
    assert calls == [(1, 2, 3), (4,)]
    stats = service.statistics()
    assert stats["native_requests"] == 2
    assert stats["native_cache_hits"] == 1
    assert stats["native_decoded_frames"] == 4
    assert stats["native_cache_frames"] == 3
    assert not list(tmp_path.glob("*.phdf"))


def test_unix_service_streams_native_stage_without_files(tmp_path: Path) -> None:
    def decode(outputs: dict[int, Path]) -> dict[int, dict]:
        raise AssertionError("native staging must not materialize PHDF")

    service = DDCFrameService(
        _index(4),
        tmp_path / "cache",
        decode,
        decode_staged_frames=lambda sequences: {
            sequence: _native_frame(sequence) for sequence in sequences
        },
        maximum_cache_files=3,
    )
    socket_path = tmp_path / "ddc-native.sock"
    server = DDCUnixServer(socket_path, service)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        frame = request_ddc_stage(socket_path, 1, timeout=2.0)
        assert frame["sequence"] == 1
        assert frame["meshblock_size"] == (2, 1, 1)
        np.testing.assert_array_equal(
            frame["datasets"]["prims.rho"],
            _native_frame(1)["datasets"]["prims.rho"],
        )
        np.testing.assert_array_equal(
            frame["datasets"]["prims.B"],
            _native_frame(1)["datasets"]["prims.B"],
        )
        assert service.statistics()["native_bytes_sent"] > 0
        assert not list((tmp_path / "cache").glob("*.phdf"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_unix_service_warms_fixed_range_before_local_prefetch(tmp_path: Path) -> None:
    calls: list[tuple[int, ...]] = []

    def decode(outputs: dict[int, Path]) -> dict[int, dict]:
        raise AssertionError("native staging must not materialize PHDF")

    def decode_staged(sequences: tuple[int, ...]) -> dict[int, dict]:
        calls.append(sequences)
        return {sequence: _native_frame(sequence) for sequence in sequences}

    service = DDCFrameService(
        _index(7),
        tmp_path / "cache",
        decode,
        decode_staged_frames=decode_staged,
        maximum_cache_files=6,
        prefetch_files=2,
    )
    socket_path = tmp_path / "ddc-warm.sock"
    server = DDCUnixServer(socket_path, service)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        result = warm_ddc_sequences(socket_path, 1, 5, timeout=2.0)
        assert result == {
            "start_sequence": 1,
            "end_sequence": 5,
            "sequence_count": 5,
            "decoded_count": 5,
        }
        assert calls == [(1, 2, 3, 4, 5)]
        assert request_ddc_stage(socket_path, 3, timeout=2.0)["sequence"] == 3
        assert calls == [(1, 2, 3, 4, 5)]
        assert service.statistics()["prefetch_files"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_native_stage_batches_on_aligned_prefetch_boundaries(tmp_path: Path) -> None:
    calls: list[tuple[int, ...]] = []

    def decode(outputs: dict[int, Path]) -> dict[int, dict]:
        raise AssertionError("native staging must not materialize PHDF")

    def decode_staged(sequences: tuple[int, ...]) -> dict[int, dict]:
        calls.append(sequences)
        return {sequence: _native_frame(sequence) for sequence in sequences}

    service = DDCFrameService(
        _index(7),
        tmp_path,
        decode,
        decode_staged_frames=decode_staged,
        maximum_cache_files=4,
        prefetch_files=3,
    )
    assert service.stage_sequence(1)["sequence"] == 1
    assert calls == [(0, 1, 2)]
    assert service.stage_sequence(2)["sequence"] == 2
    assert calls == [(0, 1, 2)]
    assert service.stage_sequence(3)["sequence"] == 3
    assert calls == [(0, 1, 2), (3, 4, 5)]
    stats = service.statistics()
    assert stats["native_prefetch_mode"] == "aligned_batch"
    assert stats["native_cache_frames"] == 4


def test_prefetch_must_fit_bounded_cache(tmp_path: Path) -> None:
    try:
        DDCFrameService(
            _index(),
            tmp_path,
            lambda outputs: {},
            maximum_cache_files=3,
            prefetch_files=4,
        )
    except ValueError as error:
        assert "prefetch_files must not exceed maximum_cache_files" in str(error)
    else:
        raise AssertionError("oversized prefetch was accepted")
