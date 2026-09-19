from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from dense_dump_codec.native import NativeSequenceDecoder
from dense_dump_codec.streaming import StreamingFrameService
from dense_dump_codec.working_set import transcode_working_archive
from test_native_runtime import frame_index, sequence_manifest


def test_worker_exception_joins_other_tasks(sequence_manifest, monkeypatch):
    decoder = NativeSequenceDecoder(sequence_manifest, workers=3)
    original = decoder.reconstruction.decode
    active = set()
    lock = threading.Lock()

    def decode(archive, metadata, dataset, frame_index, **kwargs):
        thread = threading.get_ident()
        with lock:
            active.add(thread)
        try:
            time.sleep(0.003)
            if dataset == "prims.uvec":
                raise ValueError("injected worker failure")
            return original(archive, metadata, dataset, frame_index, **kwargs)
        finally:
            with lock:
                active.remove(thread)

    monkeypatch.setattr(decoder.reconstruction, "decode", decode)
    with pytest.raises(ValueError, match="injected worker failure"):
        decoder((1, 2, 3, 5, 6, 7))
    assert not active


def test_changed_anchor_invalidates_cache_without_mutating_published_arrays(sequence_manifest):
    decoder = NativeSequenceDecoder(sequence_manifest)
    first = decoder((0,))[0]
    path = Path(first["source_phdf"])
    before = first["datasets"]["prims.rho"].tobytes()
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    second = decoder((0,))[0]
    assert first["datasets"]["prims.rho"] is not second["datasets"]["prims.rho"]
    assert first["datasets"]["prims.rho"].tobytes() == before
    assert second["datasets"]["prims.rho"].tobytes() == before


def test_close_wakes_readers_and_stops_producer():
    entered, release = threading.Event(), threading.Event()

    def decode(sequences, *, on_frame):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("test timed out")
        for sequence in sequences:
            on_frame(sequence, {"sequence": sequence})

    service = StreamingFrameService(frame_index(), "unused", decode,
                                    maximum_cache_files=2, prefetch_files=2)
    failures = []

    def read():
        try:
            service.stage_sequence(0)
        except RuntimeError as error:
            failures.append(str(error))

    reader = threading.Thread(target=read)
    closer = threading.Thread(target=service.close)
    try:
        reader.start()
        assert entered.wait(2)
        closer.start()
        reader.join(1)
        assert not reader.is_alive()
        assert failures == ["DDC streaming service has stopped"]
    finally:
        release.set()
        closer.join(2)
        service.close()
        reader.join(2)
    assert not service._thread.is_alive()


def test_warm_pins_complete_range():
    def decode(sequences, *, on_frame):
        for sequence in sequences:
            on_frame(sequence, {"sequence": sequence})

    service = StreamingFrameService(frame_index(), "unused", decode,
                                    maximum_cache_files=4, prefetch_files=3)
    try:
        result = service.warm_sequences(2, 4)
        assert result["sequence_count"] == 4
        assert set(range(2, 6)).issubset(service._cache)
        assert service.stage_sequence(11)["sequence"] == 11
    finally:
        service.close()


def test_working_crc_failure_is_not_published(sequence_manifest, tmp_path, monkeypatch):
    manifest = json.loads(sequence_manifest.read_text())
    source = Path(next(iter(manifest["codec_schemes"].values()))["archive_paths"][0])
    import dense_dump_codec.working_set as module

    class BadArchive:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def testzip(self):
            return "injected_bad_member"

    monkeypatch.setattr(module, "open_archive", lambda *args, **kwargs: BadArchive())
    output = tmp_path / "bad.ddc"
    with pytest.raises(ValueError, match="CRC validation"):
        transcode_working_archive(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".ddc_working_*"))
