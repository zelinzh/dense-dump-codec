import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import watch_dense_codec
from watch_dense_codec import (
    expected_last_sequence,
    gop_endpoint_complete,
    gop_ready,
    next_gop,
    sha256_readable_file,
    try_encode_gop,
)


def watcher_args() -> SimpleNamespace:
    return SimpleNamespace(
        expected_frame_count=21,
        keyframe_stride=10,
        start_sequence=0,
        allow_short_final_gop=False,
    )


def test_next_gop_resumes_from_last_completed_endpoint() -> None:
    args = watcher_args()
    state = {"gops": []}

    assert expected_last_sequence(args) == 20
    assert next_gop(state, args) == (0, 10)
    state["gops"].append({"end_sequence": 10})
    assert next_gop(state, args) == (10, 20)
    state["gops"].append({"end_sequence": 20})
    assert next_gop(state, args) is None


def test_gop_ready_requires_readable_time_metadata(tmp_path: Path) -> None:
    files = {}
    for sequence in range(3):
        path = tmp_path / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = float(sequence)
        files[sequence] = path

    assert gop_ready(files, 0, 2)
    files[1].unlink()
    assert not gop_ready(files, 0, 2)


def test_gop_ready_waits_for_writer_to_make_source_readable(tmp_path: Path) -> None:
    files = {}
    for sequence in range(3):
        path = tmp_path / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = float(sequence)
        files[sequence] = path
    files[1].chmod(0o000)

    assert not gop_ready(files, 0, 2)
    assert not files[1].stat().st_mode & 0o400


def test_gop_ready_retries_transient_hdf5_runtime_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenFile:
        def __enter__(self) -> "BrokenFile":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __contains__(self, name: str) -> bool:
            raise RuntimeError("incorrect metadata checksum after all read attempts")

    class BrokenH5py:
        @staticmethod
        def File(path: Path, mode: str) -> BrokenFile:
            return BrokenFile()

    monkeypatch.setattr(watch_dense_codec, "require_h5py", lambda: BrokenH5py)
    files = {
        sequence: tmp_path / f"tiny.out0.{sequence:05d}.phdf"
        for sequence in range(3)
    }
    assert not gop_ready(files, 0, 2)


def test_repair_pending_source_permissions_is_scoped_to_completed_gop(
    tmp_path: Path,
) -> None:
    segment_dir = tmp_path / "segment"
    segment_dir.mkdir()
    files = {}
    for sequence in range(5):
        path = segment_dir / f"tiny.out0.{sequence:05d}.phdf"
        path.write_bytes(b"frame")
        path.chmod(0o000)
        files[sequence] = path
    args = SimpleNamespace(
        segment_dir=segment_dir,
        repair_source_permissions_after_final_checkpoint=True,
    )

    assert watch_dense_codec.repair_pending_source_permissions(args, files, 0, 2, 4) == []
    (segment_dir / "tiny.out1.final.rhdf").write_bytes(b"checkpoint")

    repaired = watch_dense_codec.repair_pending_source_permissions(args, files, 0, 2, 4)

    assert set(repaired) == {str(files[sequence]) for sequence in range(4)}
    assert all(files[sequence].stat().st_mode & 0o400 for sequence in range(4))
    assert files[4].stat().st_mode & 0o777 == 0
    assert all(files[sequence].stat().st_mode & 0o077 == 0 for sequence in range(4))


def test_try_encode_gop_retries_permission_error_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"gops": [], "frames": {}}

    def fail_after_mutation(candidate, args, files, start, end):
        candidate["gops"].append({"end_sequence": end})
        raise PermissionError("source frame is temporarily unreadable")

    monkeypatch.setattr(watch_dense_codec, "encode_gop", fail_after_mutation)

    result, encoded = try_encode_gop(state, SimpleNamespace(), {}, 0, 25)

    assert not encoded
    assert result is state
    assert state == {"gops": [], "frames": {}}


def test_streaming_gop_requires_following_frame_or_stable_final(tmp_path: Path) -> None:
    files = {}
    for sequence in range(4):
        path = tmp_path / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = float(sequence)
        files[sequence] = path

    ready, signature = gop_endpoint_complete(files, 2, 4, None)
    assert ready is True
    assert signature is None

    files[3].chmod(0o000)
    ready, signature = gop_endpoint_complete(files, 2, 4, None)
    assert ready is False
    files[3].chmod(0o600)

    ready, signature = gop_endpoint_complete(files, 3, 3, None)
    assert ready is False
    ready, signature = gop_endpoint_complete(files, 3, 3, signature)
    assert ready is True


def test_retain_gop_files_hardlinks_baseline_and_truth(tmp_path: Path) -> None:
    files = {}
    state = {"frames": {}, "retained_raw_files": {}}
    for sequence in range(3):
        path = tmp_path / "segment" / f"tiny.out0.{sequence:05d}.phdf"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"frame-{sequence}".encode())
        files[sequence] = path
        state["frames"][str(sequence)] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "time": float(sequence),
        }
    args = SimpleNamespace(
        retain_every=2,
        retain_dir=tmp_path / "raw_dt0p5",
        truth_dir=tmp_path / "truth",
        truth_window_json=[
            json.dumps({"window_id": "window", "start": 0.5, "end": 1.5})
        ],
    )

    watch_dense_codec.retain_gop_files(state, args, files, 0, 2)
    watch_dense_codec.retain_gop_files(state, args, files, 0, 2)

    baseline = sorted((tmp_path / "raw_dt0p5").glob("*.phdf"))
    truth = sorted((tmp_path / "truth" / "window").glob("*.phdf"))
    assert [path.name for path in baseline] == [files[0].name, files[2].name]
    assert [path.name for path in truth] == [files[1].name]
    assert baseline[0].stat().st_ino == files[0].stat().st_ino
    assert truth[0].stat().st_ino == files[1].stat().st_ino
    assert len(state["retained_raw_files"]) == 3


def test_parse_truth_windows_rejects_inverted_interval() -> None:
    with pytest.raises(ValueError, match="end >= start"):
        watch_dense_codec.parse_truth_windows(
            [json.dumps({"window_id": "bad", "start": 2, "end": 1})]
        )


def test_repair_unreadable_codec_artifacts_restores_owner_read(tmp_path: Path) -> None:
    output_dir = tmp_path / "ddc"
    gop_dir = output_dir / "gops" / "00000_00025"
    gop_dir.mkdir(parents=True)
    summary = gop_dir / "summary.json"
    archive = gop_dir / "archive.ddc"
    summary.write_text('{"complete": true}\n', encoding="utf-8")
    archive.write_bytes(b"codec")
    summary.chmod(0o000)
    archive.chmod(0o000)

    repaired = watch_dense_codec.repair_unreadable_codec_artifacts(output_dir)

    assert set(repaired) == {str(archive), str(summary)}
    assert summary.stat().st_mode & 0o400
    assert archive.stat().st_mode & 0o400
    assert watch_dense_codec.read_codec_summary(summary) == {"complete": True}


def test_sha256_readable_file_retries_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "archive.ddc"
    path.write_bytes(b"codec")
    calls = []

    def transient_sha256(candidate: Path) -> str:
        calls.append(candidate)
        if len(calls) < 3:
            raise PermissionError("temporary NAS mode")
        return "digest"

    monkeypatch.setattr(watch_dense_codec, "sha256_file", transient_sha256)

    assert sha256_readable_file(path, attempts=3) == "digest"
    assert calls == [path, path, path]


def test_read_codec_summary_repairs_referenced_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive.ddc"
    archive.write_bytes(b"codec")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {"codec_schemes": {"q4": {"archive_path": str(archive)}}}
        ),
        encoding="utf-8",
    )
    archive.chmod(0o000)

    loaded = watch_dense_codec.read_codec_summary(summary)

    assert loaded["codec_schemes"]["q4"]["archive_path"] == str(archive)
    assert archive.stat().st_mode & 0o400


def test_atomic_json_forces_readable_mode_under_restrictive_umask(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    previous_umask = os.umask(0o777)
    try:
        watch_dense_codec.atomic_json(path, {"ok": True})
    finally:
        os.umask(previous_umask)

    assert path.stat().st_mode & 0o400
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_safe_configuration_extension_only_allows_longer_run_and_truth(tmp_path: Path) -> None:
    previous = {
        "expected_frame_count": 101,
        "keyframe_stride": 25,
        "retention": {
            "retain_every": 5,
            "truth_windows": [{"window_id": "truth", "start": 10.0, "end": 30.0}],
        },
    }
    extended = {
        "expected_frame_count": 151,
        "keyframe_stride": 25,
        "retention": {
            "retain_every": 5,
            "truth_windows": [{"window_id": "truth", "start": 10.0, "end": 50.0}],
        },
    }

    migration = watch_dense_codec.safe_configuration_extension(previous, extended)
    assert migration is not None
    assert migration["previous_expected_frame_count"] == 101
    assert migration["expected_frame_count"] == 151

    shortened = json.loads(json.dumps(extended))
    shortened["retention"]["truth_windows"][0]["end"] = 20.0
    assert watch_dense_codec.safe_configuration_extension(previous, shortened) is None

    changed_codec = json.loads(json.dumps(extended))
    changed_codec["keyframe_stride"] = 20
    assert watch_dense_codec.safe_configuration_extension(previous, changed_codec) is None


def test_repair_completed_artifacts_covers_all_campaign_roots(tmp_path: Path) -> None:
    roots = {
        "output_dir": tmp_path / "ddc",
        "segment_dir": tmp_path / "runs",
        "retain_dir": tmp_path / "raw",
        "truth_dir": tmp_path / "truth",
    }
    files = []
    for name, directory in roots.items():
        directory.mkdir()
        path = directory / f"{name}.dat"
        path.write_bytes(b"data")
        path.chmod(0o000)
        files.append(path)

    repaired = watch_dense_codec.repair_completed_artifacts(SimpleNamespace(**roots))

    assert set(repaired) == {str(path) for path in files}
    assert all(path.stat().st_mode & 0o400 for path in files)
