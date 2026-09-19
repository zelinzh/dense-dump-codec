import json
import os
import subprocess
import sys
from pathlib import Path

import h5py

from dense_dump_codec.raw_sequence import load_raw_sequence_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_phdf(path: Path, time: float) -> None:
    with h5py.File(path, "w") as handle:
        info = handle.create_group("Info")
        info.attrs["Time"] = time


def test_raw_index_filters_global_sequence_stride_and_reuses_cache(tmp_path: Path) -> None:
    for sequence in range(7):
        write_phdf(tmp_path / f"restart.out0.{sequence:05d}.phdf", 10.0 + 0.1 * sequence)
    cache = tmp_path / "index.json"

    index, payload, reused = load_raw_sequence_index(
        tmp_path,
        dense_sequence_stride=2,
        cache_path=cache,
    )
    assert not reused
    assert [frame.sequence for frame in index.frames] == [0, 2, 4, 6]
    assert payload["time_end_M"] == 10.6

    _, _, reused = load_raw_sequence_index(
        tmp_path,
        dense_sequence_stride=2,
        cache_path=cache,
    )
    assert reused

    os.utime(tmp_path / "restart.out0.00004.phdf", None)
    _, _, reused = load_raw_sequence_index(
        tmp_path,
        dense_sequence_stride=2,
        cache_path=cache,
    )
    assert not reused


def test_raw_kpolaris_dry_run_selects_covering_frames(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    for sequence in range(7):
        write_phdf(source / f"restart.out0.{sequence:05d}.phdf", 10.0 + 0.1 * sequence)
    binary = tmp_path / "kpolaris"
    binary.write_bytes(b"binary")
    parameters = tmp_path / "base.par"
    parameters.write_text("model=kharma\n")
    output = tmp_path / "image.h5"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_kpolaris_raw_window.py"),
            "--input-dir",
            str(source),
            "--dense-sequence-stride",
            "2",
            "--kpolaris-binary",
            str(binary),
            "--parameter-file",
            str(parameters),
            "--observation-time",
            "11",
            "--fluid-time-min",
            "10.15",
            "--fluid-time-max",
            "10.45",
            "--output",
            str(output),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(Path(f"{output}.raw-run.json").read_text())
    assert [row["sequence"] for row in report["selected_frames"]] == [0, 2, 4, 6]
    adapter = Path(f"{output}.raw.par").read_text()
    assert "restart.out0.00002.phdf" in adapter
    assert "restart.out0.00001.phdf" not in adapter
