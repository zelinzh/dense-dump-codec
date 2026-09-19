import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from dense_dump_codec.cli import profile_arguments
from dense_dump_codec.native import NativeSequenceDecoder


ROOT = Path(__file__).resolve().parents[1]


def command(*arguments, check=True):
    return subprocess.run([sys.executable, "-m", "dense_dump_codec.cli", *map(str, arguments)],
                          check=check, capture_output=True, text=True, timeout=60)


@pytest.fixture
def encoded(tmp_path):
    raw = tmp_path / "raw"
    target = tmp_path / "encoded"
    subprocess.run([sys.executable, str(ROOT / "examples/make_synthetic_sequence.py"),
                    "--output-dir", str(raw), "--frames", "26"], check=True, capture_output=True)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw.glob("*.phdf")}
    command("encode", "--segment-dir", raw, "--output-dir", target)
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw.glob("*.phdf")}
    return raw, target / "sequence_manifest.json"


def test_default_profile_matches_documented_config():
    profile = json.loads((ROOT / "configs/local_u8.json").read_text())
    options = profile_arguments("local-u8")
    parsed = dict(zip(options[::2], options[1::2]))
    assert parsed["--dataset-tile-shapes"] == "prims.u=8x16x32"
    assert profile["dataset_tile_shapes"] == {"prims.u": [8, 16, 32]}
    bits = {name: int(value) for name, value in (item.split("=") for item in parsed["--dataset-bits"].split(","))}
    assert bits == profile["dataset_bits"]


def test_public_file_native_and_integrity_paths(encoded, tmp_path):
    raw, manifest = encoded
    payload = json.loads(manifest.read_text())
    assert next(iter(payload["codec_schemes"].values()))["dataset_tile_shapes"] == {"prims.u": [8, 16, 32]}
    verified = json.loads(command("verify", "--manifest", manifest, "--sha256").stdout)
    assert verified["archives_verified"] == 1
    assert len(verified["sha256"]) == 3
    output = tmp_path / "decoded.phdf"
    command("decode", "--manifest", manifest, "--sequence", "13", "--output-phdf", output)
    native = NativeSequenceDecoder(manifest, workers=2)([0, 13, 25])
    with h5py.File(output) as handle:
        for name, values in native[13]["datasets"].items():
            np.testing.assert_array_equal(values, handle[name][...])
    for sequence in (0, 25):
        with h5py.File(raw / f"synthetic.out0.{sequence:05d}.phdf") as handle:
            for name, values in native[sequence]["datasets"].items():
                np.testing.assert_array_equal(values, handle[name][...])
    repeated = command("encode", "--segment-dir", raw, "--output-dir", manifest.parent, check=False)
    assert repeated.returncode != 0 and "empty output directory" in repeated.stderr
    first = payload["integrity"]["archives"][0]
    first["sha256"] = "0" * 64
    damaged = tmp_path / "bad-hash.json"
    damaged.write_text(json.dumps(payload))
    assert command("verify", "--manifest", damaged, "--sha256", check=False).returncode != 0


def test_streaming_local_u8_and_resume(encoded, tmp_path):
    raw, _ = encoded
    target = tmp_path / "stream"
    arguments = ("watch", "--segment-dir", raw, "--output-dir", target,
                 "--expected-frame-count", "26", "--poll-seconds", "0.01")
    command(*arguments)
    command(*arguments)
    manifest = json.loads((target / "sequence_manifest.json").read_text())
    assert manifest["complete"] is True
    assert next(iter(manifest["codec_schemes"].values()))["dataset_tile_shapes"] == {"prims.u": [8, 16, 32]}
    command("verify", "--manifest", target / "sequence_manifest.json", "--sha256")
