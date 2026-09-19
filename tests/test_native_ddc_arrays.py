from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.decode_dense_sequence import (
    decode_sequence_frame,
    decode_sequence_frames_to_arrays,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("prims.rho", "prims.u", "prims.uvec", "prims.B")


def _write_kharma_frame(path: Path, sequence: int) -> dict[str, np.ndarray]:
    scalar = np.asarray([1.0 + sequence, 2.0 + 0.5 * sequence], dtype=np.float32)
    uvec = np.arange(6, dtype=np.float32).reshape(1, 3, 1, 1, 2) + sequence * 0.125
    bvec = np.arange(6, dtype=np.float32).reshape(1, 3, 1, 1, 2) - sequence * 0.25
    arrays = {
        "prims.rho": scalar.reshape(1, 1, 1, 1, 2),
        "prims.u": (scalar + 3.0).reshape(1, 1, 1, 1, 2),
        "prims.uvec": uvec,
        "prims.B": bvec,
    }
    par_text = """<parthenon/mesh>
nx1=2
nx2=1
nx3=1
x1min=0
x1max=1
x2min=0
x2max=1
x3min=0
x3max=6.283185307179586
<coordinates>
a=0.5
r_in=1
r_out=20
hslope=0.3
transform=fmks
mks_smooth=0.5
poly_xt=0.82
poly_alpha=14
<GRMHD>
gamma=1.3333333333333333
"""
    with h5py.File(path, "w") as handle:
        info = handle.create_group("Info")
        info.attrs["Time"] = 10.0 + 0.1 * sequence
        info.attrs["NumMeshBlocks"] = 1
        info.attrs["MeshBlockSize"] = np.asarray([2, 1, 1], dtype=np.int64)
        handle.create_group("Input").attrs["File"] = par_text
        handle.create_group("Blocks")["loc.lx123"] = np.asarray([[0, 0, 0]], dtype=np.int64)
        for name, array in arrays.items():
            handle[name] = array
    return arrays


def test_native_array_decode_matches_phdf_decode_without_materialization(
    tmp_path: Path,
) -> None:
    segment = tmp_path / "segment"
    codec = tmp_path / "codec"
    segment.mkdir()
    for sequence in range(3):
        _write_kharma_frame(segment / f"tiny.out0.{sequence:05d}.phdf", sequence)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "encode_dense_sequence.py"),
            "--segment-dir",
            str(segment),
            "--output-dir",
            str(codec),
            "--keyframe-stride",
            "2",
            "--datasets",
            ",".join(DATASETS),
            "--bits",
            "8",
            "--archive-backend",
            "channel-bzip2-adaptive",
            "--keyframe-backend",
            "raw",
            "--delete-middle-frames",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = codec / "sequence_manifest.json"
    native = decode_sequence_frames_to_arrays(manifest, (1,))[1]
    reconstructed = tmp_path / "reconstructed.phdf"
    decode_sequence_frame(manifest, 1, reconstructed)

    assert native["sequence"] == 1
    assert native["time"] == 10.1
    assert native["meshblock_size"] == (2, 1, 1)
    assert native["block_order"].dtype == np.dtype("<i8")
    with h5py.File(reconstructed, "r") as handle:
        for dataset in DATASETS:
            array = native["datasets"][dataset]
            assert array.flags.c_contiguous
            assert array.dtype == np.dtype("<f4")
            np.testing.assert_array_equal(array, handle[dataset][...])
    assert not list(tmp_path.glob("ddc_frame_*.phdf"))
