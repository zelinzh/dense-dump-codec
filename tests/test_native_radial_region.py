from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dense_dump_codec import pack_int4
from dense_dump_codec.native import NativeSequenceDecoder
from dense_dump_codec.reconstruction import Reconstruction
from dense_dump_codec.region import RadialRegion, parameter_sections, replace_parameters
from scripts import decode_dump_codec as codec
from test_native_ddc_arrays import DATASETS, _write_kharma_frame
from test_native_runtime import array_archive


ROOT = Path(__file__).resolve().parents[1]


def native_metadata():
    return {"par_text": """<parthenon/mesh>
nx1=288
nx2=128
nx3=128
x1min=0.16993748338110734
x1max=6.9077552789821359
refinement=none
<parthenon/meshblock>
nx1=288
nx2=128
nx3=64
<coordinates>
transform=fmks
r_out=1000
""", "meshblock_size": (288, 128, 64), "num_meshblocks": 2,
            "block_order": np.array([[0, 0, 0], [0, 0, 1]])}


def test_roi_preserves_all_interior_coordinates_and_halo():
    native = native_metadata()
    region = RadialRegion.from_metadata(native, 100)
    assert region.retained_cells == 192
    assert region.describe()["array_fraction"] == 2 / 3
    modified = region.metadata(native)
    sections = parameter_sections(modified["par_text"])
    original = parameter_sections(native["par_text"])
    assert sections["coordinates"] == original["coordinates"]
    assert sections["parthenon/meshblock"]["nx1"] == "192"
    spacing = (float(sections["parthenon/mesh"]["x1max"]) - region.startx1) / 192
    assert spacing == region.dx1
    assert modified["block_order"] is native["block_order"]
    for radius in np.geomspace(1.2, 100, 10000):
        first = math.floor((math.log(radius) - region.startx1) / spacing - 0.5)
        assert first + 1 < region.retained_cells - region.halo_cells
    full = RadialRegion.from_metadata(native, 1000)
    assert full.retained_cells == 288


@pytest.mark.parametrize("change", ["amr", "radial_blocks", "missing_blocks", "coordinates"])
def test_unsupported_layouts_fail_closed(change):
    native = native_metadata()
    if change == "amr":
        native["par_text"] = native["par_text"].replace("refinement=none", "refinement=adaptive")
    elif change == "radial_blocks":
        native["meshblock_size"] = (144, 128, 64)
    elif change == "missing_blocks":
        native["block_order"] = np.array([[0, 0, 0], [0, 0, 0]])
    else:
        native["par_text"] = native["par_text"].replace("transform=fmks", "transform=cartesian")
    with pytest.raises(ValueError):
        RadialRegion.from_metadata(native, 100)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), 0.1])
def test_invalid_roi_rejected(radius):
    with pytest.raises(ValueError):
        RadialRegion.from_metadata(native_metadata(), radius)


@pytest.mark.parametrize("bits", [4, 5, 7, 8, 16])
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("compact", [False, True])
def test_roi_reconstruction_is_bitwise_subset(bits, dataset, compact):
    rng = np.random.default_rng(42)
    shape, tiles, stop = (2, 3, 4, 8, 64), (2, 4, 16), 32
    start, end = (rng.normal(size=shape).astype("f4") for _ in range(2))
    codes = rng.integers(-7, 8, shape, dtype="i1" if bits < 16 else "i2")
    scales = (rng.uniform(0.001, 0.01, (2, 3, 2, 2, 4)).astype("f4") if compact else
              rng.uniform(0.001, 0.01, (2, 3, 1, 1, 1)).astype("f4"))
    indices = np.array([0, 31, 32, 63, 64, 95, 127, math.prod(shape) - 1], dtype="u4")
    arrays = {"scale": scales, "exception_indices": indices,
              "exception_values": rng.uniform(-1, 1, len(indices)).astype("f4")}
    if bits == 4:
        arrays.update(q4_packed=pack_int4(codes), q4_shape=np.array(shape, dtype="i8"))
    else:
        arrays[f"q{bits}"] = codes
    archive = array_archive({codec.archive_member_name(dataset, 1, name): values
                             for name, values in arrays.items()})
    metadata = {"bits": bits, "middle_taus": [0.3125]}
    if compact:
        metadata["dataset_tile_shapes"] = {dataset: tiles}
    expected = Reconstruction(codec, "reference").decode(
        archive, metadata, dataset, 1, start=start, end=end,
        start_transformed=start, end_transformed=end)
    cropped = Reconstruction(codec, radial_stop=stop).decode(
        archive, metadata, dataset, 1, start=start[..., :stop], end=end[..., :stop],
        start_transformed=start[..., :stop], end_transformed=end[..., :stop])
    assert cropped.tobytes() == expected[..., :stop].tobytes()
    with pytest.raises(ValueError, match="numpy"):
        Reconstruction(codec, "reference", radial_stop=stop)


def test_native_roi_sequence_including_exact_anchors(tmp_path):
    source, output = tmp_path / "source", tmp_path / "encoded"
    source.mkdir()
    for sequence in range(5):
        path = source / f"tiny.out0.{sequence:05d}.phdf"
        _write_kharma_frame(path, sequence)
        with h5py.File(path, "r+") as handle:
            handle["Info"].attrs["MeshBlockSize"] = [64, 1, 1]
            handle["Input"].attrs["File"] = replace_parameters(
                handle["Input"].attrs["File"],
                {("parthenon/mesh", "nx1"): "64", ("parthenon/mesh", "x1max"): "6.4"})
            for dataset in DATASETS:
                values = np.tile(handle[dataset][...], (1, 1, 1, 1, 32))
                del handle[dataset]
                handle[dataset] = values
    subprocess.run([
        sys.executable, str(ROOT / "scripts/encode_dense_sequence.py"),
        "--segment-dir", str(source), "--output-dir", str(output),
        "--keyframe-stride", "4", "--datasets", ",".join(DATASETS), "--bits", "8",
        "--archive-backend", "channel-bzip2-adaptive", "--keyframe-backend", "raw",
    ], check=True, capture_output=True)
    manifest = output / "sequence_manifest.json"
    reference = NativeSequenceDecoder(manifest, workers=2)(range(5))
    decoder = NativeSequenceDecoder(manifest, radial_max=math.exp(3), workers=2)
    actual = decoder(range(5))
    assert decoder.region.retained_cells == 32
    for sequence in range(5):
        assert actual[sequence]["meshblock_size"] == (32, 1, 1)
        assert actual[sequence]["time"] == reference[sequence]["time"]
        for dataset in DATASETS:
            assert actual[sequence]["datasets"][dataset].tobytes() == (
                reference[sequence]["datasets"][dataset][..., :32].tobytes())
