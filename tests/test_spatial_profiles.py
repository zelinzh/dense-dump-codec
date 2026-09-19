import io
import json
import zipfile

import numpy as np
import pytest

from dense_dump_codec import pack_int4
from dense_dump_codec.reconstruction import Reconstruction
from dense_dump_codec.spatial_cache import SpatialWorkingArchive, prepare_spatial_cache
from scripts import decode_dump_codec as codec
from test_native_runtime import array_archive


@pytest.mark.parametrize("bits", [4, 5, 7, 8, 16])
@pytest.mark.parametrize("compact", [False, True])
def test_spatial_profiles_match_independent_reference(tmp_path, bits, compact):
    rng = np.random.default_rng(79)
    shape, tiles = (2, 3, 4, 8, 64), (2, 4, 16)
    start, end = (rng.normal(size=shape).astype("f4") for _ in range(2))
    metadata = {"bits": bits, "middle_taus": [0.375]}
    members = {}
    datasets = ("prims.rho", "prims.u", "prims.uvec", "prims.B")
    if compact:
        metadata["dataset_tile_shapes"] = {dataset: tiles for dataset in datasets}
    for dataset in datasets:
        codes = rng.integers(-7, 8, shape, dtype="i1" if bits < 16 else "i2")
        scale = rng.uniform(0.001, 0.01, (2, 3, 2, 2, 4) if compact else (2, 3, 1, 1, 1)).astype("f4")
        arrays = dict(scale=scale, exception_indices=np.array([0, 15, 16, 31, 32, 63, 64, 127]),
                      exception_values=rng.normal(size=8).astype("f4"))
        if bits == 4:
            arrays.update(q4_packed=pack_int4(codes), q4_shape=np.array(shape))
        else:
            arrays[f"q{bits}"] = codes
        members.update({codec.archive_member_name(dataset, 1, name): values for name, values in arrays.items()})
    original = array_archive(members)
    source = tmp_path / "source.ddc"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        for name, values in members.items():
            stream = io.BytesIO()
            np.save(stream, values)
            archive.writestr(name, stream.getvalue())
    record = prepare_spatial_cache(source, tmp_path / "cache", codec=codec, slab_cells=16, workers=2)
    with SpatialWorkingArchive(record["output"]) as archive:
        for dataset in datasets:
            expected = Reconstruction(codec, "reference").decode(
                original, metadata, dataset, 1, start=start, end=end,
                start_transformed=start, end_transformed=end)
            for stop in (32, 64):
                actual = Reconstruction(codec, radial_stop=stop).decode(
                    archive, metadata, dataset, 1, start=start[..., :stop], end=end[..., :stop],
                    start_transformed=start[..., :stop], end_transformed=end[..., :stop])
                assert actual.tobytes() == expected[..., :stop].tobytes()
