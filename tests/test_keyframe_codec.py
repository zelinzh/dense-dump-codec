from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dense_dump_codec import (
    KEYFRAME_BZIP2_SHUFFLE_FORMAT,
    KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT,
    KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT,
    KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT,
    compress_keyframe,
    decompress_keyframe,
    validate_keyframe_archive,
)
from scripts.decode_dense_sequence import (
    archive_sequence_bounds,
    decode_sequence_frame,
    decode_sequence_frames,
    locate_archive,
    materialize_keyframes,
)
from scripts.repack_sequence_keyframes import repack_manifest


def write_phdf(path: Path) -> None:
    random = np.random.default_rng(42)
    with h5py.File(path, "w") as handle:
        handle.attrs["version"] = "test"
        info = handle.create_group("Info")
        info.attrs["time"] = 12.5
        handle.create_dataset("Levels", data=np.array([0, 1], dtype=np.int64))
        density = handle.create_dataset(
            "prims.rho",
            data=random.normal(size=(2, 4, 8, 16)).astype(np.float32),
            chunks=(1, 4, 8, 16),
            compression="gzip",
            compression_opts=5,
        )
        density.attrs["units"] = "code"
        handle.create_dataset(
            "prims.uvec",
            data=random.normal(size=(2, 3, 4, 8, 16)).astype(np.float32),
            chunks=(1, 1, 4, 8, 16),
            compression="gzip",
            compression_opts=5,
        )


def assert_hdf5_equal(expected_path: Path, actual_path: Path) -> None:
    with h5py.File(expected_path, "r") as expected, h5py.File(actual_path, "r") as actual:
        assert dict(actual.attrs) == dict(expected.attrs)
        expected_names: list[str] = []
        actual_names: list[str] = []
        expected.visit(expected_names.append)
        actual.visit(actual_names.append)
        assert actual_names == expected_names
        for name in expected_names:
            expected_object = expected[name]
            actual_object = actual[name]
            assert dict(actual_object.attrs) == dict(expected_object.attrs)
            if isinstance(expected_object, h5py.Dataset):
                np.testing.assert_array_equal(actual_object[...], expected_object[...])
                assert actual_object.dtype == expected_object.dtype
                assert actual_object.shape == expected_object.shape
                assert actual_object.chunks == expected_object.chunks
                assert actual_object.compression == expected_object.compression
                assert actual_object.compression_opts == expected_object.compression_opts


def test_archive_sequence_bounds_from_gop_name() -> None:
    assert archive_sequence_bounds(Path("gop_00025_00050_linear.ddc")) == (25, 50)
    assert archive_sequence_bounds(Path("unexpected.ddc")) is None
    assert archive_sequence_bounds(Path("gop_00050_00025_bad.ddc")) is None


def test_locate_archive_prefilters_by_gop_name(monkeypatch: pytest.MonkeyPatch) -> None:
    archive_paths = [
        "/codec/gop_00000_00025_linear.ddc",
        "/codec/gop_00025_00050_linear.ddc",
        "/codec/gop_00050_00075_linear.ddc",
    ]
    visited: list[Path] = []

    def fake_read_metadata(archive_path: Path) -> dict[str, list[int]]:
        visited.append(archive_path)
        return {"middle_sequences": list(range(26, 50))}

    monkeypatch.setattr(
        "scripts.decode_dense_sequence.read_archive_metadata_with_retry",
        fake_read_metadata,
    )
    archive_path, frame_index = locate_archive(archive_paths, 30)
    assert archive_path == Path(archive_paths[1])
    assert frame_index == 5
    assert visited == [Path(archive_paths[1])]


def test_keyframe_archive_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.phdf"
    archive = tmp_path / "source.ddckf"
    reconstructed = tmp_path / "reconstructed.phdf"
    write_phdf(source)

    compressed = compress_keyframe(
        source,
        archive,
        ("prims.rho", "prims.uvec"),
        workers=2,
    )
    assert compressed["format"] == KEYFRAME_BZIP2_SHUFFLE_FORMAT
    assert compressed["dataset_count"] == 2
    assert compressed["chunk_count"] == 8
    validation = validate_keyframe_archive(archive)
    assert validation["chunk_count"] == 8

    decompressed = decompress_keyframe(archive, reconstructed)
    assert decompressed["dataset_count"] == 2
    assert decompressed["chunk_count"] == 8
    assert_hdf5_equal(source, reconstructed)


def test_temporal_keyframe_archives_round_trip_exactly(tmp_path: Path) -> None:
    sources = []
    for index in range(3):
        source = tmp_path / f"source_{index}.phdf"
        write_phdf(source)
        with h5py.File(source, "r+") as handle:
            handle["prims.rho"][...] += np.float32(index * 0.125)
            handle["prims.uvec"][...] += np.float32(index * index * 0.03125)
        sources.append(source)

    first_order = tmp_path / "first_order.ddckf"
    second_order = tmp_path / "second_order.ddckf"
    second_order_xz = tmp_path / "second_order_xz.ddckf"
    second_order_xz_zigzag = tmp_path / "second_order_xz_zigzag.ddckf"
    first_result = compress_keyframe(
        sources[1],
        first_order,
        ("prims.rho", "prims.uvec"),
        reference_paths=(sources[0],),
        temporal_order=1,
        workers=2,
    )
    second_result = compress_keyframe(
        sources[2],
        second_order,
        ("prims.rho", "prims.uvec"),
        reference_paths=(sources[1], sources[0]),
        temporal_order=2,
        workers=2,
    )
    second_xz_result = compress_keyframe(
        sources[2],
        second_order_xz,
        ("prims.rho", "prims.uvec"),
        reference_paths=(sources[1], sources[0]),
        temporal_order=2,
        compression="xz",
        workers=2,
    )
    second_xz_zigzag_result = compress_keyframe(
        sources[2],
        second_order_xz_zigzag,
        ("prims.rho", "prims.uvec"),
        reference_paths=(sources[1], sources[0]),
        temporal_order=2,
        compression="xz",
        temporal_zigzag=True,
        workers=2,
    )

    assert first_result["format"] == KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT
    assert second_result["format"] == KEYFRAME_TEMPORAL_BZIP2_SHUFFLE_FORMAT
    assert second_xz_result["format"] == KEYFRAME_TEMPORAL_XZ_SHUFFLE_FORMAT
    assert (
        second_xz_zigzag_result["format"]
        == KEYFRAME_TEMPORAL_XZ_ZIGZAG_SHUFFLE_FORMAT
    )
    assert second_xz_zigzag_result["temporal_zigzag"]
    assert not validate_keyframe_archive(second_order)["target_crc_verified"]
    assert validate_keyframe_archive(
        second_order,
        reference_paths=(sources[1], sources[0]),
    )["target_crc_verified"]
    with pytest.raises(ValueError, match="requires 2 reference paths"):
        decompress_keyframe(second_order, tmp_path / "missing_refs.phdf")
    with pytest.raises(ValueError, match="CRC failure"):
        decompress_keyframe(
            second_order,
            tmp_path / "wrong_refs.phdf",
            reference_paths=(sources[0], sources[0]),
        )

    reconstructed_first = tmp_path / "reconstructed_first.phdf"
    reconstructed_second = tmp_path / "reconstructed_second.phdf"
    reconstructed_second_xz = tmp_path / "reconstructed_second_xz.phdf"
    reconstructed_second_xz_zigzag = tmp_path / "reconstructed_second_xz_zigzag.phdf"
    decompress_keyframe(
        first_order,
        reconstructed_first,
        reference_paths=(sources[0],),
    )
    decompress_keyframe(
        second_order,
        reconstructed_second,
        reference_paths=(sources[1], sources[0]),
    )
    decompress_keyframe(
        second_order_xz,
        reconstructed_second_xz,
        reference_paths=(sources[1], sources[0]),
    )
    zigzag_validation = validate_keyframe_archive(
        second_order_xz_zigzag,
        reference_paths=(sources[1], sources[0]),
    )
    assert zigzag_validation["target_crc_verified"]
    assert zigzag_validation["temporal_zigzag"]
    decompress_keyframe(
        second_order_xz_zigzag,
        reconstructed_second_xz_zigzag,
        reference_paths=(sources[1], sources[0]),
    )
    assert_hdf5_equal(sources[1], reconstructed_first)
    assert_hdf5_equal(sources[2], reconstructed_second)
    assert_hdf5_equal(sources[2], reconstructed_second_xz)
    assert_hdf5_equal(sources[2], reconstructed_second_xz_zigzag)


def test_keyframe_archive_rejects_missing_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.phdf"
    write_phdf(source)
    with pytest.raises(KeyError, match="missing"):
        compress_keyframe(source, tmp_path / "source.ddckf", ("prims.missing",))


def test_keyframe_archive_requires_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.phdf"
    archive = tmp_path / "source.ddckf"
    write_phdf(source)
    archive.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        compress_keyframe(source, archive, ("prims.rho",))


def test_sequence_decodes_after_raw_keyframes_are_deleted(tmp_path: Path) -> None:
    segment = tmp_path / "segment"
    output = tmp_path / "codec"
    segment.mkdir()
    for sequence, value in enumerate((1.0, 2.5, 3.25, 4.0)):
        path = segment / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = sequence * 0.1
            handle.create_dataset(
                "prims.rho",
                data=np.full((1, 1, 2, 2, 2), value, dtype=np.float32),
            )

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/encode_dense_sequence.py"),
            "--segment-dir",
            str(segment),
            "--output-dir",
            str(output),
            "--keyframe-stride",
            "3",
            "--datasets",
            "prims.rho",
            "--bits",
            "8",
            "--archive-backend",
            "channel-bzip2-adaptive",
            "--keyframe-backend",
            "bzip2-shuffle-temporal",
            "--keyframe-anchor-interval",
            "4",
            "--delete-middle-frames",
            "--delete-keyframes-after-encode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_path = output / "sequence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["keyframe_storage"]["backend"] == "bzip2-shuffle-temporal"
    assert [
        row["temporal_order"] for row in manifest["keyframe_storage"]["archives"]
    ] == [0, 1]
    assert manifest["keyframes_deleted"]
    assert manifest["middle_frames_deleted"]
    assert not list(segment.glob("*.phdf"))

    exact = tmp_path / "exact.phdf"
    middle_one = tmp_path / "middle_one.phdf"
    middle_two = tmp_path / "middle_two.phdf"
    decode_sequence_frame(manifest_path, 0, exact)
    decode_sequence_frames(
        manifest_path,
        {1: middle_one, 2: middle_two},
    )
    with h5py.File(exact, "r") as handle:
        np.testing.assert_array_equal(handle["prims.rho"][...], 1.0)
    with h5py.File(middle_one, "r") as handle:
        np.testing.assert_allclose(handle["prims.rho"][...], 2.5, rtol=1.0e-6)
    with h5py.File(middle_two, "r") as handle:
        np.testing.assert_allclose(handle["prims.rho"][...], 3.25, rtol=1.0e-6)


def test_sequence_prefers_retained_raw_keyframe(tmp_path: Path) -> None:
    retained = tmp_path / "raw_dt0p5" / "tiny.out0.00000.phdf"
    retained.parent.mkdir()
    write_phdf(retained)
    missing_source = tmp_path / "segment" / retained.name
    manifest = {
        "keyframes": [str(missing_source)],
        "keyframe_storage": {
            "archives": [
                {
                    "sequence": 0,
                    "archive_path": str(tmp_path / "missing.ddckf"),
                }
            ]
        },
        "retention": {
            "files": [
                {
                    "sequence": 0,
                    "target": str(retained),
                    "purpose": "raw_baseline",
                }
            ]
        },
    }

    output = tmp_path / "decoded.phdf"
    result = materialize_keyframes(manifest, {0: output})[0]

    assert result["source_retained_keyframe"] == str(retained)
    assert result["source_keyframe"] == str(retained)
    assert_hdf5_equal(retained, output)


def test_sequence_middle_frame_uses_retained_raw_keyframes(tmp_path: Path) -> None:
    segment = tmp_path / "segment"
    output = tmp_path / "codec"
    retained = tmp_path / "raw_dt5"
    segment.mkdir()
    retained.mkdir()
    for sequence, value in enumerate((1.0, 2.5, 4.0)):
        path = segment / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = sequence * 0.1
            handle.create_dataset(
                "prims.rho",
                data=np.full((1, 1, 2, 2, 2), value, dtype=np.float32),
            )

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/encode_dense_sequence.py"),
            "--segment-dir",
            str(segment),
            "--output-dir",
            str(output),
            "--keyframe-stride",
            "2",
            "--datasets",
            "prims.rho",
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

    manifest_path = output / "sequence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    retention = []
    for sequence in (0, 2):
        source = segment / f"tiny.out0.{sequence:05d}.phdf"
        target = retained / source.name
        source.replace(target)
        retention.append(
            {
                "sequence": sequence,
                "target": str(target),
                "purpose": "raw_baseline",
            }
        )
    manifest["retention"] = {"files": retention}
    manifest_path.write_text(json.dumps(manifest))

    reconstructed = tmp_path / "middle.phdf"
    decode_sequence_frame(manifest_path, 1, reconstructed)

    with h5py.File(reconstructed, "r") as handle:
        np.testing.assert_allclose(handle["prims.rho"][...], 2.5, rtol=1.0e-6)


def test_existing_sequence_manifest_can_be_repacked(tmp_path: Path) -> None:
    keyframes = []
    for sequence, value in ((0, 1.0), (2, 4.0)):
        path = tmp_path / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle["prims.rho"] = np.full((4, 8), value, dtype=np.float32)
        keyframes.append(path)
    residual = tmp_path / "gop.ddc"
    residual.write_bytes(b"residual")
    manifest = tmp_path / "sequence_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "dense_dump_codec_sequence_v1",
                "complete": True,
                "datasets": ["prims.rho"],
                "keyframes": [str(path) for path in keyframes],
                "storage": {
                    "dense_phdf_bytes": 10_000,
                    "keyframe_bytes": sum(path.stat().st_size for path in keyframes),
                },
                "codec_schemes": {
                    "scheme": {
                        "archive_size_bytes": residual.stat().st_size,
                        "total_with_keyframes_bytes": residual.stat().st_size
                        + sum(path.stat().st_size for path in keyframes),
                        "ratio_vs_dense_phdf": 1.0,
                    }
                },
            }
        )
    )
    output = tmp_path / "repacked.json"

    result = repack_manifest(
        manifest,
        output,
        tmp_path / "keyframes",
        workers=2,
    )

    repacked = json.loads(output.read_text())
    assert result["all_chunks_verified"]
    assert repacked["keyframe_storage"]["backend"] == "bzip2-shuffle"
    assert len(repacked["keyframe_storage"]["archives"]) == 2
    assert repacked["codec_schemes"]["scheme"]["keyframe_backend"] == (
        "bzip2-shuffle"
    )


def test_temporal_repack_decodes_dependency_chain_after_source_deletion(
    tmp_path: Path,
) -> None:
    keyframes = []
    for sequence in (0, 2, 4, 6, 8):
        path = tmp_path / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = sequence * 0.1
            handle["prims.rho"] = np.full(
                (4, 8),
                1.0 + sequence * 0.125 + sequence * sequence * 0.01,
                dtype=np.float32,
            )
        keyframes.append(path)
    manifest = tmp_path / "sequence_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "dense_dump_codec_sequence_v1",
                "complete": True,
                "datasets": ["prims.rho"],
                "keyframes": [str(path) for path in keyframes],
                "storage": {
                    "dense_phdf_bytes": 100_000,
                    "keyframe_bytes": sum(path.stat().st_size for path in keyframes),
                },
                "codec_schemes": {
                    "scheme": {
                        "archive_size_bytes": 100,
                        "total_with_keyframes_bytes": 100,
                        "ratio_vs_dense_phdf": 1.0,
                    }
                },
            }
        )
    )
    output_manifest = tmp_path / "temporal.json"
    repack_manifest(
        manifest,
        output_manifest,
        tmp_path / "temporal_keyframes",
        workers=2,
        backend="bzip2-xz-zigzag-temporal",
        anchor_interval=4,
        delete_source_keyframes=True,
    )

    repacked = json.loads(output_manifest.read_text())
    assert repacked["keyframe_storage"]["backend"] == "bzip2-xz-zigzag-temporal"
    assert [
        row["compression"] for row in repacked["keyframe_storage"]["archives"]
    ] == ["bzip2", "xz", "xz", "xz", "bzip2"]
    assert [
        row["temporal_zigzag"] for row in repacked["keyframe_storage"]["archives"]
    ] == [False, True, True, True, False]
    assert [
        row["temporal_order"] for row in repacked["keyframe_storage"]["archives"]
    ] == [0, 1, 2, 2, 0]
    assert not any(path.exists() for path in keyframes)

    reconstructed = tmp_path / "reconstructed_chain.phdf"
    result = decode_sequence_frame(output_manifest, 6, reconstructed)
    assert result["exact_keyframe"]
    with h5py.File(reconstructed, "r") as handle:
        np.testing.assert_array_equal(
            handle["prims.rho"][...],
            np.full((4, 8), 1.0 + 6 * 0.125 + 36 * 0.01, dtype=np.float32),
        )

    batch_outputs = {
        sequence: tmp_path / f"batch_{sequence:05d}.phdf"
        for sequence in (0, 2, 4, 6, 8)
    }
    materialize_keyframes(repacked, batch_outputs)
    for sequence, path in batch_outputs.items():
        with h5py.File(path, "r") as handle:
            np.testing.assert_array_equal(
                handle["prims.rho"][...],
                np.full(
                    (4, 8),
                    1.0 + sequence * 0.125 + sequence * sequence * 0.01,
                    dtype=np.float32,
                ),
            )
