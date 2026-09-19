from __future__ import annotations

import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dense_dump_codec import (
    CHANNEL_BZIP2_ADAPTIVE_FORMAT,
    CHANNEL_BZIP2_DELTA_SHUFFLE_FORMAT,
    CHANNEL_BZIP2_DELTA_FORMAT,
    CHANNEL_BZIP2_FORMAT,
    ChannelBzip2Archive,
    open_archive,
    repack_zip_to_channel_bzip2,
)
from decode_dump_codec import decode_dataset, read_metadata
from prototype_dump_codec import evaluate_codec
from rebase_codec_comparison_storage import rebase_storage
from repack_ddc_sequence import repack_sequence


def write_source(path: Path) -> dict[str, bytes]:
    members = {
        "metadata.json": b'{"compression":"zip_deflate","bits":8}',
        "frames/00001/prims_rho_q8.npy": b"rho-one" * 31,
        "frames/00002/prims_rho_q8.npy": b"rho-two" * 37,
        "frames/00001/prims_rho_scale.npy": b"scale-one" * 11,
        "frames/00002/prims_rho_scale.npy": b"scale-two" * 13,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return members


def test_open_archive_preserves_legacy_zip(tmp_path: Path) -> None:
    source = tmp_path / "legacy.ddc"
    members = write_source(source)
    with open_archive(source) as archive:
        assert isinstance(archive, zipfile.ZipFile)
        assert archive.read("frames/00001/prims_rho_q8.npy") == members[
            "frames/00001/prims_rho_q8.npy"
        ]


def test_repack_is_deterministic_and_random_accessible(tmp_path: Path) -> None:
    source = tmp_path / "legacy.ddc"
    members = write_source(source)
    outputs = [tmp_path / "channel-a.ddc", tmp_path / "channel-b.ddc"]
    for output in outputs:
        result = repack_zip_to_channel_bzip2(
            source,
            output,
            chunk_frames=2,
            compression_level=9,
            workers=2,
            metadata_updates={
                "compression": "channel_bzip2",
                "channel_chunk_frames": 2,
            },
        )
        assert result["format"] == CHANNEL_BZIP2_FORMAT
        assert result["chunk_count"] == 2
        with open_archive(output) as archive:
            assert isinstance(archive, ChannelBzip2Archive)
            assert archive.testzip() is None
            assert set(archive.namelist()) == set(members)
            for name, payload in members.items():
                if name != "metadata.json":
                    assert archive.read(name) == payload
            metadata = json.loads(archive.read("metadata.json"))
            assert metadata["compression"] == "channel_bzip2"
            assert metadata["channel_chunk_frames"] == 2
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_temporal_delta_repack_is_lossless_and_smaller(tmp_path: Path) -> None:
    source = tmp_path / "legacy.ddc"
    members: dict[str, bytes] = {"metadata.json": b"{}"}
    random = np.random.default_rng(42)
    base = random.integers(-20_000, 20_000, size=(64, 256), dtype=np.int16)
    velocity = random.integers(-500, 500, size=base.shape, dtype=np.int16)
    acceleration = random.integers(-8, 9, size=base.shape, dtype=np.int16)
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", members["metadata.json"])
        for frame_index in range(1, 6):
            payload = (
                base.astype(np.int32)
                + frame_index * velocity.astype(np.int32)
                + frame_index**2 * acceleration.astype(np.int32)
            ).astype(np.int16)
            buffer = __import__("io").BytesIO()
            np.save(buffer, payload, allow_pickle=False)
            name = f"frames/{frame_index:05d}/prims_B_q16.npy"
            members[name] = buffer.getvalue()
            archive.writestr(name, members[name])
            scale_buffer = __import__("io").BytesIO()
            np.save(
                scale_buffer,
                np.full((3,), 0.125 + frame_index * 1.0e-5, dtype=np.float32),
                allow_pickle=False,
            )
            scale_name = f"frames/{frame_index:05d}/prims_B_scale.npy"
            members[scale_name] = scale_buffer.getvalue()
            archive.writestr(scale_name, members[scale_name])

    plain = tmp_path / "plain.ddc"
    delta = tmp_path / "delta.ddc"
    repack_zip_to_channel_bzip2(source, plain, chunk_frames=5)
    result = repack_zip_to_channel_bzip2(
        source,
        delta,
        chunk_frames=5,
        temporal_delta=True,
    )

    assert result["format"] == CHANNEL_BZIP2_DELTA_FORMAT
    assert result["preconditioner"] == "temporal-npy-delta-xor-v1"
    assert delta.stat().st_size < plain.stat().st_size
    with open_archive(delta) as archive:
        assert archive.testzip() is None
        for name, payload in members.items():
            assert archive.read(name) == payload

    delta_shuffle = tmp_path / "delta-shuffle.ddc"
    result_delta_shuffle = repack_zip_to_channel_bzip2(
        delta,
        delta_shuffle,
        chunk_frames=5,
        temporal_delta_shuffle=True,
    )
    assert result_delta_shuffle["format"] == CHANNEL_BZIP2_DELTA_SHUFFLE_FORMAT
    assert (
        result_delta_shuffle["preconditioner"]
        == "temporal-npy-delta-zigzag-byte-shuffle-v1"
    )
    assert delta_shuffle.stat().st_size < delta.stat().st_size
    with open_archive(delta_shuffle) as archive:
        assert archive.testzip() is None
        for name, payload in members.items():
            assert archive.read(name) == payload

    adaptive = tmp_path / "adaptive.ddc"
    result_adaptive = repack_zip_to_channel_bzip2(
        delta_shuffle,
        adaptive,
        chunk_frames=5,
        adaptive_temporal_order=True,
    )
    assert result_adaptive["format"] == CHANNEL_BZIP2_ADAPTIVE_FORMAT
    assert result_adaptive["adaptive_candidate_chunk_count"] == 1
    assert result_adaptive["adaptive_second_order_chunk_count"] == 1
    assert result_adaptive["adaptive_saving_vs_first_order_fraction"] > 0.0
    assert adaptive.stat().st_size < delta_shuffle.stat().st_size
    with open_archive(adaptive) as archive:
        assert archive.testzip() is None
        for name, payload in members.items():
            assert archive.read(name) == payload

    delta_from_plain = tmp_path / "delta-from-plain.ddc"
    result_from_plain = repack_zip_to_channel_bzip2(
        plain,
        delta_from_plain,
        chunk_frames=5,
        temporal_delta=True,
    )
    assert result_from_plain["format"] == CHANNEL_BZIP2_DELTA_FORMAT
    with open_archive(delta_from_plain) as archive:
        assert archive.testzip() is None
        for name, payload in members.items():
            assert archive.read(name) == payload


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_frames": 0}, "chunk_frames"),
        ({"compression_level": 0}, "compression_level"),
        ({"workers": 0}, "workers"),
    ],
)
def test_repack_rejects_invalid_configuration(
    tmp_path: Path,
    kwargs: dict[str, int],
    message: str,
) -> None:
    source = tmp_path / "legacy.ddc"
    write_source(source)
    with pytest.raises(ValueError, match=message):
        repack_zip_to_channel_bzip2(source, tmp_path / "output.ddc", **kwargs)


def test_repack_rejects_same_path(tmp_path: Path) -> None:
    source = tmp_path / "legacy.ddc"
    write_source(source)
    with pytest.raises(ValueError, match="must differ"):
        repack_zip_to_channel_bzip2(source, source)


@pytest.mark.parametrize(
    ("backend", "compression"),
    (
        ("channel-bzip2", "channel_bzip2"),
        ("channel-bzip2-delta", "channel_bzip2_delta"),
        ("channel-bzip2-delta-shuffle", "channel_bzip2_delta_shuffle"),
        ("channel-bzip2-adaptive", "channel_bzip2_adaptive"),
    ),
)
def test_encoder_writes_decodable_channel_backend(
    tmp_path: Path,
    backend: str,
    compression: str,
) -> None:
    segment_dir = tmp_path / "segment"
    segment_dir.mkdir()
    values = (1.0, 2.5, 4.0)
    for sequence, value in enumerate(values):
        path = segment_dir / f"tiny.out0.{sequence:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = sequence * 0.1
            handle["prims.rho"] = np.full((1, 1, 2, 2, 2), value, dtype=np.float32)

    summary = evaluate_codec(
        Namespace(
            segment_dir=segment_dir,
            output_dir=tmp_path / "codec",
            datasets="prims.rho",
            bits="8",
            dataset_bits="",
            dataset_scale_percentiles="",
            scale_mode="block-channel",
            scale_percentile=99.9,
            discard_outliers=False,
            compression_level=6,
            archive_backend=backend,
            channel_chunk_frames=5,
            channel_compression_level=9,
            archive_workers=2,
            start_sequence=None,
            end_sequence=None,
            max_middle_frames=0,
            output_json=None,
            skip_archives=False,
        )
    )

    scheme = next(iter(summary["codec_schemes"].values()))
    assert scheme["archive_backend"] == backend
    archive_path = Path(scheme["archive_path"])
    with open_archive(archive_path) as archive:
        metadata = read_metadata(archive)
        reconstructed = decode_dataset(archive, metadata, "prims.rho", 1)
        assert archive.testzip() is None
    np.testing.assert_allclose(reconstructed, values[1], rtol=1.0e-6)
    assert metadata["compression"] == compression


def test_sequence_repack_updates_storage_and_integrity(tmp_path: Path) -> None:
    source_archive = tmp_path / "legacy.ddc"
    write_source(source_archive)
    manifest_path = tmp_path / "sequence_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "dense_dump_codec_sequence_v1",
                "complete": True,
                "keyframe_stride": 5,
                "middle_frame_count": 2,
                "storage": {
                    "dense_phdf_bytes": 10_000,
                    "keyframe_bytes": 2_000,
                },
                "codec_schemes": {
                    "scheme": {
                        "bits": 8,
                        "dataset_bits": {"prims.rho": 8},
                        "dataset_scale_percentiles": {"prims.rho": 99.9},
                        "archive_paths": [str(source_archive)],
                        "archive_size_bytes": source_archive.stat().st_size,
                        "total_with_keyframes_bytes": 2_000
                        + source_archive.stat().st_size,
                    }
                },
                "integrity": {"sha256_enabled": True},
            }
        ),
        encoding="utf-8",
    )
    output_manifest = tmp_path / "repacked" / "sequence_manifest.json"

    result = repack_sequence(
        manifest_path,
        tmp_path / "repacked" / "archives",
        output_manifest,
        workers=2,
    )

    repacked = json.loads(output_manifest.read_text(encoding="utf-8"))
    scheme = repacked["codec_schemes"]["scheme"]
    assert result["all_members_crc_verified"]
    assert scheme["archive_backend"] == "channel-bzip2"
    assert scheme["archive_size_bytes"] == result["output_archive_bytes"]
    assert repacked["lossless_repack"]["member_crc_verified"]
    assert len(repacked["integrity"]["archives"]) == 1
    with open_archive(Path(scheme["archive_paths"][0])) as archive:
        assert archive.testzip() is None

    comparison = {
        "codec_scheme": "scheme",
        "codec_bits": 8,
        "codec_dataset_bits": {"prims.rho": 8},
        "codec_dataset_scale_percentiles": {"prims.rho": 99.9},
        "codec_keyframe_stride": 5,
        "truth_bytes": 10_000,
        "codec_storage": {
            "archive_bytes": source_archive.stat().st_size,
            "keyframe_bytes": 2_000,
            "total_bytes": 2_000 + source_archive.stat().st_size,
            "ratio_vs_dense_phdf": 10_000
            / (2_000 + source_archive.stat().st_size),
        },
        "codec": {"errors": {"rho": {"nrmse": 0.1}}},
    }
    rebased = rebase_storage(
        comparison,
        repacked,
        manifest_path=output_manifest,
    )
    assert rebased["codec"]["errors"] == comparison["codec"]["errors"]
    assert rebased["codec_storage"]["archive_bytes"] == result[
        "output_archive_bytes"
    ]
    assert rebased["codec_lossless_repack"]["member_crc_verified"]
