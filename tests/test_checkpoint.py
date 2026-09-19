import json
from pathlib import Path

import h5py
import pytest

from dense_dump_codec.checkpoint import (
    combine_histories,
    latest_complete_checkpoint,
    prune_native_checkpoints,
    read_checkpoint_mesh_layout,
    read_native_checkpoint,
    rollback_to_native_checkpoint,
    truncate_histories,
)


def write_checkpoint(
    path: Path,
    *,
    time: float,
    cycle: int,
    output0_file_number: int,
    output1_file_number: int,
    output2_file_number: int,
    includes_ghost_zones: bool = True,
) -> None:
    input_text = f"""<parthenon/mesh>
nx1 = 288
nx2 = 128
nx3 = 128
pack_size = -1
<parthenon/meshblock>
nx1 = 72
nx2 = 32
nx3 = 32
<parthenon/job>
problem_id = resize_restart
<parthenon/output0>
dt = 0.1
next_time = {time + 0.1}
file_number = {output0_file_number}
<parthenon/output1>
dt = 10
next_time = {time + 10}
file_number = {output1_file_number}
<parthenon/output2>
dt = 0.1
next_time = {time + 0.1}
file_number = {output2_file_number}
"""
    with h5py.File(path, "w") as handle:
        info = handle.create_group("Info")
        info.attrs["Time"] = time
        info.attrs["NCycle"] = cycle
        info.attrs["dt"] = 0.01
        info.attrs["IncludesGhost"] = int(includes_ghost_zones)
        info.attrs["NGhost"] = 4
        info.attrs["NumMeshBlocks"] = 64
        info.attrs["BlocksPerPE"] = 64
        handle.create_group("Input").attrs["File"] = input_text


def test_read_and_select_latest_complete_periodic_checkpoint(tmp_path: Path) -> None:
    for number in (2, 10, 3):
        write_checkpoint(
            tmp_path / f"resize_restart.out1.{number:05d}.rhdf",
            time=25000.0 + number,
            cycle=number,
            output0_file_number=number + 1,
            output1_file_number=number + 1,
            output2_file_number=number + 1,
        )
    (tmp_path / "resize_restart.out1.00011.rhdf").write_bytes(b"partial")
    (tmp_path / "resize_restart.out1.final.rhdf").write_bytes(b"final")

    checkpoint = read_native_checkpoint(tmp_path / "resize_restart.out1.00010.rhdf")

    assert checkpoint.time == 25010.0
    assert checkpoint.last_dense_sequence == 10
    assert latest_complete_checkpoint(tmp_path).name.endswith("00010.rhdf")


def test_read_checkpoint_mesh_layout_validates_decomposition(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resize_restart.out1.00000.rhdf"
    write_checkpoint(
        checkpoint,
        time=25000.0,
        cycle=1,
        output0_file_number=1,
        output1_file_number=1,
        output2_file_number=0,
    )

    layout = read_checkpoint_mesh_layout(checkpoint)

    assert layout.root_grid_shape == (288, 128, 128)
    assert layout.meshblock_shape == (72, 32, 32)
    assert layout.meshblock_count == 64
    assert layout.blocks_per_rank == 64
    assert layout.ghost_zones == 4
    assert layout.pack_size == -1


def test_prune_native_checkpoints_hashes_and_keeps_latest(tmp_path: Path) -> None:
    for number in range(4):
        write_checkpoint(
            tmp_path / f"resize_restart.out1.{number:05d}.rhdf",
            time=25000.0 + number,
            cycle=number,
            output0_file_number=number + 1,
            output1_file_number=number + 1,
            output2_file_number=number + 1,
        )

    manifest = prune_native_checkpoints(tmp_path, keep=2)

    assert manifest["retained_checkpoint_count"] == 2
    assert len(manifest["pruned"]) == 2
    assert [row["checkpoint_number"] for row in manifest["checkpoints"]] == [2, 3]
    assert all(len(row["sha256"]) == 64 for row in manifest["checkpoints"])
    assert not (tmp_path / "resize_restart.out1.00000.rhdf").exists()
    assert (tmp_path / "resize_restart.out1.00003.rhdf").exists()


def test_rollback_rejects_checkpoint_without_ghost_zones(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    checkpoint = output_dir / "resize_restart.out1.00000.rhdf"
    write_checkpoint(
        checkpoint,
        time=25000.0,
        cycle=0,
        output0_file_number=1,
        output1_file_number=1,
        output2_file_number=1,
        includes_ghost_zones=False,
    )

    with pytest.raises(ValueError, match="does not include ghost zones"):
        rollback_to_native_checkpoint(checkpoint, output_dir=output_dir)


def test_restart_only_checkpoint_allows_isolated_rollback(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    checkpoint = output_dir / "resize_restart.out1.00000.rhdf"
    write_checkpoint(
        checkpoint,
        time=25000.0,
        cycle=0,
        output0_file_number=0,
        output1_file_number=1,
        output2_file_number=0,
    )
    with h5py.File(checkpoint, "r+") as handle:
        input_text = str(handle["Input"].attrs["File"])
        input_text = input_text.replace("next_time = 25000.1\nfile_number = 0\n", "")
        input_text = input_text.replace("next_time = 25000.1\nfile_number = 0\n", "")
        handle["Input"].attrs.modify("File", input_text)

    parsed = read_native_checkpoint(checkpoint)
    report = rollback_to_native_checkpoint(checkpoint, output_dir=output_dir)

    assert parsed.last_dense_sequence == -1
    assert report["checkpoint"]["last_dense_sequence"] == -1


def test_rollback_truncates_outputs_codec_state_raw_and_history(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    codec_dir = tmp_path / "ddc"
    raw_dir = tmp_path / "raw"
    output_dir.mkdir()
    codec_dir.mkdir()
    raw_dir.mkdir()
    checkpoint = output_dir / "resize_restart.out1.00002.rhdf"
    write_checkpoint(
        checkpoint,
        time=25000.4,
        cycle=40,
        output0_file_number=5,
        output1_file_number=3,
        output2_file_number=3,
    )
    write_checkpoint(
        output_dir / "resize_restart.out1.00003.rhdf",
        time=25000.6,
        cycle=60,
        output0_file_number=7,
        output1_file_number=4,
        output2_file_number=5,
    )
    (output_dir / "resize_restart.out1.final.rhdf").write_bytes(b"stale-final")
    for sequence in range(7):
        path = output_dir / f"resize_restart.out0.{sequence:05d}.phdf"
        path.write_bytes(f"frame-{sequence}".encode())
        Path(f"{path}.xdmf").write_text("xdmf")
    history = output_dir / "resize_restart.hst"
    history.write_text("# header\n# columns\n" + "".join(f"{row}\n" for row in range(5)))

    gops = []
    for start, end in ((0, 2), (2, 4), (4, 6)):
        gop_dir = codec_dir / "gops" / f"{start:05d}_{end:05d}"
        gop_dir.mkdir(parents=True)
        summary = gop_dir / "summary.json"
        summary.write_text("{}")
        gops.append(
            {
                "start_sequence": start,
                "end_sequence": end,
                "summary": str(summary),
            }
        )
    retained_path = raw_dir / "resize_restart.out0.00002.phdf"
    removed_retained_path = raw_dir / "resize_restart.out0.00006.phdf"
    retained_path.write_bytes(b"retained")
    removed_retained_path.write_bytes(b"future")
    orphan = raw_dir / "truth" / "resize_restart.out0.00006.phdf"
    orphan.parent.mkdir()
    orphan.write_bytes(b"orphan")
    state = {
        "format": "dense_dump_codec_stream_state_v1",
        "configuration": {"keyframe_stride": 2, "start_sequence": 0},
        "gops": gops,
        "frames": {str(sequence): {"time": sequence} for sequence in range(7)},
        "deleted_middle_files": [
            str(output_dir / f"resize_restart.out0.{sequence:05d}.phdf")
            for sequence in (1, 3, 5)
        ],
        "retained_raw_files": {
            str(retained_path): {"sequence": 2},
            str(removed_retained_path): {"sequence": 6},
        },
        "finalized": False,
    }
    (codec_dir / "stream_state.json").write_text(json.dumps(state))
    (codec_dir / "sequence_manifest.json").write_text("{}")
    (output_dir / "status.json").write_text("{}")

    report = rollback_to_native_checkpoint(
        checkpoint,
        output_dir=output_dir,
        codec_dir=codec_dir,
        raw_dirs=(raw_dir,),
    )

    assert report["checkpoint"]["last_dense_sequence"] == 4
    assert not (output_dir / "resize_restart.out0.00005.phdf").exists()
    assert (output_dir / "resize_restart.out0.00004.phdf").exists()
    assert not (output_dir / "resize_restart.out1.00003.rhdf").exists()
    assert not (output_dir / "resize_restart.out1.final.rhdf").exists()
    assert history.read_text().splitlines() == ["# header", "# columns", "0", "1", "2"]
    rolled_back = json.loads((codec_dir / "stream_state.json").read_text())
    assert [row["end_sequence"] for row in rolled_back["gops"]] == [2, 4]
    assert max(map(int, rolled_back["frames"])) == 4
    assert not (codec_dir / "gops" / "00004_00006").exists()
    assert retained_path.exists()
    assert not removed_retained_path.exists()
    assert not orphan.exists()
    assert not (codec_dir / "sequence_manifest.json").exists()
    assert (output_dir / "resume_recovery_latest.json").is_file()


def test_history_segments_use_global_checkpoint_row_count(tmp_path: Path) -> None:
    first = tmp_path / "resize_restart.hst"
    second = tmp_path / "resized_restart.hst"
    first.write_text("# header\n# columns\n0\n1\n2\n")
    second.write_text("3\n4\n5\n")

    reports = truncate_histories(tmp_path, 5)
    combined = combine_histories(tmp_path)

    assert [row["rows_after"] for row in reports] == [3, 2]
    assert second.read_text().splitlines() == ["3", "4"]
    assert combined["row_count"] == 5
    assert (tmp_path / "kharma_history_combined.hst").read_text().splitlines() == [
        "# header",
        "# columns",
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
