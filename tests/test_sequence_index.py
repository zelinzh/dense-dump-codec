import json
from pathlib import Path

import pytest

from dense_dump_codec.sequence import (
    DDCFrameMaterializer,
    DenseSequenceIndex,
    FrameRecord,
)


def _index() -> DenseSequenceIndex:
    return DenseSequenceIndex(
        [
            FrameRecord(0, 10.0, "frame0.phdf", exact_keyframe=True),
            FrameRecord(1, 10.1, "frame1.phdf"),
            FrameRecord(2, 10.2, "frame2.phdf", exact_keyframe=True),
        ]
    )


def test_sequence_index_brackets_physical_time() -> None:
    bracket = _index().bracket(10.15)

    assert bracket.lower.sequence == 1
    assert bracket.upper.sequence == 2
    assert bracket.upper_weight == pytest.approx(0.5)
    assert not bracket.exact


def test_sequence_index_resolves_exact_time_and_bounds() -> None:
    bracket = _index().bracket(10.1 + 1.0e-11)
    assert bracket.exact
    assert bracket.lower.sequence == 1

    with pytest.raises(ValueError, match="precedes"):
        _index().bracket(9.9)


def test_sequence_index_selects_minimal_covering_range() -> None:
    selected = _index().covering_range(10.05, 10.15)
    assert [row.sequence for row in selected] == [0, 1, 2]

    exact = _index().covering_range(10.1, 10.1)
    assert [row.sequence for row in exact] == [1, 2]

    with pytest.raises(ValueError, match="end_time"):
        _index().covering_range(10.2, 10.1)


def test_sequence_index_loads_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "sequence_manifest.json"
    manifest.write_text(
        json.dumps({"frames": [row.to_json() for row in _index().frames]})
    )

    loaded = DenseSequenceIndex.from_manifest(manifest)

    assert loaded.by_sequence(2).time == 10.2


def test_materializer_decodes_only_bracketing_frames(tmp_path: Path) -> None:
    calls: list[int] = []

    def decode(sequence: int, output: Path) -> dict:
        calls.append(sequence)
        output.write_text(str(sequence))
        return {"sequence": sequence}

    materializer = DDCFrameMaterializer(
        _index(), tmp_path, decode, maximum_cache_files=2
    )
    result = materializer.materialize_time(10.15)

    assert calls == [1, 2]
    assert result["upper_weight"] == pytest.approx(0.5)
    assert Path(result["lower"]["materialized_path"]).read_text() == "1"

    materializer.materialize_time(10.0)
    assert calls == [1, 2, 0]
    assert not (tmp_path / "ddc_frame_00001.phdf").exists()

    result = materializer.materialize_sequence(1)
    assert calls == [1, 2, 0, 1]
    assert result["sequence"] == 1
    assert Path(result["materialized_path"]).read_text() == "1"
