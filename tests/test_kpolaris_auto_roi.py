import pytest

from dense_dump_codec.kpolaris_parameters import automatic_radius, verify_effective_radius


def test_auto_radius_tracks_file_and_cli_precedence(tmp_path):
    first, second = tmp_path / "one.par", tmp_path / "two.par"
    first.write_text("outer_radius = 100\n")
    second.write_text("--outer-radius 50 # alternate supported syntax\n")
    assert automatic_radius(first, []) == 100
    assert automatic_radius(first, [f"--params={second}"]) == 50
    assert automatic_radius(first, ["--outer-radius=20", f"--params={second}"]) == 20
    first.write_text("model=kharma\n")
    assert automatic_radius(first, []) is None
    assert automatic_radius(first, ["--outer_radius=-1"]) is None
    with pytest.raises(ValueError):
        automatic_radius(first, ["--outer_radius=nan"])
    with pytest.raises(ValueError):
        automatic_radius(first, ["--slow_light_batch_jobs=unknown.json"])
    verify_effective_radius(second, 100)
    with pytest.raises(ValueError):
        verify_effective_radius(second, 20)
