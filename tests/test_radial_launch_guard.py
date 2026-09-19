from pathlib import Path

import pytest

from scripts.run_kpolaris_ddc_window import validate_radial_region


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts/serve_ddc_native.py"


@pytest.mark.parametrize("extra", [[], ["--outer_radius=50"], ["--outer_radius=100"]])
def test_roi_covers_effective_radiation_region(tmp_path, extra):
    parameters = tmp_path / "model.par"
    parameters.write_text("outer_radius = 100 # unchanged physics\n")
    validate_radial_region(100, "native", SERVER, parameters, extra)


@pytest.mark.parametrize("extra", [["--outer_radius=101"], ["--outer_radius=nan"],
                                   ["--outer_radius=0"], ["--outer_radius", "50"],
                                   ["--parameter_file=unverified.par"]])
def test_roi_rejects_unsafe_override(tmp_path, extra):
    parameters = tmp_path / "model.par"
    parameters.write_text("outer_radius=100\n")
    with pytest.raises(ValueError):
        validate_radial_region(100, "native", SERVER, parameters, extra)


def test_roi_requires_explicit_supported_setup(tmp_path):
    parameters = tmp_path / "model.par"
    parameters.write_text("model=kharma\n")
    with pytest.raises(ValueError, match="explicit"):
        validate_radial_region(100, "native", SERVER, parameters, [])
    validate_radial_region(100, "native", SERVER, parameters, ["--outer_radius=100"])
    with pytest.raises(ValueError, match="native transport"):
        validate_radial_region(100, "phdf", SERVER, parameters, ["--outer_radius=100"])
    with pytest.raises(ValueError, match="owned"):
        validate_radial_region(100, "native", tmp_path / "custom.py", parameters, [])
    validate_radial_region(None, "phdf", tmp_path / "custom.py", parameters, [])
