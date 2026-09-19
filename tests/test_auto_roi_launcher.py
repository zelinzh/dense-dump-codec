import json
import subprocess
import sys
from pathlib import Path

import pytest

from dense_dump_codec.kpolaris_parameters import automatic_radius, batch_output_paths


ROOT = Path(__file__).resolve().parents[1]


def test_shared_physics_batch_has_one_radius(tmp_path):
    parameters, jobs = tmp_path / "model.par", tmp_path / "jobs.txt"
    jobs.write_text(f'10 0 2 "{tmp_path}/first image.h5"\n11 1 3 {tmp_path}/second.h5\n')
    parameters.write_text(f"outer_radius=50\nslow_light_batch_jobs={jobs}\n")
    assert automatic_radius(parameters, []) == 50
    assert len(batch_output_paths(parameters, [])) == 2
    jobs.write_text(f"10 0 2 {tmp_path}/image.h5 outer_radius=200\n")
    with pytest.raises(ValueError):
        automatic_radius(parameters, [])


def test_auto_link_is_forwarded_without_changing_physics(tmp_path):
    manifest, parameters, binary = tmp_path / "manifest.json", tmp_path / "model.par", tmp_path / "binary"
    manifest.write_text(json.dumps({"frames": [
        {"sequence": sequence, "time": 10 + 0.1 * sequence, "path": f"frame{sequence}"}
        for sequence in range(5)]}))
    parameters.write_text("model=kharma\nouter-radius 50\n")
    binary.write_bytes(b"dry-run-only")
    output = tmp_path / "result.h5"
    command = [sys.executable, str(ROOT / "scripts/run_kpolaris_ddc_window.py"),
        f"--manifest={manifest}", f"--kpolaris-binary={binary}", f"--parameter-file={parameters}",
        "--observation-time=10.4", "--fluid-time-min=10.1", "--fluid-time-max=10.3",
        f"--output={output}", f"--server={ROOT}/scripts/serve_ddc_native.py",
        "--native-radius-max=auto", f"--native-working-cache={tmp_path}/working",
        "--kpolaris-arg=--outer_radius=20", "--dry-run"]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads(Path(f"{output}.ddc-run.json").read_text())
    assert report["native_radius_policy"] == "auto"
    assert report["native_radius_max"] == 20
    service = report["server_command"]
    assert service[service.index("--native-radius-max") + 1] == "20.0"
    assert "--outer_radius=20" in report["command"]
    assert "--native-working-cache" in service
    assert parameters.read_text() == "model=kharma\nouter-radius 50\n"
