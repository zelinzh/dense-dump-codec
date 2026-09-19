import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("transport,owned,success", [("native", True, True),
                          ("phdf", True, False), ("native", False, False)])
def test_explicit_library_is_scoped_and_recorded(tmp_path, transport, owned, success):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": [dict(sequence=position, time=float(position),
        path=f"frame{position}") for position in range(5)]}))
    library = tmp_path / "predict.so"
    library.write_bytes(b"no execution during dry-run")
    parameters = tmp_path / "model.par"
    parameters.write_text("model=kharma\nouter_radius=100\n")
    output = tmp_path / "image.h5"
    command = [sys.executable, str(ROOT / "scripts/run_kpolaris_ddc_window.py"),
        f"--manifest={manifest}", f"--kpolaris-binary={tmp_path / 'unused'}",
        f"--parameter-file={parameters}", "--observation-time=3", "--fluid-time-min=1",
        "--fluid-time-max=3", f"--output={output}", f"--transport={transport}",
        f"--native-predictor-library={library}", "--dry-run"]
    if owned:
        command.append(f"--server={ROOT / 'scripts/serve_ddc_native.py'}")
    result = subprocess.run(command, capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stderr
    if success:
        report = json.loads(Path(str(output) + ".ddc-run.json").read_text())
        assert report["native_predictor_library"] == str(library)
        assert report["native_predictor_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
        position = report["server_command"].index("--native-predictor-library")
        assert report["server_command"][position + 1] == str(library)
    else:
        assert "owned native" in result.stderr
