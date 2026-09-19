import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('mode', ['float32', 'pinned', 'compact'])
def test_service_receives_the_same_transfer_environment(tmp_path, monkeypatch, mode):
    path = ROOT / 'scripts/run_kpolaris_ddc_window.py'
    spec = importlib.util.spec_from_file_location('ddc_launcher_env_test', path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'frames': [dict(sequence=index, time=float(index), filename=f'frame{index}') for index in range(5)]}))
    binary = tmp_path / 'binary'
    binary.write_bytes(b'test')
    parameters = tmp_path / 'parameters'
    parameters.write_text('model=kharma\n')
    arguments = [str(path), f'--manifest={manifest}', f'--kpolaris-binary={binary}', f'--parameter-file={parameters}', '--observation-time=3', '--fluid-time-min=1', '--fluid-time-max=3', f'--output={tmp_path / "image.h5"}', '--prefetch-files=3', f'--native-transfer-mode={mode}', f'--server={ROOT / "scripts/serve_ddc_native.py"}']
    if mode == 'compact':
        arguments.append(f'--native-working-cache={tmp_path / "working"}')
    monkeypatch.setattr(sys, 'argv', arguments)
    monkeypatch.setenv('KPOLARIS_DDC_COMPACT', '0' if mode == 'compact' else '1')
    captured = {}
    def stop_after_spawn(command, **kwargs):
        captured.update(kwargs)
        raise RuntimeError('captured-service-spawn')
    monkeypatch.setattr(launcher.subprocess, 'Popen', stop_after_spawn)
    with pytest.raises(RuntimeError, match='captured-service-spawn'):
        launcher.main()
    assert captured['env']['KPOLARIS_DDC_COMPACT'] == str(int(mode == 'compact'))
    assert captured['env']['KPOLARIS_DDC_PINNED'] == str(int(mode == 'pinned'))
