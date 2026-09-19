import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['float32', 'pinned', 'compact'])
def test_transfer_mode_is_explicit_in_provenance(tmp_path, mode):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'frames': [dict(sequence=index,time=float(index),
        filename=f'frame{index}') for index in range(5)]}))
    binary = tmp_path / 'binary'
    binary.write_bytes(b'test')
    parameter = tmp_path / 'parameters'
    parameter.write_text('model=kharma\n')
    output = tmp_path / 'image.h5'
    command = [sys.executable, str(ROOT / 'scripts/run_kpolaris_ddc_window.py'),
        f'--manifest={manifest}',f'--kpolaris-binary={binary}',f'--parameter-file={parameter}',
        '--observation-time=3','--fluid-time-min=1','--fluid-time-max=3',f'--output={output}',
        '--dry-run','--prefetch-files=3',f'--native-transfer-mode={mode}',
        f'--server={ROOT / "scripts/serve_ddc_native.py"}']
    if mode == 'compact':
        command += [f'--native-working-cache={tmp_path / "working"}']
    result = subprocess.run(command,capture_output=True,text=True,
        env=dict(os.environ,KPOLARIS_DDC_PINNED='0',KPOLARIS_DDC_COMPACT='0'))
    assert result.returncode == 0, result.stderr
    report = json.loads(Path(str(output)+'.ddc-run.json').read_text())
    assert report['native_transfer_mode'] == mode
    environment = report['adapter_environment']
    assert environment['KPOLARIS_DDC_COMPACT'] == str(int(mode=='compact'))
    assert environment['KPOLARIS_DDC_PINNED'] == str(int(mode=='pinned'))
