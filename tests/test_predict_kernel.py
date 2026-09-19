import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dense_dump_codec.native import NativeSequenceDecoder
from dense_dump_codec.predict_kernel import FusedPredictor
from test_native_runtime import sequence_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    if not shutil.which("cc"):
        pytest.skip("Optional prediction kernel needs a C compiler")
    output = tmp_path_factory.mktemp("kernel") / "predict.so"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_ddc_predict_kernel.py"),
                    "--output", str(output)], check=True, capture_output=True)
    return output


@pytest.mark.parametrize("tau", [0, 1, 0.3125, 0.199982858152, -0.5, 1.5])
@pytest.mark.parametrize("size", [0, 1, 17, 32769])
def test_single_pass_preserves_float32_order(kernel, tau, size):
    generator = np.random.default_rng(8483)
    start, end, residual = [generator.normal(size=size).astype("f4") for _ in range(3)]
    expected = (start * (1.0 - tau) + end * tau) + residual
    actual = np.empty_like(expected)
    FusedPredictor(kernel)(start, end, residual, tau, actual)
    assert actual.tobytes() == expected.tobytes()


def test_subnormal_values_and_cancellation(kernel):
    start = np.array([0, -0.0, 1e-40, -1e-40, 1e10, 1e-30], dtype="f4")
    end = -start
    residual = np.array([-0.0, 0, -1e-40, 1e-40, 1, 0], dtype="f4")
    expected = (start * 0.5 + end * 0.5) + residual
    actual = np.empty_like(start)
    FusedPredictor(kernel)(start, end, residual, 0.5, actual)
    assert actual.tobytes() == expected.tobytes()


def test_native_sequence_roundtrip_and_reference_rejection(kernel, sequence_manifest):
    original = NativeSequenceDecoder(sequence_manifest)(range(9))
    actual = NativeSequenceDecoder(sequence_manifest, predictor_library=kernel)(range(9))
    for sequence, frame in original.items():
        for name, values in frame["datasets"].items():
            assert actual[sequence]["datasets"][name].tobytes() == values.tobytes()
    with pytest.raises(ValueError, match="numpy"):
        NativeSequenceDecoder(sequence_manifest, predictor_library=kernel, reconstruction="reference")


def test_kernel_rejects_aliases_layout_and_dtype(kernel):
    predictor = FusedPredictor(kernel)
    values = np.ones(10, dtype="f4")
    with pytest.raises(ValueError, match="independently"):
        predictor(values, values, values, 0.5, values)
    for invalid in (values.astype("f8"), values[::2]):
        with pytest.raises(ValueError, match="contiguous"):
            predictor(invalid, invalid, invalid, 0.5, np.empty_like(invalid))
    with pytest.raises(ValueError, match="finite"):
        predictor(values, values, values, np.nan, np.empty_like(values))


def test_build_no_overwrite_and_flags(kernel):
    record = json.loads(Path(str(kernel) + ".json").read_text())
    assert "-ffp-contract=off" in record["command"]
    assert "-fno-fast-math" in record["command"]
    process = subprocess.run([sys.executable, str(ROOT / "scripts/build_ddc_predict_kernel.py"),
                              "--output", str(kernel)], capture_output=True)
    assert process.returncode != 0


@pytest.mark.parametrize("bits", [4, 5, 7, 8, 16])
@pytest.mark.parametrize("dataset", ["prims.rho", "prims.u", "prims.uvec", "prims.B"])
@pytest.mark.parametrize("scale_dtype", [np.float32, np.float64])
def test_all_bit_profiles_preserve_reference(kernel, monkeypatch, bits, dataset, scale_dtype):
    from functools import partial
    import test_native_runtime as reference_tests
    from dense_dump_codec.reconstruction import Reconstruction
    monkeypatch.setattr(reference_tests, "Reconstruction",
                        partial(Reconstruction, predictor_library=kernel))
    reference_tests.test_reconstruction_is_bitwise_and_independently_owned(
        bits, dataset, scale_dtype)


@pytest.mark.parametrize("bits", [4, 5, 7, 8, 16])
@pytest.mark.parametrize("dataset", ["prims.rho", "prims.u", "prims.uvec", "prims.B"])
@pytest.mark.parametrize("compact", [False, True])
def test_strided_roi_falls_back_without_changing_values(kernel, monkeypatch, bits, dataset, compact):
    import test_native_radial_region as reference_tests
    from dense_dump_codec.reconstruction import Reconstruction

    def decoder(codec, mode="auto", **kwargs):
        return Reconstruction(codec, mode, predictor_library=None if mode == "reference" else kernel,
                              **kwargs)

    monkeypatch.setattr(reference_tests, "Reconstruction", decoder)
    reference_tests.test_roi_reconstruction_is_bitwise_subset(bits, dataset, compact)
