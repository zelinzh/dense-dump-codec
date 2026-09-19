import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dense_dump_codec import (
    dequantize_residual,
    inverse_transformed,
    linear_predictor,
    pack_int4,
    quantize_residual,
    transformed,
    unpack_int4,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compare_dense_cadences import evaluate_raw_stride, validate_comparison_inputs
from encode_dense_sequence import (
    aggregate_error_rows,
    build_gop_ranges,
    sha256_file,
    wait_for_streaming_manifest,
)
from prototype_dump_codec import (
    dataset_bits_for_scheme,
    dataset_scale_percentiles_for_scheme,
    mixed_percentile_tag,
    mixed_scheme_tag,
    parse_dataset_bits,
    parse_dataset_scale_percentiles,
    phdf_time,
)


def test_log_transform_and_endpoint_predictor_preserve_positive_fields() -> None:
    start = np.asarray([1.0, 4.0, 16.0], dtype=np.float32)
    end = np.asarray([9.0, 36.0, 144.0], dtype=np.float32)

    predicted, predicted_transformed, start_transformed = linear_predictor(
        start,
        end,
        0.5,
        "prims.rho",
    )

    np.testing.assert_allclose(predicted, np.sqrt(start * end), rtol=2.0e-6)
    np.testing.assert_allclose(inverse_transformed(start_transformed, "prims.rho"), start)
    np.testing.assert_allclose(
        inverse_transformed(predicted_transformed, "prims.rho"),
        predicted,
    )
    assert np.all(predicted > 0.0)


def test_q8_block_channel_quantization_obeys_half_scale_error_bound() -> None:
    rng = np.random.default_rng(17)
    residual = rng.normal(size=(2, 3, 5, 6, 7)).astype(np.float32)

    payload = quantize_residual(residual, 8, "block-channel")
    reconstructed = dequantize_residual(payload)

    assert payload.scale.shape == (2, 3, 1, 1, 1)
    assert payload.values.dtype == np.int8
    assert payload.exception_count == 0
    assert np.all(np.abs(reconstructed - residual) < payload.scale / 2.0 + 1.0e-6)


def test_percentile_quantization_preserves_sparse_outliers_exactly() -> None:
    residual = np.linspace(-1.0, 1.0, 2000, dtype=np.float32).reshape(2, 1, 10, 10, 10)
    residual[0, 0, 0, 0, 0] = -1000.0
    residual[1, 0, 9, 9, 9] = 2000.0

    robust = quantize_residual(
        residual,
        6,
        "block-channel",
        scale_percentile=99.0,
        preserve_outliers=True,
    )
    max_scaled = quantize_residual(residual, 6, "block-channel")
    reconstructed = dequantize_residual(robust)

    assert robust.exception_count > 0
    assert robust.scale.max() < max_scaled.scale.max() / 100.0
    flat_reconstructed = reconstructed.reshape(-1)
    flat_target = residual.reshape(-1)
    np.testing.assert_array_equal(
        flat_reconstructed[robust.exception_indices.astype(np.int64)],
        flat_target[robust.exception_indices.astype(np.int64)],
    )


def test_discarded_outliers_are_clipped() -> None:
    residual = np.concatenate(
        [np.linspace(-1.0, 1.0, 999, dtype=np.float32), np.asarray([1000.0], dtype=np.float32)]
    )
    payload = quantize_residual(
        residual,
        8,
        "frame",
        scale_percentile=99.0,
        preserve_outliers=False,
    )
    reconstructed = dequantize_residual(payload)

    assert payload.exception_count == 0
    assert reconstructed[-1] < 2.0


def test_quantization_rejects_nonfinite_residuals() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        quantize_residual(
            np.asarray([0.0, np.nan, np.inf], dtype=np.float32),
            8,
            "frame",
        )


@pytest.mark.parametrize("shape", [(7,), (2, 3), (2, 2, 3)])
def test_signed_int4_packing_roundtrips_odd_and_even_shapes(shape: tuple[int, ...]) -> None:
    size = int(np.prod(shape))
    values = ((np.arange(size) % 15) - 7).astype(np.int8).reshape(shape)

    packed = pack_int4(values)
    unpacked = unpack_int4(packed, shape)

    np.testing.assert_array_equal(unpacked, values)


def test_int4_packing_rejects_unrepresentable_value() -> None:
    with pytest.raises(ValueError, match=r"\[-7, 7\]"):
        pack_int4(np.asarray([-8, 0, 7], dtype=np.int8))


def test_transform_clamps_nonpositive_density() -> None:
    values = transformed(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32), "prims.u")

    assert np.isfinite(values).all()
    assert values[0] == values[1]


def test_build_gop_ranges_covers_sequence_with_shared_keyframes() -> None:
    assert build_gop_ranges(51, 10, False) == [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
    ]
    assert build_gop_ranges(13, 10, True) == [(0, 10), (10, 12)]

    with pytest.raises(ValueError, match="complete"):
        build_gop_ranges(13, 10, False)


def test_dataset_bit_overrides_create_stable_mixed_scheme() -> None:
    datasets = ("prims.rho", "prims.u", "prims.B")
    overrides = parse_dataset_bits("prims.rho=7,prims.u=7,prims.B=10", datasets)
    allocation = dataset_bits_for_scheme(datasets, 6, overrides)

    assert allocation == {"prims.rho": 7, "prims.u": 7, "prims.B": 10}
    assert mixed_scheme_tag(6, allocation) == "q6_rho7_u7_B10"
    with pytest.raises(ValueError, match="unknown dataset"):
        parse_dataset_bits("divB=6", datasets)


def test_dataset_percentile_overrides_create_stable_mixed_scheme() -> None:
    datasets = ("prims.rho", "prims.B")
    overrides = parse_dataset_scale_percentiles("prims.B=99.5", datasets)
    allocation = dataset_scale_percentiles_for_scheme(datasets, 99.9, overrides)

    assert allocation == {"prims.rho": 99.9, "prims.B": 99.5}
    assert mixed_percentile_tag(99.9, allocation) == "_Bp99p5"
    with pytest.raises(ValueError, match="unknown dataset"):
        parse_dataset_scale_percentiles("divB=99.5", datasets)


def test_aggregate_error_rows_reconstructs_global_norms() -> None:
    rows = [
        {
            "count": 2,
            "sse": 2.0,
            "target_sse": 8.0,
            "sae": 2.0,
            "target_abs": 4.0,
            "max_abs_error": 1.0,
            "min_reconstructed_value": -1.0,
        },
        {
            "count": 2,
            "sse": 6.0,
            "target_sse": 24.0,
            "sae": 3.0,
            "target_abs": 6.0,
            "max_abs_error": 2.0,
            "min_reconstructed_value": -2.0,
        },
    ]

    aggregate = aggregate_error_rows(rows)

    assert aggregate["rmse"] == pytest.approx(np.sqrt(2.0))
    assert aggregate["target_rms"] == pytest.approx(np.sqrt(8.0))
    assert aggregate["nrmse"] == pytest.approx(0.5)
    assert aggregate["rel_l1"] == pytest.approx(0.5)
    assert aggregate["max_abs_error"] == 2.0
    assert aggregate["min_reconstructed_value"] == -2.0


def test_raw_cadence_interpolation_uses_physical_phdf_time(tmp_path: Path) -> None:
    times = [0.0, 0.2, 1.0]
    values = [0.0, 2.0, 10.0]
    files = []
    for index, (frame_time, value) in enumerate(zip(times, values, strict=True)):
        path = tmp_path / f"tiny.out0.{index:05d}.phdf"
        with h5py.File(path, "w") as handle:
            handle.create_group("Info").attrs["Time"] = frame_time
            handle["prims.rho"] = np.asarray([[[value]]], dtype=np.float32)
        files.append(path)

    assert [phdf_time(path) for path in files] == times
    summary = evaluate_raw_stride(files, times, [1], 2, ("prims.rho",))

    assert summary["common_errors"]["rho"]["nrmse"] == 0.0


def test_cadence_comparison_rejects_incomplete_truth_or_codec_coverage() -> None:
    files = [Path(f"tiny.out0.{index:05d}.phdf") for index in range(3)]
    manifest = {
        "format": "dense_dump_codec_sequence_v1",
        "complete": True,
        "dense_frame_count": 3,
        "middle_frame_count": 1,
        "keyframes": [str(files[0]), str(files[2])],
    }
    frame_map = {1: (Path("gop.ddc"), 1)}

    validate_comparison_inputs(files, [0.0, 0.1, 0.2], manifest, frame_map)

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_comparison_inputs(files, [0.0, 0.2, 0.1], manifest, frame_map)

    with pytest.raises(ValueError, match="cover"):
        validate_comparison_inputs(
            files,
            [0.0, 0.1, 0.2],
            {**manifest, "middle_frame_count": 0},
            {},
        )


def test_sha256_file_streams_reproducible_digest(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"dense-dump-codec")

    assert sha256_file(path, chunk_size=3) == (
        "bcbfb1a18ff7dc81574469f659d503b3df096693276fd4a351e0e0a8ca2e0fb9"
    )


def test_batch_encoder_reuses_complete_streaming_manifest(tmp_path: Path) -> None:
    (tmp_path / "stream_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sequence_manifest.json").write_text(
        '{"format":"dense_dump_codec_sequence_v1","complete":true}',
        encoding="utf-8",
    )

    assert wait_for_streaming_manifest(tmp_path, 0.0)


def test_batch_encoder_rejects_stalled_streaming_state(tmp_path: Path) -> None:
    (tmp_path / "stream_state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(TimeoutError, match="did not finalize"):
        wait_for_streaming_manifest(tmp_path, 0.0)
