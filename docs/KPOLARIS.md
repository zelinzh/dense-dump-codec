# KPolaris integration

Use a KPolaris revision supporting native DDC, `STAGEQ1`, local tile scales
and decoupled slow-light integration.

## Build and working cache

For KPolaris compact GPU reconstruction, configure its Kokkos CUDA build with:

```text
-DKPOLARIS_ENABLE_DDC_COMPACT_CUDA=ON
-DKPOLARIS_DDC_CODEC_INCLUDE_DIR=/path/to/ddc/src/dense_dump_codec/include
```

Set the target GPU architecture and CUDA/HDF5 environment in the KPolaris
build. Native float32 decoding on the CPU is also supported.

Compact decoding requires a prepared local LZ4 working cache. Create it with
`ddc prepare-cache`; see [working caches](WORKFLOW.md#working-caches).

## Run a slow-light calculation

From a source checkout, adapt all times and paths below to a supported fluid
window and a calibrated model parameter file:

```bash
ddc run-kpolaris --manifest /data/ddc/sequence_manifest.json \
  --kpolaris-binary /path/to/kpolaris_model_image_kharma \
  --parameter-file /data/camera_and_physics.par \
  --observation-time 26300 --fluid-time-min 26000 --fluid-time-max 26300 \
  --output /data/images/image.h5 --transport native \
  --server scripts/serve_ddc_native.py \
  --maximum-cache-files 50 --prefetch-files 25 --native-decode-workers 8 \
  --native-working-cache /scratch/ddc-work \
  --native-radius-max auto --native-transfer-mode compact
```

Set the fluid-time range to cover the geodesics used by the calculation. The adapter
derives the data ROI from the effective KPolaris radiative outer radius and
verifies that it covers that radius. The native `--server` is explicit; the
older compatibility server is also included for file-based readers.
For a wheel installation, locate the server with:

```bash
python -c 'from dense_dump_codec.runtime import runtime_project; print(runtime_project()/"scripts/serve_ddc_native.py")'
```

In the tested decoupled KPolaris line, use
`slow_light_step_mode=decoupled` and `slow_light_interpolation=fluid`.
The old `continuous` name is an alias in that line. Cache windows and prefetch
sizes control data residency, not the physical integration segmentation.

Choose image batches and GPU cache sizes in KPolaris. Multiple images can
share overlapping fluid windows and amortize data loading. Measure cache
preparation separately from image generation. CPU and GPU reconstruction may
differ at floating-point roundoff level because of their exponential operations.
