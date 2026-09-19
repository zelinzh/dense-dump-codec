# Format and numerical conventions

## What is retained

`sequence_manifest.json` (`dense_dump_codec_sequence_v1`) lists physical times,
sequence numbers, exact anchors, GOP paths, codec settings and integrity data.
Adjacent GOPs share one original exact anchor. The recommended profile retains
original anchors alongside the manifest and GOP archives.

GOP metadata (`ddc_endpoint_residual_v2`) specifies start/end files and times,
middle times and prediction weights, fields, bit allocation, percentiles and
optional `dataset_tile_shapes`. Missing tile metadata means the legacy
native-block/component scale. Codes and exception indices use native flattened
array order; tile scales contain only one float32 value per tile, not a full
per-cell scale array. Scalar spatial axes are `(phi, theta, r)`; a preceding
MeshBlock/component axis is not mixed into the percentile calculation.

Logical NPY members contain integer codes, float32 scales, and optional sparse
exception indices/float32 values. Signed q4 is nibble-packed. Other supported
nominal bit widths use int8 or int16 carriers; their nominal bits are **not**
their full on-disk storage after entropy coding. NPY object arrays are rejected.

## Prediction and quantization

Let `tau = (t - t_left)/(t_right - t_left)`. For rho and u,
`T(x) = ln(max(float32(x), 1e-30))`; the other channels use identity.
The predicted transformed state is `(1-tau)*T(left) + tau*T(right)`.
The residual is `r = T(original) - prediction`.

For signed b-bit codes, `qmax = 2**(b-1)-1`. Each block/component or tile uses
`s = max(percentile(abs(r), p)/qmax, 1e-30)`, cast to float32. Codes are rounded
to nearest using NumPy's `rint` (ties to even) and stored in the symmetric
range. With outlier preservation enabled, cells with `abs(r/s) > qmax` also
store the unquantized float32 residual and its native flattened index; this
value overrides the code on reconstruction. Exception values retain float32
precision; conversion to physical fields includes float32 and log/exp operations.

Ordinary cells have nominal transform-space quantization error at most `s/2`,
apart from floating-point arithmetic. This bound applies in transform space;
field-level errors can be measured with `ddc compare`.

## Container layer

The recommended backend groups each channel in chunks of up to five middle
frames in a ZIP64 container. Integer first/second temporal differences are
reversible; the adaptive backend retains the shorter candidate after zigzag
mapping, byte-plane ordering and bzip2. Other dtype-specific reversible paths
are implemented in `archive.py`. Chains restart locally at chunks.

Logical-member CRC and reconstructed-array checks detect corruption.
SHA-256 records identify retained files. The optional
`ddc verify --sha256` compares stored archive/anchor hashes where present and
emits computed hashes.

The optional `.ddcw` format (`ddc_spatial_working_lz4_v1`) is a regenerable,
source-identity-bound radial-slab working cache. Its storage is counted
separately from the retained archives.
Native `STAGE1` transports arrays; `STAGEQ1` transports compact quantized
payloads to a reconstruction client. Both use local Unix-socket services.
