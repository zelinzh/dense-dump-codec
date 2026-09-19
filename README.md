# Dense Dump Codec

[Format](docs/FORMAT.md) · [KPolaris integration](docs/KPOLARIS.md) · [中文使用说明](README.zh.md)

DDC stores a dense simulation time series as **original exact anchors plus
compressed, quantized residuals for the intervening calculated states**. It
supports random access without decoding the entire preceding sequence.

The numerical method is independent of a particular simulator. The file and
native-array adapters in this distribution currently target KHARMA/Parthenon
PHDF output. An adapter is needed for other layouts.

## Key features

- **Temporal compression:** predict intermediate states from exact anchors
  using physical time, then quantize and compress their residuals.
- **Channel-aware precision:** use 8/8/5/16 bits for density, internal energy,
  velocity and magnetic-field residuals, with local internal-energy scales.
- **Random access:** restore individual states as PHDF files or float32 arrays
  without decoding the full sequence.
- **Streaming:** encode completed groups of states during a simulation,
  resume interrupted work and verify archive integrity with CRC/SHA checks.
- **GRRT integration:** provide radial working caches, compact data transport
  and C++/CUDA reconstruction interfaces for KPolaris.
- **Validation:** compare reconstructed fields against dense originals and
  sparse-cadence interpolation.

## Installation

Linux and Python **3.11+** are the supported baseline. NumPy and h5py are required;
the archive compressors are in the Python standard library. Spatial working
caches additionally need the **system liblz4** (for example `liblz4-1` on Debian/Ubuntu).
Only optional C/CUDA acceleration needs a compiler or GPU.

```bash
git clone https://github.com/zelinzh/dense-dump-codec.git
cd dense-dump-codec
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The `dev` extra installs testing and packaging tools. For normal use without
these tools, install with `python -m pip install -e .` instead. The optional
installation check is:

```bash
python -m pytest -q
ddc --help
```

You can also build an sdist and wheel with `python -m build` and install the
wheel normally. Command helpers are included in the wheel; a source checkout
is not required for native decoding. Method plots require `pip install -e '.[figures]'`.

## Five-minute synthetic round trip

Run from the repository root, using an empty `demo/` directory:

**1. Create example inputs.** Generate 51 small synthetic PHDF states; no
external simulation data are needed.

```bash
python examples/make_synthetic_sequence.py --output-dir demo/raw
```

**2. Encode and check integrity.** Create two GOPs and retain three original
anchors. The verifier checks files and recorded hashes. Original inputs are retained.

```bash
ddc encode --segment-dir demo/raw --output-dir demo/ddc
ddc verify --manifest demo/ddc/sequence_manifest.json --sha256
```

**3. Restore a conventional file.** Decode state number 13 into PHDF, without
first decoding the whole sequence.

```bash
ddc decode --manifest demo/ddc/sequence_manifest.json --sequence 13 \
  --output-phdf demo/reconstructed.out0.00013.phdf
```

**4. Alternatively, read arrays directly (optional).** Read the same state in
memory and print its time and array information, without writing a PHDF file.

```bash
python examples/read_native_frame.py --manifest demo/ddc/sequence_manifest.json --sequence 13
```

**5. Measure reconstruction error (optional).** Compare DDC against the
synthetic originals and against linear interpolation from every 2nd, 5th and
10th original state. Save field-level statistics to JSON.
This step requires the retained dense original inputs.

```bash
ddc compare --truth-dir demo/raw --codec-manifest demo/ddc/sequence_manifest.json \
  --datasets prims.rho,prims.u,prims.uvec,prims.B \
  --raw-strides 2,5,10 --output-json demo/comparison.json
```

The timestamps are deliberately slightly nonuniform; prediction uses the
stored physical times, not frame numbers.

## Encode real data

```bash
ddc encode --segment-dir /data/run --output-dir /data/ddc \
  --profile local-u8 --keyframe-stride 25 --archive-workers 4
```

The input adapter selects `*.out0.<integer>.phdf`, reads `Info/Time`, and expects
the four `prims.*` datasets. Scalar layouts are `(blocks, phi, theta, radius)`;
vectors insert a component axis after `blocks`. The local-u8 tile dimensions
must divide the spatial dimensions. Use `--dataset-tile-shapes prims.u=4x8x16`
to select another tile shape.

`--profile local-u8` is the recommended preset. `legacy-u8` uses native-block u
scales; `custom` leaves the low-level defaults to the user. Explicit CLI options
override preset values. `configs/local_u8.json` records the preset for inspection;
it is not an extra automatically loaded configuration file.

Cadence follows the input sequence: twenty-five steps span approximately 2.5M
for a 0.1M input cadence. A complete archive consists of the manifest, GOP
containers and original anchors. Keep all three and maintain their recorded
path references when moving data. Encoding retains source files by default;
simulation restarts use separately retained checkpoints.

## Decode back to the original file format

Decoding needs the manifest, the relevant GOP containers and their original
anchors, at the recorded paths. Encoded intermediate originals are not needed.
For example, restore output sequence number 13:

```bash
ddc decode --manifest /data/ddc/sequence_manifest.json --sequence 13 \
  --output-phdf /data/restored/torus.out0.00013.phdf
```

- `--sequence` is the output number in the original filename, **not physical
  time** or necessarily a zero-based row index. The manifest's `frames` list
  records available `sequence` and `time` values.
- The output directory is created automatically. Existing outputs are rejected
  unless `--overwrite` is explicitly supplied. Use a separate output directory
  to protect originals and anchors.
- All encoded fields are restored by default: density, internal energy,
  velocity and magnetic field for the recommended profile.
- Intermediate fields are reconstructed with quantization error; original
  anchors are copied exactly.
- Intermediate files use an anchor as a structural template and update the
  encoded fields and physical time. Unencoded diagnostics retain template
  values, not values at the requested time. Cycle counts are interpolated from
  anchors. XDMF sidecars need to be generated separately.

See [batch restoration](docs/WORKFLOW.md#restore-conventional-files) for several
states or a complete sequence. Native readers can avoid expanding the sequence
back to files altogether.

## Streaming, random access and acceleration

See [the workflow guide](docs/WORKFLOW.md) for streaming, retention and cache
lifecycle; [the Python API guide](docs/API.md) for bracket and batch reads; and
[KPolaris integration](docs/KPOLARIS.md) for compact transport and radial ROI.
Use one writer per output directory.

## License

[BSD-3-Clause](LICENSE).

## Citation

If you use DDC in your research, please cite the software:

```bibtex
@software{zhang_dense_dump_codec,
  author  = {Zhang, Zelin},
  title   = {{Dense Dump Codec}},
  year    = {2026},
  version = {0.4.0rc1},
  url     = {https://github.com/zelinzh/dense-dump-codec}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).
