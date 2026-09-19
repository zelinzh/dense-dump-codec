# Encoding, retention and cache lifecycle

## Recommended workflow

1. Retain a representative original truth window for validation.
2. Encode into a new directory using `ddc encode --profile local-u8`.
3. Run `ddc verify --manifest ... --sha256` and decode sample frames.
4. Compare against original states and sparse interpolation with `ddc compare`.
5. Keep original exact anchors and independent restart checkpoints.
6. Only after validation, decide whether to remove intermediate original files.

`ddc encode` requires an empty output directory. Additional low-level options
are listed by each script's `--help`. The verifier currently
targets the recommended external-original-anchor mode; optional legacy packed
anchors have their own validation helpers.

## Restore conventional files

Use `ddc decode` for one output number (not a physical time). It restores all
encoded fields by default, writing PHDF in the original structural layout.
Intermediate states remain lossy reconstructions; original anchors are copied
exactly. Unencoded fields retain values from the anchor template rather than
the restored time. Simulation restarts use independently retained checkpoints.

```bash
ddc decode --manifest demo/ddc/sequence_manifest.json --sequence 13 \
  --output-phdf demo/restored/torus.out0.00013.phdf
```

For a small range of existing, consecutive output numbers, use a shell loop.
This example works with the 51-state synthetic sequence in the README:

```bash
for sequence in $(seq 10 15); do
  output=$(printf 'demo/restored-range/torus.out0.%05d.phdf' "$sequence")
  ddc decode --manifest demo/ddc/sequence_manifest.json \
    --sequence "$sequence" --output-phdf "$output" || break
done
```

For a complete sequence, iterate over the manifest rather than assuming its
first number or cadence. Run the following from the repository root after
the README example, with the environment containing `ddc` activated:

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path

manifest = Path("demo/ddc/sequence_manifest.json")
output_dir = Path("demo/restored-all")
for frame in json.loads(manifest.read_text())["frames"]:
    sequence = int(frame["sequence"])
    output = output_dir / f"torus.out0.{sequence:05d}.phdf"
    subprocess.run([
        "ddc", "decode", "--manifest", str(manifest),
        "--sequence", str(sequence), "--output-phdf", str(output),
    ], check=True)
PY
```

The `torus` prefix above is a chosen output name, not a requirement of the
decoder. Parent directories are created automatically. These examples refuse
existing outputs; use a fresh destination when repeating them. Do not direct
output onto required anchors or truth data. Large restorations require space
comparable to the original dense sequence.

The loops favor simplicity, not maximum throughput: each `ddc decode` starts a
process and may decode a shared chunk again. For repeated analysis, prefer the
bounded native-array batches in [the API guide](API.md), without permanently
materializing PHDF. XDMF visualization sidecars are not created by the decoder.

## Streaming

`ddc watch` requires the planned frame count. It detects completed GOPs from
readable states and a following output; the final GOP must be stable. Resume
with the same input, output, and profile. Tile choices are part of the resume
configuration and must not change mid-sequence. At most one encoder should own
an output directory.

Default behavior preserves all source outputs. The advanced
`--delete-middle-frames` option is destructive and must be explicitly requested;
keep backups and a truth window before using it. Checkpoint pruning is disabled
by default (`--checkpoint-keep 0`).

Retained-byte statistics count archives and required anchors, **not** transient
dense files, restart checkpoints, working caches, or downstream images.
Use the same field selection in the original and compressed data when
comparing storage.

## Working caches

Optional spatial working files reorganize archive members into LZ4 radial
slabs without changing quantized values. This sacrifices some disk space to
avoid repeated bzip2 decompression of irrelevant cells during GRRT.

```bash
ddc prepare-cache --manifest /data/ddc/sequence_manifest.json \
  --output-dir /scratch/ddc-work --sequence-min 0 --sequence-max 1000 \
  --workers 8 --slab-cells 32 --minimum-free-gib 40
```

The selected range is in sequence numbers, not physical time. Preparation
operates on whole selected GOPs. Keep enough space for a GOP in addition to the
free-space threshold. The source archive is unchanged. Cache validation binds
to the source identity (including local filesystem metadata); copying a cache
between machines/mounts can require regeneration.

Cache files (`*.ddcw`) and staged frame caches can be removed **when no reader
uses them**. Original anchors, `.ddc` files and manifests are retained data, not
caches. Archive location references currently use recorded paths: moving a
dataset requires maintaining those paths.

## Choosing resources

Encoding and archive decompression are CPU operations; CUDA is optional only
for compact reconstruction inside a compatible GRRT client. More workers can
increase peak RAM and compete for storage bandwidth. Begin with 4--8 workers,
measure actual throughput, and avoid simultaneous dense full-frame batches
that exceed host memory. GPU cache budgets are not total GPU memory limits;
the integration and image buffers also consume memory.
