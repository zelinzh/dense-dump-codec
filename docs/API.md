# Python reads

```python
from dense_dump_codec import DenseSequenceIndex
from dense_dump_codec.native import NativeSequenceDecoder

manifest = "demo/ddc/sequence_manifest.json"
index = DenseSequenceIndex.from_manifest(manifest)
bracket = index.bracket(1.35)
decoder = NativeSequenceDecoder(manifest, workers=4, maximum_batch_frames=25)
frames = decoder([bracket.lower.sequence, bracket.upper.sequence])
lower = frames[bracket.lower.sequence]["datasets"]["prims.rho"]
upper = frames[bracket.upper.sequence]["datasets"]["prims.rho"]
weight = bracket.upper_weight
interpolated_density = (1.0 - weight) * lower + weight * upper
```

The index brackets actual physical times and rejects extrapolation. Returned
arrays are float32 in native block order. Exact-anchor arrays can be shared and
read-only; copy them before mutation. The example interpolates primitives in
ordinary space at a requested time. This is different from the encoder's
log-space endpoint prediction for rho/u. The decoder itself returns calculated
frame times; radiative-transfer time interpolation belongs to the GRRT client.

`decoder(sequences, on_frame=callback)` publishes frames as they become ready
instead of retaining the entire batch in the return mapping. A decoder
serializes calls; bound requested batches by `maximum_batch_frames`.

For radial-prefix reads, use `radial_max=...`, `reconstruction="numpy"` and a
prepared `working_cache=...`. The spatial cache enables radial-slab I/O;
without it, cropping reduces returned arrays after archive decoding.
The supported ROI layout is a single-radial-block logarithmic grid.
The region includes a grid halo and
aligns to spatial tiles. The ROI must cover the client's entire radiative region.

Build the optional C prediction kernel with
`ddc build-predictor --output /scratch/predict.so`
and pass `predictor_library=...`. This path uses separate float32 operations and
does not enable fast-math. The C++/CUDA SDK is in
`src/dense_dump_codec/include/`; it requires a compatible external consumer.
