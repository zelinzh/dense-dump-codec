# Validation

The software tests use small synthetic arrays and HDF5 files. They cover
physical-time interpolation, archive CRCs, corruption/failure paths, exact
anchor handling, local-scale exception remapping, random-access file/array
agreement, spatial working-cache reads, service protocols and watcher resume.
The generalized tile encoder is checked against an independently expressed
version of the tile formula.

The release also includes the existing field/cadence evaluator:

```bash
ddc compare --truth-dir /data/retained-truth \
  --codec-manifest /data/ddc/sequence_manifest.json \
  --datasets prims.rho,prims.u,prims.uvec,prims.B --raw-strides 2,5,10 \
  --output-json /data/comparison.json
```

Sparse references are subsampled from the same trajectory and interpolated
linearly in physical time and primitive space. A stride is a frame count, not a
hard-coded M interval. NRMSE is `sqrt(sum(error**2)/sum(truth**2))`; aggregate
errors accumulate sums before taking ratios, not means of per-frame ratios.
Reports include absolute errors and ratios.

For a paper-style rate--distortion summary, use
`R_storage = (DDC residual bytes + unique exact-anchor bytes) / sparse-original bytes`
and `R_worst = max_over_fields(NRMSE_DDC / NRMSE_sparse_interpolation)` on the
same evaluated cells and times. Record the field list, mask and evaluation
intervals. Retained storage includes residuals and each required anchor once;
working caches are counted separately.

Legacy derived field diagnostics are deliberately named *proxies*. They use
`velocity_sq = sum(uvec**2)`, `Bsq = sum(B**2)` in the stored coordinate components,
and natural logarithms of `max(Bsq/2, eps)`, `max(Bsq,eps)/max(rho,eps)`, and
`max(2*u,eps)/max(Bsq,eps)` (evaluated as differences of logs where applicable),
with `eps=1e-30`. They are not metric-contracted fluid-frame `b^2`, physical
magnetization or thermodynamic plasma beta. These diagnostics apply no funnel
mask. Consult the evaluator for the exact float operations.

## Method diagrams

Install the `figures` extra, then generate the method and GOP structure diagrams:

```bash
python scripts/plot_ddc_method_document.py --figure overview --output-dir output/figures
python scripts/plot_ddc_gop_structure.py --output output/figures/ddc_gop_structure.pdf
```

Other plot modes take measurement files as inputs; see the script's `--help`.
