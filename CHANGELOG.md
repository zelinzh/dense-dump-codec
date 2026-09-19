# Changelog

## 0.4.0rc1

- Promote the tested local-u8 tile formula into the offline and streaming
  encoding interfaces; record tile shapes in metadata and resume configuration.
- Ensure conventional-file decoding understands the same compact tile scales
  as the native-array and compact transport paths.
- Add the installed `ddc` command, a local-u8 preset, synthetic round trip and
  source/wheel packaging of command helpers.
- Retain native arrays, ROI, LZ4 working caches, C/CUDA headers and compatibility
  backends; document their optional dependencies and limits.
- Add source provenance, checksums, CI and release instructions.
