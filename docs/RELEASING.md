# Release checklist

1. Update the version in `pyproject.toml`, `src/dense_dump_codec/__init__.py`
   and `CITATION.cff`, and describe changes in `CHANGELOG.md`.
2. Regenerate `SHA256SUMS` for the source files and run `sha256sum -c SHA256SUMS`.
3. In a fresh environment, install `.[dev]` and run `python -m pytest -q`.
   Install system liblz4 for spatial-cache tests and a C compiler for kernel tests.
4. Run the README encode, verify, decode and compare examples.
5. Build with `python -m build`. Install the wheel in a separate environment
   and run the examples outside the source checkout.
6. Check the GitHub Actions results, then tag the version and attach the
   source package and wheel to its release.
