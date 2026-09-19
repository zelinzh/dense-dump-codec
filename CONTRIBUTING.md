# Contributing

Use Python 3.11+, install `.[dev]`, and run `python -m pytest -q`. Keep changes
small and add synthetic tests rather than embedding simulation files. Numerical
changes must preserve a named reference path or explicitly version the change.

When reporting a problem, provide version, profile, array layout, physical time
range, backend and a minimal synthetic reproducer. A GRRT performance report
must identify its external solver revision and distinguish cold/warm cache time.

Report field accuracy and archive integrity separately. Include all evaluated
cases when comparing profiles. Update format/API documentation whenever
compatibility changes.
