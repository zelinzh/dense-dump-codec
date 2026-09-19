#!/usr/bin/env python3
"""Build the optional local CPU prediction kernel without fast-math or FMA contraction."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--native-cpu", action="store_true")
    args = parser.parse_args()
    import dense_dump_codec
    source = Path(dense_dump_codec.__file__).with_name("predict_kernel.c")
    output = args.output.resolve()
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c11", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off"]
    if args.native_cpu:
        flags.append("-march=native")
    with tempfile.TemporaryDirectory(prefix=".ddc_predict_", dir=output.parent) as temporary:
        candidate = Path(temporary) / "predict.so"
        command = [args.compiler, *flags, str(source), "-o", str(candidate)]
        subprocess.run(command, check=True)
        report = dict(command=command, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      library_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                      native_cpu=args.native_cpu, abi=1, machine=os.uname().machine,
                      compiler=subprocess.check_output([args.compiler, "--version"], text=True))
        output.hardlink_to(candidate)
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
