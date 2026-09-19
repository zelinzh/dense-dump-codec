"""Installed command-line entry points for the curated DDC distribution."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from .runtime import runtime_project


COMMANDS = {
    "encode": "encode_dense_sequence.py",
    "decode": "decode_dense_sequence.py",
    "watch": "watch_dense_codec.py",
    "compare": "compare_dense_cadences.py",
    "prepare-cache": "prepare_ddc_working_cache.py",
    "serve": "serve_ddc_native.py",
    "run-kpolaris": "run_kpolaris_ddc_window.py",
    "build-predictor": "build_ddc_predict_kernel.py",
    "verify": "verify_ddc_sequence.py",
}


def profile_arguments(name):
    if name == "custom":
        return []
    arguments = [
        "--keyframe-stride", "25", "--datasets", "prims.rho,prims.u,prims.uvec,prims.B",
        "--bits", "8", "--dataset-bits", "prims.rho=8,prims.u=8,prims.uvec=5,prims.B=16",
        "--scale-mode", "block-channel", "--scale-percentile", "99.5",
        "--dataset-scale-percentiles", "prims.rho=99.9,prims.u=99.9,prims.uvec=99.5,prims.B=99.5",
        "--archive-backend", "channel-bzip2-adaptive", "--channel-chunk-frames", "5",
        "--channel-compression-level", "9", "--keyframe-backend", "raw",
    ]
    if name == "local-u8":
        arguments += ["--dataset-tile-shapes", "prims.u=8x16x32"]
    return arguments


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Dense Dump Codec: encode, read and validate dense states.")
    parser.add_argument("command", choices=tuple(COMMANDS))
    parser.add_argument("--version", action="version", version=__import__("dense_dump_codec").__version__)
    if not arguments or arguments[0].startswith("-"):
        parser.parse_args(arguments)
        return 0
    command = arguments.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    if command in {"encode", "watch"}:
        profile_parser = argparse.ArgumentParser(add_help=False)
        profile_parser.add_argument("--profile", choices=("local-u8", "legacy-u8", "custom"),
                                    default="local-u8")
        options, arguments = profile_parser.parse_known_args(arguments)
        if "--help" in arguments or "-h" in arguments:
            print("Preset: --profile {local-u8,legacy-u8,custom} (default: local-u8).\n"
                  "Explicit command options override preset values.", flush=True)
        if command == "encode" and not any(flag in arguments for flag in ("--help", "-h")):
            output_parser = argparse.ArgumentParser(add_help=False)
            output_parser.add_argument("--output-dir", type=Path, required=True)
            output, _ = output_parser.parse_known_args(arguments)
            if output.output_dir.exists() and any(output.output_dir.iterdir()):
                parser.error("encode requires an empty output directory; use watch to resume a stream")
        arguments = profile_arguments(options.profile) + arguments
    script = runtime_project() / "scripts" / COMMANDS[command]
    previous = sys.argv
    sys.argv = [str(script), *arguments]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
