"""Resolve explicit KPolaris radiation settings with its file-then-CLI precedence."""

import math
import shlex
from pathlib import Path


FILE_KEYS = {"parameter_file", "input", "input_file", "params", "config"}


def normalize_key(key):
    key = key.strip()
    while key.startswith("--"):
        key = key[2:]
    return key.replace("-", "_")


def read_parameters(path):
    output = {}
    for line in Path(path).read_text().splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        parts = body.split("=", 1) if "=" in body else body.split()
        if len(parts) != 2:
            raise ValueError(f"Cannot resolve KPolaris parameter line: {body}")
        key, value = map(str.strip, parts)
        output[normalize_key(key)] = value
    return output


def effective_explicit_parameters(parameter_file, extra_args):
    command = []
    for argument in extra_args:
        if not argument.startswith("--") or "=" not in argument:
            raise ValueError("KPolaris options must use --key=value")
        key, value = argument.split("=", 1)
        command.append((normalize_key(key), value))
    files = [Path(parameter_file)] + [Path(value) for key, value in command if key in FILE_KEYS]
    values = {}
    for path in files:
        values.update(read_parameters(path))
    values.update(command)
    return values


def radial_request(value):
    return "auto" if value == "auto" else float(value)


def batch_output_paths(parameter_file, extra_args):
    values = effective_explicit_parameters(parameter_file, extra_args)
    path = values.get("slow_light_batch_jobs", "")
    if not path:
        return []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as error:
        raise ValueError("Cannot verify the KPolaris batch job list") from error
    outputs = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = shlex.split(line, comments=False)
        if (len(fields) != 4 or not math.isfinite(float(fields[0]))
                or not 0 <= int(fields[1]) < int(fields[2]) or not fields[3]):
            raise ValueError("Expected shared-physics batch rows: time first last output")
        outputs.append(Path(fields[3]).resolve())
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("Empty or duplicate KPolaris batch destinations")
    return outputs


def automatic_radius(parameter_file, extra_args):
    values = effective_explicit_parameters(parameter_file, extra_args)
    if values.get("slow_light_batch_jobs", ""):
        batch_output_paths(parameter_file, extra_args)
    radius = float(values.get("outer_radius", -1))
    if not math.isfinite(radius):
        raise ValueError("KPolaris outer_radius is nonfinite")
    return radius if radius > 0 else None


def verify_effective_radius(path, radius):
    if radius is None:
        return
    parameters = read_parameters(path)
    effective = float(parameters["outer_radius"])
    if not math.isfinite(effective) or not 0 < effective <= radius:
        raise ValueError("KPolaris effective radiation radius exceeds the served DDC region")
