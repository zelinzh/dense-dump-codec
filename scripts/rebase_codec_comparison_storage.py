#!/usr/bin/env python3
"""Attach verified lossless-repack storage to an existing codec comparison."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--repacked-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def rebase_storage(
    comparison: dict[str, Any],
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    repack = manifest.get("lossless_repack", {})
    if repack.get("member_crc_verified") is not True:
        raise ValueError("Repacked manifest lacks complete member CRC verification")
    schemes = manifest.get("codec_schemes", {})
    if len(schemes) != 1:
        raise ValueError("Repacked manifest must contain exactly one codec scheme")
    scheme_name, scheme = next(iter(schemes.items()))
    if scheme_name != comparison["codec_scheme"]:
        raise ValueError("Codec scheme mismatch")
    checks = (
        (int(scheme["bits"]), int(comparison["codec_bits"]), "bits"),
        (
            dict(scheme.get("dataset_bits", {})),
            dict(comparison.get("codec_dataset_bits", {})),
            "dataset bits",
        ),
        (
            dict(scheme.get("dataset_scale_percentiles", {})),
            dict(comparison.get("codec_dataset_scale_percentiles", {})),
            "dataset scale percentiles",
        ),
        (
            int(manifest["keyframe_stride"]),
            int(comparison["codec_keyframe_stride"]),
            "keyframe stride",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"Codec {label} mismatch")
    old_storage = comparison["codec_storage"]
    if int(repack["source_archive_bytes"]) != int(old_storage["archive_bytes"]):
        raise ValueError("Source archive bytes do not match the comparison")
    keyframe_bytes = int(manifest["storage"]["keyframe_bytes"])
    if keyframe_bytes != int(old_storage["keyframe_bytes"]):
        raise ValueError("Keyframe bytes do not match the comparison")
    archive_bytes = int(scheme["archive_size_bytes"])
    total_bytes = keyframe_bytes + archive_bytes
    backend = str(scheme["archive_backend"])
    result = copy.deepcopy(comparison)
    result["codec_archive_backend"] = backend
    result["codec_storage"] = {
        "archive_bytes": archive_bytes,
        "keyframe_bytes": keyframe_bytes,
        "total_bytes": total_bytes,
        "ratio_vs_dense_phdf": int(comparison["truth_bytes"])
        / max(total_bytes, 1),
    }
    result["codec_lossless_repack"] = {
        "manifest": str(manifest_path.resolve()),
        "backend": backend,
        "chunk_frames": int(repack["chunk_frames"]),
        "compression_level": int(repack["compression_level"]),
        "member_crc_verified": True,
        "source_archive_bytes": int(repack["source_archive_bytes"]),
        "output_archive_bytes": archive_bytes,
        "archive_saving_fraction": float(repack["archive_saving_fraction"]),
        "source_total_bytes": int(old_storage["total_bytes"]),
        "output_total_bytes": total_bytes,
        "total_saving_fraction": 1.0
        - total_bytes / int(old_storage["total_bytes"]),
        "error_metrics_reused_because_payload_members_are_byte_identical": True,
    }
    return result


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    manifest = json.loads(args.repacked_manifest.read_text(encoding="utf-8"))
    result = rebase_storage(
        comparison,
        manifest,
        manifest_path=args.repacked_manifest,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"archive_saving={result['codec_lossless_repack']['archive_saving_fraction']:.3%} "
        f"total_saving={result['codec_lossless_repack']['total_saving_fraction']:.3%}"
    )
    print(f"comparison: {args.output_json}")


if __name__ == "__main__":
    main()
