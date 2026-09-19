#!/usr/bin/env python3
"""Check a sequence index, required anchors, archive checksums and optional hashes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dense_dump_codec import DenseSequenceIndex, open_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", action="store_true", help="Compute hashes of archives and raw anchors")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    index = DenseSequenceIndex.from_manifest(args.manifest)
    archives = sorted({path for scheme in manifest["codec_schemes"].values()
                       for path in scheme["archive_paths"]})
    anchors = manifest["keyframes"]
    if not all(Path(path).is_file() for path in anchors):
        raise FileNotFoundError("This verifier requires original exact anchors; a referenced file is missing")
    for path in archives:
        with open_archive(path) as archive:
            failure = archive.testzip()
            if failure:
                raise ValueError(f"Archive checksum failed: {path}: {failure}")
            metadata = json.loads(archive.read("metadata.json"))
            if not all(Path(metadata[key]).is_file() for key in ("start_file", "end_file")):
                raise FileNotFoundError(f"An archive anchor is missing: {path}")
    hashes = {}
    if args.sha256:
        for path in sorted(set(archives + anchors)):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                    digest.update(chunk)
            hashes[path] = digest.hexdigest()
        integrity = manifest.get("integrity", {})
        for row in integrity.get("archives", []) + integrity.get("keyframes", []):
            if row["path"] in hashes and hashes[row["path"]] != row["sha256"]:
                raise ValueError(f"Recorded SHA-256 mismatch: {row['path']}")
    print(json.dumps({"frames": len(index.frames), "archives_verified": len(archives),
                      "original_anchors_present": len(anchors), "complete": manifest.get("complete"),
                      "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
