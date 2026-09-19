"""Losslessly transcode indexed DDC chunks for temporary read-heavy workloads."""

from __future__ import annotations

import bz2
import hashlib
import json
import tempfile
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .archive import (
    CHANNEL_BZIP2_FORMATS, CHANNEL_DEFLATE_FORMAT, CONTAINER_MEMBER, INDEX_MEMBER,
    _zip_info, open_archive,
)


def transcode_working_archive(source, output, *, level=1, workers=4):
    """Change only entropy compression, retaining every logical CRC and numeric bit.

    Temporal transforms and chunk boundaries are retained. The new file is fully
    validated before atomic publication. Existing outputs are never replaced.
    """
    source, output = Path(source), Path(output)
    if not 0 <= level <= 9 or workers < 1:
        raise ValueError("Deflate level must be 0..9 and workers must be positive")
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        container = json.loads(archive.read(CONTAINER_MEMBER))
        index = json.loads(archive.read(INDEX_MEMBER))
        if container["format"] not in CHANNEL_BZIP2_FORMATS:
            raise ValueError("Working transcoding requires a channel-bzip2 source")
        if index["format"] != container["format"]:
            raise ValueError("Source index/container format mismatch")
        direct = {name: archive.read(name) for name, row in index["members"].items()
                  if row["storage"] == "direct"}
    with tempfile.TemporaryDirectory(prefix=".ddc_working_", dir=output.parent) as temporary:
        temporary = Path(temporary)

        def convert(chunk):
            with zipfile.ZipFile(source) as archive:
                payload = bz2.decompress(archive.read(chunk["name"]))
            compressed = zlib.compress(payload, level)
            name = chunk["name"].removesuffix(".bz2") + ".deflate"
            path = temporary / Path(name).name
            path.write_bytes(compressed)
            return chunk["name"], name, path, len(compressed)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(convert, container["chunks"]))
        mapping = {old: new for old, new, _, _ in rows}
        for chunk, (_, name, _, size) in zip(container["chunks"], rows):
            chunk.update(name=name, compressed_size=size)
        for row in index["members"].values():
            if row["storage"] != "direct":
                row["chunk"] = mapping[row["chunk"]]
        container.update(format=CHANNEL_DEFLATE_FORMAT, compression="deflate",
                         compression_level=level)
        index["format"] = CHANNEL_DEFLATE_FORMAT
        candidate = temporary / "validated.ddc"
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, value in ((CONTAINER_MEMBER, container), (INDEX_MEMBER, index)):
                archive.writestr(_zip_info(name), json.dumps(value, sort_keys=True) + "\n")
            for name, value in direct.items():
                archive.writestr(_zip_info(name), value)
            for _, name, path, _ in rows:
                archive.write(path, name)
        with open_archive(candidate, cache_chunks=0) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Working archive failed CRC validation: {bad}")
        if output.exists():
            raise FileExistsError(output)
        output.hardlink_to(candidate)
        candidate.unlink()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for payload in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(payload)
    return dict(source=str(source), output=str(output), source_sha256=digest.hexdigest(),
                source_bytes=source.stat().st_size, output_bytes=output.stat().st_size,
                level=level, workers=workers, elapsed_seconds=time.monotonic() - started,
                logical_crcs_verified=True, re_quantized=False)
