"""Public native-array decoder for bounded slow-light data access."""

from __future__ import annotations

import importlib
import sys
import tempfile
import threading
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from .reconstruction import Reconstruction
from .region import RadialRegion
from .spatial_cache import SpatialWorkingArchive, cache_path
from .runtime import runtime_project


NATIVE_API_VERSION = 1
NATIVE_DATASETS = ("prims.rho", "prims.u", "prims.uvec", "prims.B")


def native_capabilities():
    return {
        "api_version": NATIVE_API_VERSION,
        "per_frame_callback": True,
        "readonly_anchors": True,
        "compact_tile_scales": True,
        "radial_roi": "single_radial_meshblock_log_grid",
        "spatial_working_cache": "lz4_radial_slabs_v1",
        "reconstruction_modes": ["numpy", "reference"],
        "optional_prediction_kernel": "binary32_separate_operations_v1",
        "wire_protocol": "STAGE1",
    }


def _codec_modules(project):
    project = Path(project).resolve()
    script_directory = project / "scripts"
    if not (script_directory / "decode_dense_sequence.py").is_file():
        raise FileNotFoundError("Native decoding currently requires the DDC source scripts")
    sys.path.insert(0, str(script_directory))
    modules = tuple(importlib.import_module(name)
                    for name in ("decode_dense_sequence", "decode_dump_codec"))
    if any(not Path(module.__file__).resolve().is_relative_to(project) for module in modules):
        raise RuntimeError("A decoder from another DDC checkout is already imported")
    return modules


class NativeSequenceDecoder:
    """Decode complete frames with independent channel/chunk worker handles.

    A callback receives each completed frame immediately and the returned mapping
    is empty. Without a callback the caller owns all requested outputs. Calls on
    one decoder are serialized; applications should bound their requested batch.
    """

    datasets = NATIVE_DATASETS

    def __init__(self, manifest, *, workers=16, cache_chunks=8, scheme=None,
                 reconstruction="auto", maximum_batch_frames=100, codec_project=None,
                 radial_max=None, working_cache=None, predictor_library=None):
        if workers < 1 or cache_chunks < 0 or maximum_batch_frames < 1:
            raise ValueError("Invalid worker, chunk cache or batch limit")
        project = codec_project or runtime_project()
        self.sequence, self.codec = _codec_modules(project)
        self.manifest_path = Path(manifest)
        self.manifest = self.sequence.load_manifest(self.manifest_path)
        self.times = {int(row["sequence"]): float(row["time"])
                      for row in self.manifest["frames"]}
        self.keyframes = {self.sequence.phdf_sequence(Path(path)): Path(path)
                          for path in self.manifest["keyframes"]}
        self.workers, self.cache_chunks, self.scheme = workers, cache_chunks, scheme
        self.maximum_batch_frames = maximum_batch_frames
        self.reconstruction = Reconstruction(self.codec, reconstruction, predictor_library=predictor_library)
        if radial_max is not None and self.reconstruction.mode != "numpy":
            raise ValueError("Radial ROI requires numpy reconstruction")
        self.radial_max, self.region = radial_max, None
        self.working_cache = Path(working_cache) if working_cache is not None else None
        if self.working_cache is not None and self.reconstruction.mode != "numpy":
            raise ValueError("Spatial working cache requires numpy reconstruction")
        self._anchors = OrderedDict()
        self._lock = threading.Lock()
        self._access_lock = threading.Lock()
        self._access = Counter()

    @contextmanager
    def _open_archive(self, path, **kwargs):
        candidate = cache_path(path, self.working_cache) if self.working_cache is not None else None
        spatial = candidate is not None and candidate.is_file()
        handle = SpatialWorkingArchive(candidate, source=path) if spatial else self.sequence.open_archive(path, **kwargs)
        with handle as archive:
            try:
                yield archive
            finally:
                with self._access_lock:
                    self._access["spatial_archive_opens" if spatial else "original_archive_opens"] += 1
                    if spatial:
                        self._access["spatial_blocks_read"] += archive.blocks_read
                        self._access["spatial_compressed_payload_bytes"] += archive.compressed_bytes_read

    def access_statistics(self):
        with self._access_lock:
            return dict(self._access)

    @staticmethod
    def _anchor_key(path):
        stat = path.stat()
        return str(path.resolve()), stat.st_size, stat.st_mtime_ns

    def _read_anchor(self, path, *, persistent=True):
        key = self._anchor_key(path)
        if persistent and key in self._anchors:
            self._anchors.move_to_end(key)
            return self._anchors[key]
        with self.codec.require_h5py().File(path, "r") as handle:
            native = self.codec.read_kharma_native_metadata(handle)
            radial_stop = None
            if self.radial_max is not None:
                if (int(handle["Info"].attrs.get("IncludesGhost", 0))
                        or int(handle["Info"].attrs.get("Multilevel", 0))):
                    raise ValueError("Radial ROI does not support ghosts or AMR")
                region = RadialRegion.from_metadata(native, self.radial_max)
                if self.region is not None and self.region != region:
                    raise ValueError("Native radial grid changed within the sequence")
                self.region = region
                radial_stop = region.retained_cells
                self.reconstruction.radial_stop = radial_stop
                if any(handle[name].shape[-1] != region.source_cells for name in self.datasets):
                    raise ValueError("Native dataset and radial grid dimensions differ")
                native = region.metadata(native)
            raw = {name: np.asarray(handle[name][..., :radial_stop], dtype=np.float32)
                   for name in self.datasets}
        transformed = {name: self.codec.transformed(values, name)
                       for name, values in raw.items()}
        for values in (*raw.values(), *transformed.values(), native["block_order"]):
            values.flags.writeable = False
        result = native, raw, transformed
        if persistent:
            self._anchors[key] = result
            self._anchors.move_to_end(key)
            while len(self._anchors) > 4:
                self._anchors.popitem(last=False)
        return result

    def __call__(self, sequences, *, on_frame=None):
        requested = tuple(dict.fromkeys(map(int, sequences)))
        if not requested:
            return {}
        if len(requested) > self.maximum_batch_frames:
            raise ValueError("Requested frame batch exceeds maximum_batch_frames")
        if any(sequence not in self.times for sequence in requested):
            raise KeyError("Requested sequence is absent from the manifest")
        with self._lock:
            return self._decode(requested, on_frame)

    def _decode(self, requested, on_frame):
        results = {}
        with tempfile.TemporaryDirectory(prefix="ddc_native_anchors_") as temporary:
            temporary = Path(temporary)
            keyframe_cache, active = {}, set()

            def anchor(preferred):
                sequence = self.sequence.phdf_sequence(preferred)
                path = self.sequence.existing_raw_keyframe(self.manifest, sequence, preferred)
                persistent = path is not None
                if path is None:
                    path = temporary / f"keyframe_{sequence:05d}.phdf"
                    if not path.is_file():
                        self.sequence._materialize_keyframe(
                            self.manifest, sequence, path, overwrite=True,
                            temporary_path=temporary, cache=keyframe_cache, active=active
                        )
                return path, self._read_anchor(path, persistent=persistent)

            for sequence in requested:
                if sequence not in self.keyframes:
                    continue
                path, (native, raw, _) = anchor(self.keyframes[sequence])
                frame = {**native, "sequence": sequence, "time": self.times[sequence],
                         "datasets": dict(raw), "source_phdf": str(path),
                         "exact_keyframe": True}
                if on_frame is None:
                    results[sequence] = frame
                else:
                    on_frame(sequence, frame)
            middle = [sequence for sequence in requested if sequence not in self.keyframes]
            if not middle:
                return results
            scheme_name, scheme = self.sequence.select_scheme(self.manifest, self.scheme)
            groups = self.sequence.group_archive_frames(scheme["archive_paths"], middle)

            def tasks():
                for path, rows in groups.items():
                    metadata = self.sequence.read_archive_metadata_with_retry(path)
                    _, (native, start, start_transformed) = anchor(Path(metadata["start_file"]))
                    _, (_, end, end_transformed) = anchor(Path(metadata["end_file"]))
                    with self._open_archive(path, cache_chunks=0) as archive:
                        chunk_frames = max(1, int(getattr(archive, "container", {}).get(
                            "chunk_frames", 1)))
                        for frame_index, sequence in rows.items():
                            tau = self.codec.frame_tau(archive, metadata, self.datasets, frame_index)
                            results[sequence] = {
                                **native, "sequence": sequence, "time": self.times[sequence],
                                "tau": float(tau), "datasets": {}, "archive": str(path),
                                "frame_index": frame_index, "exact_keyframe": False,
                                "scheme": scheme_name,
                            }
                    chunks = defaultdict(list)
                    for frame_index in rows:
                        chunks[(frame_index - 1) // chunk_frames].append(frame_index)
                    for indices in chunks.values():
                        for dataset in self.datasets:
                            yield (path, metadata, rows, dataset, tuple(indices),
                                   start[dataset], end[dataset],
                                   start_transformed[dataset], end_transformed[dataset])

            def decode_task(task):
                path, metadata, rows, dataset, indices, start, end, start_t, end_t = task
                with self._open_archive(path, cache_chunks=self.cache_chunks) as archive:
                    decoded = [(rows[frame_index], np.ascontiguousarray(
                        self.reconstruction.decode(
                            archive, metadata, dataset, frame_index, start=start, end=end,
                            start_transformed=start_t, end_transformed=end_t
                        ), dtype="<f4")) for frame_index in indices]
                return dataset, decoded

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                iterator = iter(tasks())
                pending = set()

                def submit():
                    task = next(iterator, None)
                    if task is not None:
                        pending.add(pool.submit(decode_task, task))

                for _ in range(self.workers):
                    submit()
                try:
                    while pending:
                        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            dataset, decoded = future.result()
                            for sequence, values in decoded:
                                results[sequence]["datasets"][dataset] = values
                                if (on_frame is not None
                                        and len(results[sequence]["datasets"]) == len(self.datasets)):
                                    on_frame(sequence, results.pop(sequence))
                            submit()
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise
        return results
