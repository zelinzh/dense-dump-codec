"""Opt-in compact transport preserving existing quantization and source CRCs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import struct

import numpy as np

from .native import NativeSequenceDecoder


class CompactSequenceDecoder(NativeSequenceDecoder):
    """Expose validated compact ROI slabs; no CPU floating-point reconstruction."""

    reconstruction_label = "gpu_float32"

    def __init__(self, *args, **kwargs):
        kwargs["predictor_library"] = None
        super().__init__(*args, **kwargs)

    def _decode(self, requested, on_frame):
        if self.working_cache is None:
            raise ValueError("Compact transport requires the spatial working cache")
        results = {}

        def anchor(preferred):
            sequence = self.sequence.phdf_sequence(Path(preferred))
            path = self.sequence.existing_raw_keyframe(self.manifest, sequence, Path(preferred))
            if path is None:
                raise ValueError("Compact transport currently requires retained original anchors")
            native, raw, transformed = self._read_anchor(path)
            return sequence, (native, raw, transformed)

        def publish(sequence):
            if on_frame is not None:
                on_frame(sequence, results.pop(sequence))

        for sequence in requested:
            if sequence in self.keyframes:
                _, value = anchor(self.keyframes[sequence])
                results[sequence] = {**value[0], 'sequence': sequence, 'time': self.times[sequence],
                    'exact_keyframe': True, 'tau': 0.0, 'start': sequence, 'end': sequence,
                    'anchors': {sequence: value}, 'channels': {}}
                publish(sequence)
        middle = [sequence for sequence in requested if sequence not in self.keyframes]
        _, scheme = self.sequence.select_scheme(self.manifest, self.scheme)
        groups = self.sequence.group_archive_frames(scheme['archive_paths'], middle)

        def extract(task):
            path, dataset, rows = task
            decoded = []
            with self._open_archive(path, cache_chunks=0) as archive:
                if not hasattr(archive, 'quantized_region'):
                    raise ValueError(f'Missing valid compact working archive for {path}')
                for frame_index, sequence in rows.items():
                    channel = archive.quantized_region(dataset, frame_index,
                        self.region.retained_cells if self.region else None)
                    codes, scales, indices, values, tiles = channel
                    if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
                        order = np.argsort(indices, kind="stable")
                        indices, values = indices[order], values[order]
                        if np.any(indices[1:] == indices[:-1]):
                            raise ValueError("Duplicate compact exception indices")
                    decoded.append((sequence, (codes, scales, indices, values, tiles)))
            return dataset, decoded

        tasks = []
        for path, rows in groups.items():
            metadata = self.sequence.read_archive_metadata_with_retry(path)
            start, first = anchor(metadata['start_file'])
            end, last = anchor(metadata['end_file'])
            with self._open_archive(path, cache_chunks=0) as archive:
                for frame_index, sequence in rows.items():
                    tau = self.codec.frame_tau(archive, metadata, self.datasets, frame_index)
                    results[sequence] = {**first[0], 'sequence': sequence, 'time': self.times[sequence],
                        'exact_keyframe': False, 'tau': float(tau), 'start': start, 'end': end,
                        'anchors': {start: first, end: last}, 'channels': {}}
            items = list(rows.items())
            for offset in range(0, len(items), 5):
                subset = dict(items[offset:offset+5])
                tasks.extend((path, dataset, subset) for dataset in self.datasets)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(extract, task) for task in tasks]
            for future in as_completed(futures):
                dataset, decoded = future.result()
                for sequence, channel in decoded:
                    results[sequence]['channels'][dataset] = channel
                    if len(results[sequence]['channels']) == 4:
                        publish(sequence)
        return results


def compact_wire_parts(frame, cached=()):
    """Return scatter/gather wire views without concatenating full frame bodies."""
    par_text = str(frame['par_text']).encode('utf-8')
    order = np.ascontiguousarray(frame['block_order'], dtype='<i8').reshape(-1)
    dimensions = tuple(map(int, frame['meshblock_size']))
    blocks = int(frame['num_meshblocks'])
    cells = blocks * int(np.prod(dimensions))
    if order.size != 3 * blocks or min(dimensions) < 1:
        raise ValueError('Invalid compact mesh layout')
    channel_parts, descriptors, offset = [], [], 0
    for dataset, count in zip(NativeSequenceDecoder.datasets, (cells, cells, 3*cells, 3*cells)):
        if frame['exact_keyframe']:
            descriptors.append((0,) * 12)
            continue
        codes, scales, indices, values, tiles = frame['channels'][dataset]
        arrays = tuple(np.ascontiguousarray(array) for array in (codes, scales, indices, values))
        codes, scales, indices, values = arrays
        if (codes.dtype not in (np.dtype('int8'), np.dtype('int16'))
                or scales.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or indices.dtype != np.dtype('int64') or values.dtype != np.dtype('float32')
                or codes.size != count or indices.size != values.size):
            raise ValueError('Unsupported compact channel dtype or size')
        offsets = []
        for array in arrays:
            padding = (-offset) % 8
            if padding:
                channel_parts.append(memoryview(bytes(padding)))
                offset += padding
            offsets.append(offset)
            view = memoryview(array).cast('B')
            channel_parts.append(view)
            offset += view.nbytes
        tiles = tuple(tiles) if tiles is not None else (dimensions[2], dimensions[1], dimensions[0])
        descriptors.append((*offsets, count, scales.size, indices.size, codes.itemsize, scales.itemsize, *tiles))
    missing = [sequence for sequence in frame['anchors'] if sequence not in cached]
    metadata = struct.pack('<7Qd', 1, int(frame['exact_keyframe']), frame['start'], frame['end'],
                           len(missing), offset, 0, frame['tau'])
    metadata += b''.join(struct.pack('<12Q', *descriptor) for descriptor in descriptors)
    payloads = [memoryview(par_text), memoryview(order).cast('B'), memoryview(metadata)]
    for sequence in missing:
        _, raw, transformed = frame['anchors'][sequence]
        payloads.append(memoryview(struct.pack('<Q', sequence)))
        for array in [*(raw[name] for name in NativeSequenceDecoder.datasets),
                      transformed['prims.rho'], transformed['prims.u']]:
            payloads.append(memoryview(np.ascontiguousarray(array, dtype='<f4')).cast('B'))
    payloads.extend(channel_parts)
    size = sum(view.nbytes for view in payloads)
    header = (f"STAGEQ1 {frame['sequence']} {frame['time']:.17g} {blocks} "
              f'{dimensions[0]} {dimensions[1]} {dimensions[2]} {len(par_text)} {order.size} '
              f'{cells} {cells} {3*cells} {3*cells} {size}\n').encode('ascii')
    return header, tuple(payloads)
