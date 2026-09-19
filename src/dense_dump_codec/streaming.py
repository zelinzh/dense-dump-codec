"""Bounded per-frame publication for the native DDC Unix service."""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict


class _Stopped(Exception):
    pass


class StreamingFrameService:
    """One producer; completed frames can be consumed before the batch finishes."""

    def __init__(self, index, cache_dir, decoder, *, maximum_cache_files=32,
                 prefetch_files=25, validate_frame=None):
        if not 1 <= prefetch_files <= maximum_cache_files or maximum_cache_files < 2:
            raise ValueError("Require 1 <= prefetch <= cache and cache >= 2")
        if prefetch_files > getattr(decoder, "maximum_batch_frames", prefetch_files):
            raise ValueError("Prefetch exceeds the decoder batch limit")
        self.index, self.cache_dir, self.decoder = index, str(cache_dir), decoder
        self.maximum_cache_files, self.prefetch_files = maximum_cache_files, prefetch_files
        self._positions = {record.sequence: position
                           for position, record in enumerate(index.frames)}
        self._cache, self._wanted = OrderedDict(), Counter()
        self._condition = threading.Condition()
        self._stop, self._error = False, None
        self._validate = validate_frame or (lambda frame: None)
        self.started_at = time.time()
        self._stats = dict(native_requests=0, native_cache_hits=0, native_batch_decode_calls=0,
                           native_decoded_frames=0, native_evicted_frames=0,
                           native_decode_seconds=0.0, native_bytes_sent=0,
                           native_wait_seconds=0.0, native_first_published_seconds=None,
                           native_peak_cache_frames=0)
        self._thread = threading.Thread(target=self._produce, name="ddc-stream", daemon=True)
        self._thread.start()

    def _produce(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or any(
                        sequence not in self._cache for sequence in self._wanted))
                    if self._stop:
                        return
                    first = next(sequence for sequence in self._wanted
                                 if sequence not in self._cache)
                    position = self._positions[first]
                    sequences = tuple(record.sequence for record in self.index.frames[
                        position:position + self.prefetch_files]
                        if record.sequence not in self._cache)
                started = time.monotonic()
                published = set()

                def publish(sequence, frame):
                    if sequence not in sequences or sequence in published:
                        raise ValueError("Decoder published an unexpected or duplicate sequence")
                    if int(frame.get("sequence", sequence)) != sequence:
                        raise ValueError("Decoder returned a mismatched frame sequence")
                    self._validate(frame)
                    published.add(sequence)
                    with self._condition:
                        if self._stop:
                            raise _Stopped()
                        while len(self._cache) >= self.maximum_cache_files:
                            victim = next((cached for cached in self._cache
                                           if cached not in self._wanted), None)
                            if victim is None:
                                self._condition.wait()
                                if self._stop:
                                    raise _Stopped()
                                continue
                            del self._cache[victim]
                            self._stats["native_evicted_frames"] += 1
                        self._cache[sequence] = frame
                        self._stats["native_decoded_frames"] += 1
                        self._stats["native_peak_cache_frames"] = max(
                            len(self._cache), self._stats["native_peak_cache_frames"])
                        if self._stats["native_first_published_seconds"] is None:
                            self._stats["native_first_published_seconds"] = time.monotonic() - started
                        self._condition.notify_all()

                try:
                    self.decoder(sequences, on_frame=publish)
                    if published != set(sequences):
                        raise RuntimeError("DDC decoder did not publish all requested frames")
                finally:
                    with self._condition:
                        self._stats["native_batch_decode_calls"] += 1
                        self._stats["native_decode_seconds"] += time.monotonic() - started
        except _Stopped:
            pass
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def stage_sequence(self, sequence):
        sequence = int(sequence)
        if sequence not in self._positions:
            raise KeyError(f"Sequence {sequence} is not present")
        with self._condition:
            self._stats["native_requests"] += 1
            if sequence in self._cache:
                self._stats["native_cache_hits"] += 1
            self._wanted[sequence] += 1
            self._condition.notify_all()
            started = time.monotonic()
            try:
                self._condition.wait_for(
                    lambda: sequence in self._cache or self._error is not None or self._stop)
                if self._error is not None:
                    raise RuntimeError("DDC streaming decode failed") from self._error
                if self._stop:
                    raise RuntimeError("DDC streaming service has stopped")
                self._cache.move_to_end(sequence)
                return self._cache[sequence]
            finally:
                self._stats["native_wait_seconds"] += time.monotonic() - started
                self._wanted[sequence] -= 1
                if not self._wanted[sequence]:
                    del self._wanted[sequence]
                self._condition.notify_all()

    def warm_sequences(self, start_sequence, count):
        if not 1 <= count <= self.maximum_cache_files:
            raise ValueError("Warm range must fit the native cache")
        position = self._positions[int(start_sequence)]
        sequences = tuple(record.sequence for record in self.index.frames[position:position + count])
        if len(sequences) != count:
            raise ValueError("Warm range exceeds index")
        with self._condition:
            before = self._stats["native_decoded_frames"]
            self._wanted.update(sequences)
            self._condition.notify_all()
        try:
            for sequence in sequences:
                self.stage_sequence(sequence)
            with self._condition:
                return dict(start_sequence=sequences[0], end_sequence=sequences[-1],
                            sequence_count=count,
                            decoded_count=self._stats["native_decoded_frames"] - before)
        finally:
            with self._condition:
                self._wanted.subtract(sequences)
                self._wanted += Counter()
                self._condition.notify_all()

    def materialize_sequence(self, sequence):
        raise ValueError("This service supports native STAGE input only; set kharma_ddc_native=1")

    def record_native_bytes_sent(self, count):
        with self._condition:
            self._stats["native_bytes_sent"] += count

    def statistics(self):
        predictor = getattr(getattr(self.decoder, "reconstruction", None), "predictor", None)
        with self._condition:
            return dict(
                native_prediction_kernel=None if predictor is None else dict(
                    path=str(predictor.path), sha256=predictor.sha256, abi=1),
                format="kpolaris_ddc_service_stats_v1", uptime_seconds=time.time() - self.started_at,
                cache_dir=self.cache_dir, maximum_cache_files=self.maximum_cache_files,
                prefetch_files=self.prefetch_files, native_prefetch_mode="streaming_batch",
                native_reconstruction=getattr(self.decoder, "reconstruction_label",
                    getattr(getattr(self.decoder, "reconstruction", None), "mode", "external")),
                native_cache_frames=len(self._cache), **self._stats,
                native_access=getattr(self.decoder, "access_statistics", lambda: {})(),
                native_mean_decode_seconds_per_frame=self._stats["native_decode_seconds"]
                / max(1, self._stats["native_decoded_frames"]),
                native_error=str(self._error) if self._error is not None else None,
            )

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join()
