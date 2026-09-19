"""Persistent bounded DDC frame service for KPolaris slow-light runs."""

from __future__ import annotations

import json
import os
import socket
import socketserver
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from dense_dump_codec.sequence import DenseSequenceIndex


DecodeFrames = Callable[[dict[int, Path]], dict[int, dict[str, Any]]]
DecodeStagedFrames = Callable[[tuple[int, ...]], dict[int, dict[str, Any]]]
NATIVE_DATASETS = ("prims.rho", "prims.u", "prims.uvec", "prims.B")


def _native_wire_parts(frame: dict[str, Any]) -> tuple[bytes, tuple[memoryview, ...]]:
    par_text = str(frame["par_text"]).encode("utf-8")
    block_order = np.ascontiguousarray(frame["block_order"], dtype="<i8").reshape(-1)
    arrays = tuple(
        np.ascontiguousarray(frame["datasets"][name], dtype="<f4").reshape(-1)
        for name in NATIVE_DATASETS
    )
    num_meshblocks = int(frame["num_meshblocks"])
    nx1_mb, nx2_mb, nx3_mb = (int(value) for value in frame["meshblock_size"])
    cells_per_meshblock = nx1_mb * nx2_mb * nx3_mb
    expected_scalar = num_meshblocks * cells_per_meshblock
    expected_vector = 3 * expected_scalar
    expected_sizes = (expected_scalar, expected_scalar, expected_vector, expected_vector)
    if block_order.size != 3 * num_meshblocks:
        raise ValueError("native DDC block-order size mismatch")
    if tuple(array.size for array in arrays) != expected_sizes:
        raise ValueError("native DDC primitive array size mismatch")
    payloads = (
        memoryview(par_text),
        memoryview(block_order).cast("B"),
        *(memoryview(array).cast("B") for array in arrays),
    )
    payload_bytes = sum(payload.nbytes for payload in payloads)
    header = (
        "STAGE1 "
        f"{int(frame['sequence'])} {float(frame['time']):.17g} "
        f"{num_meshblocks} {nx1_mb} {nx2_mb} {nx3_mb} "
        f"{len(par_text)} {block_order.size} "
        + " ".join(str(array.size) for array in arrays)
        + f" {payload_bytes}\n"
    ).encode("ascii")
    return header, payloads


class DDCFrameService:
    """Batch-decode forward frames into an atomically published LRU cache."""

    def __init__(
        self,
        index: DenseSequenceIndex,
        cache_dir: Path | str,
        decode_frames: DecodeFrames,
        *,
        decode_staged_frames: DecodeStagedFrames | None = None,
        maximum_cache_files: int = 4,
        prefetch_files: int | None = None,
    ) -> None:
        if maximum_cache_files < 2:
            raise ValueError("maximum_cache_files must be at least 2")
        self.index = index
        self.cache_dir = Path(cache_dir)
        self.decode_frames = decode_frames
        self.decode_staged_frames = decode_staged_frames
        self.maximum_cache_files = int(maximum_cache_files)
        self.prefetch_files = int(prefetch_files or maximum_cache_files)
        if self.prefetch_files < 1:
            raise ValueError("prefetch_files must be positive")
        if self.prefetch_files > self.maximum_cache_files:
            raise ValueError("prefetch_files must not exceed maximum_cache_files")
        self.native_aligned_prefetch = prefetch_files is not None and self.prefetch_files > 1
        self._native_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._positions = {
            record.sequence: position for position, record in enumerate(index.frames)
        }
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = 0
        self.cache_hits = 0
        self.batch_decode_calls = 0
        self.decoded_frames = 0
        self.evicted_frames = 0
        self.decode_seconds = 0.0
        self.native_requests = 0
        self.native_cache_hits = 0
        self.native_batch_decode_calls = 0
        self.native_decoded_frames = 0
        self.native_evicted_frames = 0
        self.native_decode_seconds = 0.0
        self.native_bytes_sent = 0

    def frame_path(self, sequence: int) -> Path:
        return self.cache_dir / f"ddc_frame_{sequence:05d}.phdf"

    def _prefetch_sequences(self, sequence: int) -> tuple[int, ...]:
        try:
            position = self._positions[int(sequence)]
        except KeyError as error:
            raise KeyError(f"Sequence {sequence} is not present") from error
        return tuple(
            record.sequence
            for record in self.index.frames[
                position : position + self.prefetch_files
            ]
        )

    def _native_prefetch_sequences(self, sequence: int) -> tuple[int, ...]:
        if not self.native_aligned_prefetch:
            return self._prefetch_sequences(sequence)
        try:
            position = self._positions[int(sequence)]
        except KeyError as error:
            raise KeyError(f"Sequence {sequence} is not present") from error
        batch_start = (position // self.prefetch_files) * self.prefetch_files
        return tuple(
            record.sequence
            for record in self.index.frames[
                batch_start : batch_start + self.prefetch_files
            ]
        )

    def _decode_missing(self, sequences: tuple[int, ...]) -> list[int]:
        missing = [sequence for sequence in sequences if not self.frame_path(sequence).is_file()]
        if not missing:
            return []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}.{threading.get_ident()}"
        temporary_outputs = {
            sequence: self.cache_dir
            / f".{self.frame_path(sequence).name}.{token}.partial"
            for sequence in missing
        }
        started = time.monotonic()
        try:
            self.decode_frames(temporary_outputs)
            absent = [
                sequence
                for sequence, path in temporary_outputs.items()
                if not path.is_file()
            ]
            if absent:
                raise RuntimeError(
                    "DDC batch decoder did not create sequences "
                    + ", ".join(map(str, absent))
                )
            for sequence, temporary in temporary_outputs.items():
                os.replace(temporary, self.frame_path(sequence))
        finally:
            for temporary in temporary_outputs.values():
                temporary.unlink(missing_ok=True)
        self.batch_decode_calls += 1
        self.decoded_frames += len(missing)
        self.decode_seconds += time.monotonic() - started
        return missing

    def _prune(self, protected: set[Path]) -> list[Path]:
        paths = list(self.cache_dir.glob("ddc_frame_*.phdf"))
        excess = max(0, len(paths) - self.maximum_cache_files)
        candidates = sorted(
            (path for path in paths if path not in protected),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        evicted = candidates[:excess]
        for path in evicted:
            path.unlink(missing_ok=True)
            Path(f"{path}.xdmf").unlink(missing_ok=True)
        self.evicted_frames += len(evicted)
        return evicted

    def materialize_sequence(self, sequence: int) -> dict[str, Any]:
        with self._lock:
            self.requests += 1
            sequence = int(sequence)
            self.index.by_sequence(sequence)
            requested = self.frame_path(sequence)
            hit = requested.is_file()
            if hit:
                self.cache_hits += 1
            prefetch_sequences = self._prefetch_sequences(sequence)
            decoded = self._decode_missing(prefetch_sequences)
            if not requested.is_file():
                raise RuntimeError(f"DDC frame {sequence} was not materialized")
            protected = {self.frame_path(item) for item in prefetch_sequences}
            now_ns = time.time_ns()
            for path in protected:
                if path.is_file():
                    os.utime(path, ns=(now_ns, now_ns))
            evicted = self._prune(protected)
            return {
                "sequence": sequence,
                "materialized_path": str(requested),
                "cache_hit": hit,
                "prefetched_sequences": list(prefetch_sequences),
                "batch_decoded_sequences": decoded,
                "evicted_paths": [str(path) for path in evicted],
            }

    def _decode_native_missing(self, sequences: tuple[int, ...]) -> list[int]:
        if self.decode_staged_frames is None:
            raise RuntimeError("native DDC staging is not configured")
        missing = [sequence for sequence in sequences if sequence not in self._native_cache]
        if not missing:
            return []
        started = time.monotonic()
        decoded = self.decode_staged_frames(tuple(missing))
        absent = [sequence for sequence in missing if sequence not in decoded]
        if absent:
            raise RuntimeError(
                "DDC native decoder did not return sequences "
                + ", ".join(map(str, absent))
            )
        for sequence in missing:
            frame = decoded[sequence]
            if int(frame.get("sequence", sequence)) != sequence:
                raise ValueError("DDC native decoder returned a mismatched sequence")
            _native_wire_parts(frame)
            self._native_cache[sequence] = frame
        self.native_batch_decode_calls += 1
        self.native_decoded_frames += len(missing)
        self.native_decode_seconds += time.monotonic() - started
        return missing

    def stage_sequence(self, sequence: int) -> dict[str, Any]:
        with self._lock:
            self.native_requests += 1
            sequence = int(sequence)
            self.index.by_sequence(sequence)
            hit = sequence in self._native_cache
            if hit:
                self.native_cache_hits += 1
            prefetch_sequences = self._native_prefetch_sequences(sequence)
            self._decode_native_missing(prefetch_sequences)
            for item in prefetch_sequences:
                if item in self._native_cache:
                    self._native_cache.move_to_end(item)
            frame = self._native_cache[sequence]
            while len(self._native_cache) > self.maximum_cache_files:
                self._native_cache.popitem(last=False)
                self.native_evicted_frames += 1
            return frame

    def warm_sequences(self, start_sequence: int, count: int) -> dict[str, Any]:
        """Decode and retain one exact contiguous native sequence range."""
        if count < 1:
            raise ValueError("warm sequence count must be positive")
        with self._lock:
            try:
                position = self._positions[int(start_sequence)]
            except KeyError as error:
                raise KeyError(f"Sequence {start_sequence} is not present") from error
            sequences = tuple(
                record.sequence
                for record in self.index.frames[position : position + int(count)]
            )
            if len(sequences) != int(count):
                raise ValueError("warm sequence range exceeds the DDC index")
            decoded = self._decode_native_missing(sequences)
            for sequence in sequences:
                self._native_cache.move_to_end(sequence)
            while len(self._native_cache) > self.maximum_cache_files:
                self._native_cache.popitem(last=False)
                self.native_evicted_frames += 1
            return {
                "start_sequence": sequences[0],
                "end_sequence": sequences[-1],
                "sequence_count": len(sequences),
                "decoded_count": len(decoded),
            }

    def record_native_bytes_sent(self, count: int) -> None:
        with self._lock:
            self.native_bytes_sent += int(count)

    def statistics(self) -> dict[str, Any]:
        return {
            "format": "kpolaris_ddc_service_stats_v1",
            "uptime_seconds": time.time() - self.started_at,
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "batch_decode_calls": self.batch_decode_calls,
            "decoded_frames": self.decoded_frames,
            "evicted_frames": self.evicted_frames,
            "decode_seconds": self.decode_seconds,
            "mean_decode_seconds_per_frame": (
                self.decode_seconds / self.decoded_frames if self.decoded_frames else 0.0
            ),
            "maximum_cache_files": self.maximum_cache_files,
            "prefetch_files": self.prefetch_files,
            "native_prefetch_mode": (
                "aligned_batch" if self.native_aligned_prefetch else "sliding"
            ),
            "cache_dir": str(self.cache_dir),
            "native_requests": self.native_requests,
            "native_cache_hits": self.native_cache_hits,
            "native_batch_decode_calls": self.native_batch_decode_calls,
            "native_decoded_frames": self.native_decoded_frames,
            "native_evicted_frames": self.native_evicted_frames,
            "native_decode_seconds": self.native_decode_seconds,
            "native_mean_decode_seconds_per_frame": (
                self.native_decode_seconds / self.native_decoded_frames
                if self.native_decoded_frames
                else 0.0
            ),
            "native_bytes_sent": self.native_bytes_sent,
            "native_cache_frames": len(self._native_cache),
        }


class _DDCUnixRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request = self.rfile.readline(4096).decode("utf-8", errors="replace").strip()
        if request == "SHUTDOWN":
            self.wfile.write(b"OK shutdown\n")
            self.wfile.flush()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if request.startswith("WARM "):
            try:
                _, start_value, count_value = request.split()
                result = self.server.frame_service.warm_sequences(
                    int(start_value), int(count_value)
                )
                self.wfile.write(
                    (
                        "OK warm "
                        f"{result['start_sequence']} {result['end_sequence']} "
                        f"{result['sequence_count']} {result['decoded_count']}\n"
                    ).encode("ascii")
                )
            except Exception as error:
                message = str(error).replace("\n", " ")
                self.wfile.write(f"ERROR {message}\n".encode("utf-8"))
            return
        command, separator, value = request.partition(" ")
        if not separator or command not in {"MATERIALIZE", "STAGE", "STAGEQ"}:
            self.wfile.write(b"ERROR expected MATERIALIZE or STAGE <sequence>\n")
            return
        try:
            if command == "MATERIALIZE":
                result = self.server.frame_service.materialize_sequence(int(value))
                self.wfile.write(f"OK {result['materialized_path']}\n".encode("utf-8"))
                return
            if command == "STAGEQ":
                from .compact import compact_wire_parts
                fields = value.split()
                if not 1 <= len(fields) <= 3:
                    raise ValueError("STAGEQ accepts one sequence and up to two cached anchors")
                frame = self.server.frame_service.stage_sequence(int(fields[0]))
                header, payloads = compact_wire_parts(frame, tuple(map(int, fields[1:])))
            else:
                frame = self.server.frame_service.stage_sequence(int(value))
                header, payloads = _native_wire_parts(frame)
            self.wfile.write(header)
            self.wfile.flush()
            sent = 0
            for payload in payloads:
                self.connection.sendall(payload)
                sent += payload.nbytes
            self.server.frame_service.record_native_bytes_sent(sent)
        except Exception as error:
            if command == "STAGE" and getattr(self.wfile, "closed", False):
                return
            message = str(error).replace("\n", " ")
            try:
                self.wfile.write(f"ERROR {message}\n".encode("utf-8"))
            except OSError:
                pass


class DDCUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: Path | str, frame_service: DDCFrameService) -> None:
        self.socket_path = Path(socket_path)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.frame_service = frame_service
        super().__init__(str(self.socket_path), _DDCUnixRequestHandler)
        os.chmod(self.socket_path, 0o600)

    def server_close(self) -> None:
        super().server_close()
        self.socket_path.unlink(missing_ok=True)


def _send_request(socket_path: Path | str, request: str, *, timeout: float) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        connection.sendall(f"{request}\n".encode("utf-8"))
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = connection.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > 65536:
                raise RuntimeError("DDC service response exceeded 65536 bytes")
    return response.decode("utf-8", errors="replace").strip()


def request_ddc_frame(
    socket_path: Path | str,
    sequence: int,
    *,
    timeout: float = 3600.0,
) -> Path:
    response = _send_request(
        socket_path,
        f"MATERIALIZE {int(sequence)}",
        timeout=timeout,
    )
    if not response.startswith("OK "):
        raise RuntimeError(f"DDC service request failed: {response}")
    path = Path(response[3:])
    if not path.is_file():
        raise RuntimeError(f"DDC service returned missing frame {path}")
    return path


def _receive_line(connection: socket.socket, *, maximum_bytes: int = 65536) -> bytes:
    response = bytearray()
    while not response.endswith(b"\n"):
        chunk = connection.recv(1)
        if not chunk:
            raise RuntimeError("DDC service closed before the native header")
        response.extend(chunk)
        if len(response) > maximum_bytes:
            raise RuntimeError("DDC native header exceeded its size limit")
    return bytes(response[:-1])


def _receive_exact(connection: socket.socket, count: int) -> bytes:
    payload = bytearray(count)
    view = memoryview(payload)
    received = 0
    while received < count:
        size = connection.recv_into(view[received:])
        if size == 0:
            raise RuntimeError("DDC service closed during native frame transfer")
        received += size
    return bytes(payload)


def request_ddc_stage(
    socket_path: Path | str,
    sequence: int,
    *,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        connection.sendall(f"STAGE {int(sequence)}\n".encode("ascii"))
        header = _receive_line(connection).decode("ascii", errors="replace")
        if header.startswith("ERROR "):
            raise RuntimeError(f"DDC native stage failed: {header}")
        fields = header.split()
        if len(fields) != 14 or fields[0] != "STAGE1":
            raise RuntimeError(f"Invalid DDC native header: {header}")
        (
            _, sequence_value, time_value, num_meshblocks, nx1_mb, nx2_mb, nx3_mb,
            par_bytes, block_count, rho_count, u_count, uvec_count, b_count,
            payload_bytes,
        ) = fields
        counts = tuple(
            int(value)
            for value in (par_bytes, block_count, rho_count, u_count, uvec_count, b_count)
        )
        byte_counts = (
            counts[0], counts[1] * 8, counts[2] * 4, counts[3] * 4,
            counts[4] * 4, counts[5] * 4,
        )
        if sum(byte_counts) != int(payload_bytes):
            raise RuntimeError("DDC native payload byte count mismatch")
        payloads = tuple(_receive_exact(connection, count) for count in byte_counts)
    return {
        "sequence": int(sequence_value),
        "time": float(time_value),
        "num_meshblocks": int(num_meshblocks),
        "meshblock_size": (int(nx1_mb), int(nx2_mb), int(nx3_mb)),
        "par_text": payloads[0].decode("utf-8"),
        "block_order": np.frombuffer(payloads[1], dtype="<i8").copy(),
        "datasets": {
            name: np.frombuffer(payload, dtype="<f4").copy()
            for name, payload in zip(NATIVE_DATASETS, payloads[2:], strict=True)
        },
    }


def shutdown_ddc_server(socket_path: Path | str, *, timeout: float = 10.0) -> None:
    response = _send_request(socket_path, "SHUTDOWN", timeout=timeout)
    if response != "OK shutdown":
        raise RuntimeError(f"DDC service shutdown failed: {response}")


def warm_ddc_sequences(
    socket_path: Path | str,
    start_sequence: int,
    count: int,
    *,
    timeout: float = 14400.0,
) -> dict[str, int]:
    response = _send_request(
        socket_path,
        f"WARM {int(start_sequence)} {int(count)}",
        timeout=timeout,
    )
    fields = response.split()
    if len(fields) != 6 or fields[:2] != ["OK", "warm"]:
        raise RuntimeError(f"DDC warm request failed: {response}")
    return {
        "start_sequence": int(fields[2]),
        "end_sequence": int(fields[3]),
        "sequence_count": int(fields[4]),
        "decoded_count": int(fields[5]),
    }


def write_service_statistics(path: Path | str, service: DDCFrameService) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(service.statistics(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
