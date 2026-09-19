"""Time-indexed random access to dense-dump codec sequence manifests."""

from __future__ import annotations

import bisect
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameRecord:
    sequence: int
    time: float
    path: str
    size_bytes: int | None = None
    exact_keyframe: bool = False

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> FrameRecord:
        return cls(
            sequence=int(row["sequence"]),
            time=float(row["time"]),
            path=str(row.get("path", "")),
            size_bytes=(None if row.get("size_bytes") is None else int(row["size_bytes"])),
            exact_keyframe=bool(row.get("exact_keyframe", False)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "time": self.time,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "exact_keyframe": self.exact_keyframe,
        }


@dataclass(frozen=True)
class FrameBracket:
    requested_time: float
    lower: FrameRecord
    upper: FrameRecord
    upper_weight: float

    @property
    def exact(self) -> bool:
        return self.lower.sequence == self.upper.sequence

    def to_json(self) -> dict[str, Any]:
        return {
            "requested_time": self.requested_time,
            "lower": self.lower.to_json(),
            "upper": self.upper.to_json(),
            "upper_weight": self.upper_weight,
            "lower_weight": 1.0 - self.upper_weight,
            "exact": self.exact,
        }


class DenseSequenceIndex:
    """Validated physical-time index for a DDC sequence manifest."""

    def __init__(self, frames: Iterable[FrameRecord]) -> None:
        ordered = tuple(sorted(frames, key=lambda row: (row.time, row.sequence)))
        if not ordered:
            raise ValueError("A dense sequence index requires at least one frame")
        sequences = [row.sequence for row in ordered]
        if len(sequences) != len(set(sequences)):
            raise ValueError("Frame sequence numbers must be unique")
        times = [row.time for row in ordered]
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("Frame times must be strictly increasing")
        self.frames = ordered
        self._times = tuple(times)
        self._by_sequence = {row.sequence: row for row in ordered}

    @classmethod
    def from_manifest(cls, path: Path | str) -> DenseSequenceIndex:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = manifest.get("frames")
        if not rows:
            raise ValueError(
                "Sequence manifest has no physical-time frame index; regenerate it with "
                "the current encoder or streaming watcher"
            )
        return cls(FrameRecord.from_json(row) for row in rows)

    def by_sequence(self, sequence: int) -> FrameRecord:
        try:
            return self._by_sequence[int(sequence)]
        except KeyError as error:
            raise KeyError(f"Sequence {sequence} is not present") from error

    def covering_range(
        self,
        start_time: float,
        end_time: float,
        *,
        tolerance: float = 1.0e-9,
    ) -> tuple[FrameRecord, ...]:
        """Return the smallest adjacent-frame span covering a fluid-time interval."""
        start_time = float(start_time)
        end_time = float(end_time)
        if end_time < start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        lower = self.bracket(start_time, tolerance=tolerance).lower
        upper = self.bracket(end_time, tolerance=tolerance).upper
        first = self.frames.index(lower)
        last = self.frames.index(upper)
        selected = self.frames[first : last + 1]
        if len(selected) == 1 and len(self.frames) > 1:
            if last + 1 < len(self.frames):
                selected = self.frames[first : last + 2]
            else:
                selected = self.frames[first - 1 : last + 1]
        return selected

    def bracket(self, requested_time: float, *, tolerance: float = 1.0e-9) -> FrameBracket:
        requested_time = float(requested_time)
        if requested_time < self._times[0] - tolerance:
            raise ValueError(
                f"Requested time {requested_time} precedes sequence start {self._times[0]}"
            )
        if requested_time > self._times[-1] + tolerance:
            raise ValueError(
                f"Requested time {requested_time} exceeds sequence end {self._times[-1]}"
            )
        position = bisect.bisect_left(self._times, requested_time)
        candidates = []
        if position < len(self.frames):
            candidates.append(self.frames[position])
        if position:
            candidates.append(self.frames[position - 1])
        exact = min(candidates, key=lambda row: abs(row.time - requested_time))
        if abs(exact.time - requested_time) <= tolerance:
            return FrameBracket(requested_time, exact, exact, 0.0)
        if position == 0 or position == len(self.frames):
            raise ValueError(f"Cannot bracket requested time {requested_time}")
        lower = self.frames[position - 1]
        upper = self.frames[position]
        weight = (requested_time - lower.time) / (upper.time - lower.time)
        return FrameBracket(requested_time, lower, upper, float(weight))


DecodeFrame = Callable[[int, Path], dict[str, Any]]


class DDCFrameMaterializer:
    """Materialize only the frame or frame pair needed by a GRRT request."""

    def __init__(
        self,
        index: DenseSequenceIndex,
        cache_dir: Path | str,
        decode_frame: DecodeFrame,
        *,
        maximum_cache_files: int = 4,
    ) -> None:
        if maximum_cache_files < 2:
            raise ValueError("maximum_cache_files must be at least 2")
        self.index = index
        self.cache_dir = Path(cache_dir)
        self.decode_frame = decode_frame
        self.maximum_cache_files = int(maximum_cache_files)

    def _path(self, sequence: int) -> Path:
        return self.cache_dir / f"ddc_frame_{sequence:05d}.phdf"

    def _materialize(self, record: FrameRecord) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        output = self._path(record.sequence)
        if not output.exists():
            self.decode_frame(record.sequence, output)
        os.utime(output, None)
        return output

    def _prune(self, protected: set[Path]) -> None:
        candidates = sorted(
            (path for path in self.cache_dir.glob("ddc_frame_*.phdf") if path not in protected),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        total = len(candidates) + len(protected)
        for path in candidates[: max(0, total - self.maximum_cache_files)]:
            path.unlink(missing_ok=True)
            Path(f"{path}.xdmf").unlink(missing_ok=True)

    def materialize_time(
        self, requested_time: float, *, tolerance: float = 1.0e-9
    ) -> dict[str, Any]:
        bracket = self.index.bracket(requested_time, tolerance=tolerance)
        lower_path = self._materialize(bracket.lower)
        upper_path = lower_path if bracket.exact else self._materialize(bracket.upper)
        protected = {lower_path, upper_path}
        self._prune(protected)
        result = bracket.to_json()
        result["lower"]["materialized_path"] = str(lower_path)
        result["upper"]["materialized_path"] = str(upper_path)
        return result

    def materialize_sequence(self, sequence: int) -> dict[str, Any]:
        record = self.index.by_sequence(sequence)
        output = self._materialize(record)
        self._prune({output})
        result = record.to_json()
        result["materialized_path"] = str(output)
        return result
