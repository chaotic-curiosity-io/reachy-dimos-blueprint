"""Detection → stable identity, via Roboflow ``trackers``.

Why a tracker at all: detections are per-frame and anonymous. Following the
"highest-scoring person in frame" makes the robot swap targets the moment a
second person walks past. A tracker gives each object an id that survives
occlusion and motion blur, so "follow *this* one" stays meaningful — the
same reason the microduck fetch demo needs it to pick one ball out of
identical distractors.

We use SORT with a buffered IoU (BIoU) metric: cheap (tens of microseconds
per update), no appearance model to run on the robot's CPU, and the buffer
tolerates the frame-to-frame jumps you get at a modest detection rate.

``trackers`` and ``supervision`` are optional. When they are missing — a
robot venv that hasn't been updated, or the offline test run — a small
greedy-IoU fallback with the same interface takes over, so the follow loop
degrades in quality rather than failing to start. ``backend`` says which
one is live, and the UI shows it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .types import Detection

_log = logging.getLogger(__name__)

# Roboflow tracker classes we know how to construct, by short name.
_ALGORITHMS = {
    "sort": "SORTTracker",
    "bytetrack": "ByteTrackTracker",
    "ocsort": "OCSORTTracker",
    "botsort": "BoTSORTTracker",
}


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    hits: int = 1
    misses: int = 0


class _GreedyIouTracker:
    """Fallback: greedy IoU association, same contract as the real thing.

    No motion model, so it will drop a fast target that the Kalman-based
    SORT would have coasted through — acceptable for keeping the app alive,
    not a reason to skip installing ``trackers``.
    """

    def __init__(self, minimum_iou_threshold: float = 0.25,
                 lost_track_buffer: int = 30,
                 minimum_consecutive_frames: int = 1):
        self._min_iou = minimum_iou_threshold
        self._max_misses = max(1, lost_track_buffer)
        self._min_hits = max(1, minimum_consecutive_frames)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        # Best-first over every (track, detection) pair, so a confident
        # overlap wins the assignment regardless of input order.
        pairs = sorted(
            ((_iou(t.bbox, d.bbox), ti, di)
             for ti, t in enumerate(self._tracks)
             for di, d in enumerate(detections)
             # Never let a "person" track adopt a "chair" detection.
             if t.class_name == d.class_name),
            reverse=True,
        )
        used_tracks: set[int] = set()
        assigned: dict[int, int] = {}
        for score, ti, di in pairs:
            if score < self._min_iou or ti in used_tracks or di in assigned:
                continue
            used_tracks.add(ti)
            assigned[di] = ti

        out: list[Detection] = []
        for di, det in enumerate(detections):
            ti = assigned.get(di)
            if ti is None:
                track = _Track(self._next_id, det.bbox, det.class_name)
                self._next_id += 1
                # Mark it seen on the frame it was born, or the miss sweep
                # below charges every brand-new track with a miss.
                used_tracks.add(len(self._tracks))
                self._tracks.append(track)
            else:
                track = self._tracks[ti]
                track.bbox = det.bbox
                track.hits += 1
                track.misses = 0
            # -1 mirrors the trackers library's "not confirmed yet" marker.
            out.append(det.with_track(
                track.track_id if track.hits >= self._min_hits else -1))

        for ti, track in enumerate(self._tracks):
            if ti not in used_tracks:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self._max_misses]
        return out


class TargetTracker:
    """Wraps whichever backend is available behind one ``update``."""

    def __init__(self, algorithm: str = "sort", *, frame_rate: float = 10.0,
                 buffer_ratio: float = 2.0, lost_track_buffer: int = 30,
                 minimum_iou_threshold: float = 0.25,
                 minimum_consecutive_frames: int = 1,
                 prefer_roboflow: bool = True):
        self.algorithm = algorithm if algorithm in _ALGORITHMS else "sort"
        self.backend = "fallback-iou"
        self._impl = None
        self._sv = None
        if prefer_roboflow:
            self._impl = self._build_roboflow(
                frame_rate=frame_rate, buffer_ratio=buffer_ratio,
                lost_track_buffer=lost_track_buffer,
                minimum_iou_threshold=minimum_iou_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames)
        if self._impl is None:
            self._impl = _GreedyIouTracker(
                minimum_iou_threshold=minimum_iou_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_consecutive_frames=minimum_consecutive_frames)

    # --- construction ----------------------------------------------------

    def _build_roboflow(self, **kwargs):
        try:
            import supervision as sv
            import trackers as rf_trackers
        except ImportError as exc:
            _log.info("roboflow trackers unavailable (%s) — using greedy-IoU "
                      "fallback; pip install trackers supervision", exc)
            return None

        cls = getattr(rf_trackers, _ALGORITHMS[self.algorithm], None)
        if cls is None:
            _log.warning("trackers has no %s — using greedy-IoU fallback",
                         _ALGORITHMS[self.algorithm])
            return None

        # Different trackers take different subsets of these; build the
        # richest call that this installed version actually accepts rather
        # than pinning ourselves to one release's signature.
        candidates: list[dict] = []
        try:
            from trackers.utils.iou import BIoU
            candidates.append({**kwargs, "iou": BIoU(
                buffer_ratio=kwargs.get("buffer_ratio", 2.0))})
        except ImportError:
            pass
        candidates.append(dict(kwargs))
        candidates.append({"frame_rate": kwargs.get("frame_rate", 10.0)})
        candidates.append({})

        for attempt in candidates:
            attempt.pop("buffer_ratio", None)
            try:
                impl = cls(**attempt)
            except TypeError:
                continue
            self._sv = sv
            self.backend = f"roboflow:{_ALGORITHMS[self.algorithm]}"
            _log.info("tracking backend: %s", self.backend)
            return impl
        _log.warning("could not construct %s — using greedy-IoU fallback",
                     _ALGORITHMS[self.algorithm])
        return None

    # --- use --------------------------------------------------------------

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Attach track ids. Tracks the library hasn't confirmed get -1."""
        if self._sv is None:
            return self._impl.update(detections)
        try:
            return self._update_roboflow(detections)
        except Exception:  # noqa: BLE001 — a tracker fault must not end a follow
            _log.exception("tracker update failed — passing detections through")
            return [d.with_track(-1) for d in detections]

    def _update_roboflow(self, detections: list[Detection]) -> list[Detection]:
        import numpy as np

        sv = self._sv
        if not detections:
            # SORT still needs the tick: unmatched tracks age out on empty
            # frames, and skipping them would keep stale ids alive forever.
            self._impl.update(sv.Detections.empty())
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=float)
        confidence = np.array([d.score for d in detections], dtype=float)
        class_id = np.array([max(0, d.class_id) for d in detections], dtype=int)
        # Index carried through so we can map results back even if the
        # tracker reorders or drops rows.
        data = {"_idx": np.arange(len(detections))}
        tracked = self._impl.update(sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id, data=data))

        ids = getattr(tracked, "tracker_id", None)
        idx = (tracked.data or {}).get("_idx") if hasattr(tracked, "data") else None
        out = [d.with_track(-1) for d in detections]
        if ids is None:
            return out
        for row, track_id in enumerate(ids):
            source = int(idx[row]) if idx is not None and row < len(idx) else row
            if 0 <= source < len(out):
                out[source] = detections[source].with_track(int(track_id))
        return out
