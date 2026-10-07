"""Detector interface + backend selection.

``trackers`` only does association; something has to produce boxes for it.
Two backends, chosen by the ``track_detector`` setting:

``onnx``    on-robot ONNX (COCO-80) through onnxruntime, which is already in
            the robot's shared apps venv. No LAN dependency, so the follow
            loop keeps its rate even when wifi hiccups. Closed vocabulary.
``remote``  POST the frame to a detector service on the station (LAN)
            and get boxes back. Costs a round trip per frame, buys an
            open vocabulary — "follow the red mug" verbatim.

Both are constructed lazily and may fail (no model file, service down);
that failure is a ``DetectorUnavailable`` carrying a sentence a user can
act on, which the API and the voice agent surface verbatim.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

from .types import Detection
from .vocab import COCO_CLASSES

_log = logging.getLogger(__name__)

BACKENDS = ("onnx", "remote")


class DetectorUnavailable(RuntimeError):
    """The chosen backend cannot run, with a reason worth showing a human."""


@runtime_checkable
class Detector(Protocol):
    """Frame in, boxes out."""

    name: str

    @property
    def vocabulary(self) -> Sequence[str]:
        """Labels this backend can produce; empty tuple = open vocabulary."""

    def detect(self, frame, targets: Sequence[str] = ()) -> list[Detection]:
        """Detect in a BGR uint8 ``(H, W, 3)`` frame.

        ``targets`` is a hint: closed-vocabulary backends filter by it,
        open-vocabulary ones use it as the prompt. An empty ``targets``
        means "everything you can find".
        """

    def close(self) -> None:
        ...


class NullDetector:
    """Stands in when no backend is configured, so callers can still run."""

    name = "none"

    @property
    def vocabulary(self) -> Sequence[str]:
        return ()

    def detect(self, frame, targets: Sequence[str] = ()) -> list[Detection]:
        return []

    def close(self) -> None:
        return None


def build_detector(state: dict) -> Detector:
    """Construct the detector named by ``state['track_detector']``.

    Raises ``DetectorUnavailable`` rather than returning a broken object —
    a follow that cannot see must refuse to start, not drive blind.
    """
    backend = str(state.get("track_detector") or "onnx").strip().lower()
    if backend == "remote":
        from .detect_remote import RemoteDetector
        return RemoteDetector(
            url=str(state.get("track_remote_url") or ""),
            timeout=float(state.get("track_remote_timeout", 2.0)),
            min_score=float(state.get("track_min_score", 0.35)),
        )
    if backend == "onnx":
        from .detect_onnx import OnnxDetector
        return OnnxDetector(
            model_path=str(state.get("track_model_path") or ""),
            imgsz=int(state.get("track_imgsz", 320)),
            min_score=float(state.get("track_min_score", 0.35)),
            nms_iou=float(state.get("track_nms_iou", 0.45)),
            class_names=COCO_CLASSES,
        )
    raise DetectorUnavailable(
        f"unknown detector backend {backend!r} — expected one of "
        + ", ".join(BACKENDS)
    )


def filter_detections(detections: list[Detection], targets: Sequence[str],
                      min_score: float = 0.0) -> list[Detection]:
    """Keep detections whose class is wanted and whose score clears the bar."""
    wanted = {str(t).strip().lower() for t in targets if str(t).strip()}
    out = []
    for det in detections:
        if det.score < min_score:
            continue
        if wanted and det.class_name.strip().lower() not in wanted:
            continue
        out.append(det)
    return out
