"""Preview rendering for the tracking panel.

Purely cosmetic — if cv2 isn't importable the frame still gets encoded, it
just arrives unmarked. Nothing in the control path depends on this.
"""

from __future__ import annotations

from typing import Sequence

from .types import Detection

_LOCKED = (60, 220, 90)     # BGR — the track we're actually following
_OTHER = (150, 150, 150)


def annotate(frame, detections: Sequence[Detection], locked_id: int | None = None,
             phase: str = "", quality: int = 70) -> bytes:
    """Draw the boxes and return JPEG bytes ('' if we can't encode)."""
    try:
        import cv2
    except ImportError:
        from .detect_remote import encode_jpeg
        return encode_jpeg(frame, quality)

    canvas = frame.copy()
    h, w = canvas.shape[:2]
    for det in detections:
        locked = locked_id is not None and det.track_id == locked_id
        colour = _LOCKED if locked else _OTHER
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3 if locked else 1)
        label = det.class_name or "?"
        if det.track_id is not None and det.track_id >= 0:
            label += f" #{det.track_id}"
        label += f" {det.score:.2f}"
        cv2.putText(canvas, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        if locked:
            cx, cy = (int(v) for v in det.center)
            cv2.drawMarker(canvas, (cx, cy), _LOCKED, cv2.MARKER_CROSS, 18, 2)

    # Frame centre: where the head is pointing, i.e. what the controller is
    # driving the target toward — makes an off-centre lock obvious at a glance.
    cv2.line(canvas, (w // 2, 0), (w // 2, h), (70, 70, 70), 1)
    if phase:
        cv2.putText(canvas, phase, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, _LOCKED, 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""
