"""Open-vocabulary detection offloaded to a LAN service.

The on-robot ONNX backend is fast but stuck with COCO's 80 labels. When you
want to say "follow the red mug" and have it mean the red mug, point
``track_remote_url`` at a service running a bigger, prompted model on your
station (a Mac or a GPU box on the LAN).

Wire format, kept dumb on purpose so anything can implement it:

    POST <url>?targets=red+mug     body: raw JPEG, Content-Type: image/jpeg
    200  {"detections": [{"bbox": [x1,y1,x2,y2], "score": 0.8,
                          "class_name": "red mug"}], "width": 640, ...}

Roboflow-inference-shaped replies (``predictions`` with centre x/y +
width/height) are accepted too, so an existing inference server can be
used unmodified.

stdlib-only, like ``wheels_client`` — the robot side of this adds no
dependency beyond the JPEG encode it was already doing for the camera
endpoint. Tracking still runs on the robot; only the boxes come over the
wire, so a slow reply costs detection rate, never control authority.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.parse
import urllib.request
from typing import Sequence

from .detectors import DetectorUnavailable
from .types import Detection

_log = logging.getLogger(__name__)


def encode_jpeg(frame, quality: int = 70) -> bytes:
    """BGR uint8 array → JPEG bytes, via cv2 or Pillow, whichever is here."""
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else b""
    except ImportError:
        pass
    try:
        import io

        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise DetectorUnavailable(
            "remote detection needs opencv or Pillow to encode frames"
        ) from exc
    rgb = np.ascontiguousarray(frame[:, :, 2::-1])
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def parse_detections(payload: dict, min_score: float = 0.0) -> list[Detection]:
    """Read either accepted reply shape into our Detection list."""
    rows = payload.get("detections")
    if rows is None:
        rows = payload.get("predictions") or []

    out: list[Detection] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("class_name") or row.get("class") or
                   row.get("label") or "").strip()
        try:
            score = float(row.get("score", row.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_score:
            continue

        box = row.get("bbox") or row.get("box") or row.get("xyxy")
        try:
            if box is not None and len(box) == 4:
                x1, y1, x2, y2 = (float(v) for v in box)
            else:
                # Roboflow inference: centre + extent.
                cx, cy = float(row["x"]), float(row["y"])
                bw, bh = float(row["width"]), float(row["height"])
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        except (KeyError, TypeError, ValueError):
            continue
        # NaN/Infinity: json.loads accepts them bare, and every comparison
        # with NaN is False, so `x2 <= x1` would wave one through — and a
        # NaN box poisons the controller and the preview renderer alike.
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        if not math.isfinite(score):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        try:
            class_id = int(row.get("class_id", -1))
        except (TypeError, ValueError):
            class_id = -1
        out.append(Detection(bbox=(x1, y1, x2, y2), score=score,
                             class_name=name, class_id=class_id))
    return out


class RemoteDetector:
    """Posts frames to a detection service; open vocabulary."""

    name = "remote"

    def __init__(self, url: str, *, timeout: float = 2.0,
                 min_score: float = 0.35, jpeg_quality: int = 70):
        url = (url or "").strip()
        if not url:
            raise DetectorUnavailable(
                "remote detection is selected but track_remote_url is empty — "
                "set it to e.g. http://<mac-ip>:8055/detect")
        self.url = url
        self.timeout = float(timeout)
        self.min_score = float(min_score)
        self.jpeg_quality = int(jpeg_quality)
        self._consecutive_errors = 0

    @property
    def vocabulary(self) -> Sequence[str]:
        return ()  # open — the phrase is the prompt

    def detect(self, frame, targets: Sequence[str] = ()) -> list[Detection]:
        if frame is None or getattr(frame, "ndim", 0) != 3:
            return []
        jpeg = encode_jpeg(frame, self.jpeg_quality)
        if not jpeg:
            return []

        url = self.url
        prompt = ", ".join(str(t).strip() for t in targets if str(t).strip())
        if prompt:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode({'targets': prompt})}"

        req = urllib.request.Request(
            url, data=jpeg, method="POST",
            headers={"Content-Type": "image/jpeg", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, json.JSONDecodeError) as exc:
            self._consecutive_errors += 1
            # A dropped frame is survivable — the follow loop's own
            # hold/search handles a gap — so log the first few and then go
            # quiet rather than filling the robot's log at loop rate. Past
            # the threshold we keep raising (not just once) so a later
            # follow against a still-dead service gets the same clear
            # answer instead of silently searching for nothing.
            if self._consecutive_errors <= 3:
                _log.warning("remote detector %s failed: %s", self.url, exc)
            if self._consecutive_errors >= 10:
                raise DetectorUnavailable(
                    f"remote detector at {self.url} failed 10 times in a row: {exc}"
                ) from exc
            return []

        self._consecutive_errors = 0
        if not isinstance(payload, dict):
            return []
        return parse_detections(payload, self.min_score)

    def close(self) -> None:
        return None
