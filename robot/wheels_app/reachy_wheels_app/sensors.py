"""Read-only L515 HTTP adapter. RSPC v1 is XYZ metres in optical axes.

The deployed Pi protocol has no capture timestamp or session ID. Never
claim RGB/depth synchronization or use its arrival time as measured odometry.
"""
from __future__ import annotations

import json
import math
import os
import struct
import time
from urllib.request import urlopen
from urllib.parse import urlsplit

HEADER = struct.Struct("<4sIII")
MAX_POINTS = 320 * 240
# Origin of the Raspberry Pi depth streamer (perception/depth_server), e.g.
# "http://<pi-ip>:8765". No hardcoded default: set DEPTH_SERVER_URL in the
# environment or POST it to /api/sensors/config from the web UI.
DEFAULT_DEPTH_URL = os.environ.get("DEPTH_SERVER_URL", "")
DEPTH_FRAME = "realsense_depth_optical"
RGB_FRAME = "reachy_head_camera_optical"


def validate_url(url: str) -> str:
    if not url:
        raise ValueError("depth_url is not set — set DEPTH_SERVER_URL or POST "
                         "/api/sensors/config with http://<pi-ip>:8765")
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in {"", "/"}):
        raise ValueError("depth_url must be an HTTP(S) origin without credentials")
    _ = parts.port  # validates malformed ports
    return url.rstrip("/")


def unpack_points(payload: bytes) -> tuple[int, int]:
    if len(payload) < HEADER.size:
        raise ValueError("truncated RSPC header")
    magic, version, sequence, count = HEADER.unpack_from(payload)
    if magic != b"RSPC" or version != 1:
        raise ValueError("unsupported RSPC protocol")
    if not 0 < count <= MAX_POINTS or len(payload) != HEADER.size + count * 12:
        raise ValueError("invalid RSPC point count or payload length")
    for point in struct.iter_unpack("<fff", payload[HEADER.size:]):
        if not all(math.isfinite(v) for v in point) or point[2] <= 0:
            raise ValueError("invalid XYZ point")
    return sequence, count


class DepthSource:
    def __init__(self, url=DEFAULT_DEPTH_URL, timeout=1.0, max_age_ms=500.0):
        self.url = validate_url(url)
        self.timeout = timeout
        self.max_age_ms = max_age_ms

    def _get(self, path: str, limit: int) -> bytes:
        with urlopen(self.url + path, timeout=self.timeout) as response:
            payload = response.read(limit + 1)
        if len(payload) > limit:
            raise ValueError("sensor response exceeds size limit")
        return payload

    def status(self) -> dict:
        started = time.monotonic()
        status = json.loads(self._get("/api/status", 16384))
        if not isinstance(status, dict):
            raise ValueError("invalid depth status response")
        age = status.get("last_frame_age_ms")
        if (status.get("state") != "streaming" or isinstance(age, bool)
                or not isinstance(age, (int, float)) or not math.isfinite(age)
                or age < 0 or age + (time.monotonic() - started) * 1000 > self.max_age_ms):
            raise ValueError("depth source is stale or not streaming")
        return status

    def frame(self) -> tuple[bytes, dict]:
        # Sample status first. The payload must be that frame or a subsequent
        # frame from the monotonic sequence, within the request's freshness
        # bound. Requiring exact equality starves on a continuously streaming Pi.
        started = time.monotonic()
        status = self.status()
        payload = self._get("/api/points", HEADER.size + MAX_POINTS * 12)
        sequence, count = unpack_points(payload)
        previous = status.get("sequence")
        elapsed_ms = (time.monotonic() - started) * 1000
        if isinstance(previous, bool) or not isinstance(previous, int):
            raise ValueError("invalid depth status sequence")
        advance = (sequence - previous) % (2 ** 32)
        # 60 Hz upper bound allows camera configuration changes; large jumps
        # and reconnect/reset regressions cannot make an old frame look fresh.
        if (advance > math.ceil(elapsed_ms * 0.06) + 2
                or status["last_frame_age_ms"] + elapsed_ms > self.max_age_ms):
            raise ValueError("could not obtain a fresh, sequence-matched depth frame")
        return payload, {
            "sequence": sequence, "point_count": count,
            "frame_id": DEPTH_FRAME, "units": "metres",
            "axes": "x-right,y-down,z-forward",
            "received_at_unix": time.time(),
            "timestamp_kind": "receiver_time_not_capture_time",
            "age_upper_bound_ms": status["last_frame_age_ms"] + elapsed_ms,
        }
