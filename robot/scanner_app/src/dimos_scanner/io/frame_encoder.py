"""JPEG-encode camera frames for transport over the bridge."""

from __future__ import annotations

import cv2
import numpy as np


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> tuple[bytes, int, int]:
    """Encode a BGR ndarray as a JPEG byte string.

    Returns ``(jpeg_bytes, width, height)``. Raises ``ValueError`` if encoding
    fails (cv2 returns ok=False).
    """
    if bgr.dtype != np.uint8:
        bgr = bgr.astype(np.uint8)
    h, w = bgr.shape[:2]
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(10, min(95, quality)))]
    ok, jpeg = cv2.imencode(".jpg", bgr, params)
    if not ok:
        raise ValueError("cv2.imencode returned ok=False")
    return jpeg.tobytes(), w, h
