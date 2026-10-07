"""Rehearse the scanner app's I/O loop without a real Reachy Mini.

Useful when you want to debug the bridge handshake / frame encoding pipeline
on a laptop that doesn't have a robot daemon running. We stub out the
ReachyMini class with one that exposes a synthetic camera (a colour-changing
gradient) and a no-op set_target.

Usage::

    DIMOS_SCANNER_BRIDGE_HOST=localhost \\
    python examples/stub_run.py

Make sure the station bridge (see ``../../station/``) is running first and
listening on DIMOS_SCANNER_BRIDGE_PORT (default 9876).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import numpy as np

from dimos_scanner.config import BridgeConfig
from dimos_scanner.core.scan_state import ScanState
from dimos_scanner.io.bridge_client import BridgeClient


def synthetic_frame(t: float, w: int = 320, h: int = 240) -> np.ndarray:
    """Return a BGR frame that animates a vertical hue sweep."""
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    # Time-varying horizontal gradient. Cheap, no cv2 colour-conversion.
    phase = (np.linspace(0, 255, w, dtype=np.float32) + 50 * t) % 255
    bgr[..., 0] = phase[None, :].astype(np.uint8)
    bgr[..., 1] = ((phase[None, :] + 85) % 255).astype(np.uint8)
    bgr[..., 2] = ((phase[None, :] + 170) % 255).astype(np.uint8)
    return bgr


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = BridgeConfig.from_env()
    if not cfg.host:
        raise SystemExit("set DIMOS_SCANNER_BRIDGE_HOST first")

    scan = ScanState()
    t0 = time.time()
    stop_event = threading.Event()

    def producer() -> np.ndarray:
        return synthetic_frame(time.time() - t0)

    client = BridgeClient(
        host=cfg.host,
        port=cfg.port,
        frame_producer=producer,
        scan=scan,
        jpeg_quality=cfg.jpeg_quality,
        frame_hz=cfg.frame_hz,
        name="stub_run",
    )

    def watch_state() -> None:
        last = None
        while not stop_event.is_set():
            snap = (scan.head_yaw_deg, scan.head_pitch_deg, scan.body_yaw_deg)
            if snap != last:
                print(
                    f"[stub] scan_state: yaw={scan.head_yaw_deg:+.1f}° "
                    f"pitch={scan.head_pitch_deg:+.1f}° body={scan.body_yaw_deg:+.1f}°"
                )
                last = snap
            time.sleep(0.1)

    threading.Thread(target=watch_state, daemon=True).start()

    try:
        asyncio.run(client.run(stop_event))
    except KeyboardInterrupt:
        stop_event.set()


if __name__ == "__main__":
    main()
