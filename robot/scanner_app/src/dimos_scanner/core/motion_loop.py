"""Background thread that pushes the current ``ScanState`` to the robot.

We push at ~30 Hz so motion stays smooth between operator key presses but skip
sends when nothing has changed (the motors hold their last target on their
own — no need to keep retransmitting).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from .scan_state import ScanState

logger = logging.getLogger("dimos_scanner.motion")


class _SupportsSetTarget(Protocol):
    def set_target(self, head=None, body_yaw=None, antennas=None) -> None: ...


def run_motion_loop(
    robot: _SupportsSetTarget,
    scan: ScanState,
    stop_event: threading.Event,
    hz: float = 30.0,
) -> None:
    """Pump ``robot.set_target`` whenever ``scan`` changes. Blocks until stop."""
    period = 1.0 / max(hz, 1.0)
    last = None
    while not stop_event.is_set():
        snapshot = (scan.head_yaw_deg, scan.head_pitch_deg, scan.head_roll_deg, scan.body_yaw_deg)
        if snapshot != last:
            try:
                robot.set_target(head=scan.head_pose(), body_yaw=scan.body_yaw_rad)
            except Exception as e:  # noqa: BLE001
                logger.warning("set_target failed: %s", e)
            last = snapshot
        time.sleep(period)
