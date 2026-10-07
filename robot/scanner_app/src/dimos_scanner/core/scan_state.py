"""Head/body pose tracker for arrow-key scanning.

Vendored from the station-side motion module in ``../../station/`` so this
app stays installable on the robot on its own. Keep the two copies in sync;
the station copy is the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


MAX_HEAD_PITCH_DEG = 25.0
MAX_HEAD_YAW_DEG = 35.0
MAX_BODY_YAW_DEG = 90.0


@dataclass
class ScanState:
    """Mutable scan pose. Operator nudges via arrow-key deltas."""

    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0
    body_yaw_deg: float = 0.0
    step_deg: float = 5.0

    def head_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler(
            "xyz",
            [self.head_roll_deg, self.head_pitch_deg, self.head_yaw_deg],
            degrees=True,
        ).as_matrix()
        return pose

    @property
    def body_yaw_rad(self) -> float:
        return float(np.deg2rad(self.body_yaw_deg))

    def apply(self, action: str, step_deg: float | None = None) -> None:
        s = step_deg if step_deg is not None else self.step_deg
        if action == "yaw_left":
            self._nudge_yaw(+s)
        elif action == "yaw_right":
            self._nudge_yaw(-s)
        elif action == "pitch_up":
            self.head_pitch_deg = float(
                np.clip(self.head_pitch_deg - s, -MAX_HEAD_PITCH_DEG, MAX_HEAD_PITCH_DEG)
            )
        elif action == "pitch_down":
            self.head_pitch_deg = float(
                np.clip(self.head_pitch_deg + s, -MAX_HEAD_PITCH_DEG, MAX_HEAD_PITCH_DEG)
            )
        elif action == "reset":
            self.head_yaw_deg = 0.0
            self.head_pitch_deg = 0.0
            self.head_roll_deg = 0.0
            self.body_yaw_deg = 0.0

    def _nudge_yaw(self, delta_deg: float) -> None:
        new_head = self.head_yaw_deg + delta_deg
        if -MAX_HEAD_YAW_DEG <= new_head <= MAX_HEAD_YAW_DEG:
            self.head_yaw_deg = new_head
            return
        if new_head > MAX_HEAD_YAW_DEG:
            overshoot = new_head - MAX_HEAD_YAW_DEG
            self.head_yaw_deg = MAX_HEAD_YAW_DEG
            self.body_yaw_deg = float(
                np.clip(self.body_yaw_deg + overshoot, -MAX_BODY_YAW_DEG, MAX_BODY_YAW_DEG)
            )
        else:
            overshoot = new_head + MAX_HEAD_YAW_DEG
            self.head_yaw_deg = -MAX_HEAD_YAW_DEG
            self.body_yaw_deg = float(
                np.clip(self.body_yaw_deg + overshoot, -MAX_BODY_YAW_DEG, MAX_BODY_YAW_DEG)
            )
