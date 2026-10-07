"""Value types shared by the detect → track → follow pipeline.

Deliberately dependency-free (no numpy, no cv2, no SDK) so the controller
and its tests stay importable anywhere.

Boxes are pixel ``(x1, y1, x2, y2)`` in the frame the detector was given,
top-left origin, x right / y down — the convention every detector in this
package normalises to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame."""

    bbox: tuple[float, float, float, float]
    score: float = 0.0
    class_name: str = ""
    class_id: int = -1
    # Filled in by the tracker; None until a track has been associated.
    track_id: int | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def with_track(self, track_id: int | None) -> "Detection":
        return replace(self, track_id=track_id)

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "score": round(float(self.score), 3),
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class FrameInfo:
    """Geometry of the frame a detection came from."""

    width: int
    height: int
    # Head-camera field of view in degrees. Overridable from config because
    # it is the one number that maps pixels → angles, and therefore the one
    # that decides whether the servo loop over- or under-turns.
    hfov_deg: float = 70.0
    vfov_deg: float = 55.0

    def bearing_deg(self, cx: float) -> float:
        """Horizontal angle from image centre. Positive = target is RIGHT.

        Real pinhole projection, ``x = f·tan(θ)``, not the linear
        ``(cx/W − 0.5)·HFOV`` shortcut. The two agree at the frame edge and
        nowhere else: on this robot's ~90° lens the linear form
        underestimates a centred target's bearing by about 30%, so the
        controller would consistently under-turn near the middle — which is
        precisely where a follow loop spends its time.
        """
        if self.width <= 0:
            return 0.0
        half = math.tan(math.radians(self.hfov_deg) / 2.0)
        return math.degrees(math.atan((cx / self.width - 0.5) * 2.0 * half))

    def elevation_deg(self, cy: float) -> float:
        """Vertical angle from image centre. Positive = target is ABOVE."""
        if self.height <= 0:
            return 0.0
        half = math.tan(math.radians(self.vfov_deg) / 2.0)
        return math.degrees(math.atan((0.5 - cy / self.height) * 2.0 * half))

    @property
    def focal_px(self) -> float:
        """Focal length in pixels implied by the horizontal FOV."""
        half = math.tan(math.radians(self.hfov_deg) / 2.0)
        return self.width / (2.0 * half) if half > 0 else 0.0


def robot_to_chassis(vx: float, vy: float, mount_yaw_deg: float) -> tuple[float, float]:
    """Rotate a ROBOT-frame velocity into the CHASSIS's frame.

    The robot can be bolted onto the chassis facing any direction, and the
    follow controller has no business knowing which. It says "go the way I
    am looking"; this turns that into the mix the board understands.

    ``mount_yaw_deg`` is the angle from chassis-forward to robot-facing,
    counter-clockwise. At -90 the robot faces the chassis's right, so
    robot-forward (1, 0) becomes chassis (0, -1) — a strafe to the right.
    """
    theta = math.radians(mount_yaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (vx * cos_t - vy * sin_t, vx * sin_t + vy * cos_t)
