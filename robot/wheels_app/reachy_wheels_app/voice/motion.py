"""Head + body-shell motion for the voice agent — the lower movement tiers.

The agent moves on a hierarchy: glance with the head (tier 1), rotate the
body shell (tier 2), and only then roll the wheels (tier 3, tools.py).
This module owns tiers 1–2 as a thin adapter over the ReachyMini SDK.

Frame conventions (SDK: x forward, y left, z up):
  * tool-facing angles are DEGREES, yaw positive = LEFT, pitch positive = UP
  * the SDK's head pitch is positive-down, so we negate internally
  * `look` yaw/pitch are absolute, relative to the body shell's forward
  * `turn_body` is RELATIVE degrees; we track the absolute shell angle

The SDK's ``goto_target`` BLOCKS until the motion finishes and defaults
``body_yaw=0.0`` (which would silently re-center the shell on every head
move) — head-only calls must pass ``body_yaw=None``. Calls are serialized
with a lock: dispatch runs in executor threads and overlapping goto_targets
would fight over the neck.

Import safety: no reachy_mini / numpy at module import — the SDK is only
touched inside methods, so the clamp logic stays offline-testable.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MotionLimits:
    head_yaw: float = 40.0        # deg, each side of body-forward
    head_pitch_up: float = 20.0   # deg
    head_pitch_down: float = 25.0  # deg
    body_yaw: float = 120.0       # deg, each side; cables forbid endless spin
    min_duration: float = 0.3
    max_duration: float = 2.0
    deg_per_second: float = 90.0  # pacing for auto-computed durations
    # Open-loop scaling of head commands, if the neck under-travels.
    # DEFAULT 1.0 = off, deliberately. Measured on this robot the neck is
    # ASYMMETRIC — commanding +15 deg lands at +14.7, commanding -15 lands
    # at -8.9 — so no single scalar describes it, and over-driving the side
    # that already tracks well would make the follow loop overshoot. The
    # loop re-measures the bearing every frame, so an under-travelling neck
    # costs convergence speed, not stability. Set this only if a particular
    # robot is measurably symmetric and short.
    yaw_command_gain: float = 1.0
    pitch_command_gain: float = 1.0


def clamp_angle(value: Any, lo: float, hi: float, default: float = 0.0) -> tuple[float, bool]:
    """(clamped_value, was_clamped). Junk from the model becomes `default`."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    clamped = max(lo, min(hi, v))
    return clamped, clamped != v


def motion_duration(degrees: float, limits: MotionLimits) -> float:
    """A natural-feeling duration for an angular move of this size."""
    d = abs(degrees) / limits.deg_per_second
    return max(limits.min_duration, min(limits.max_duration, d))


class RobotMotion:
    """Owns the neck and body shell; every method returns a JSON-safe dict.

    Hardware posture is read back from the SDK. Cached values are only a
    fallback for SDK-free test doubles. Tool yaw is shell-relative, whereas
    the SDK head pose is robot-base-relative.
    """

    def __init__(self, mini, limits: MotionLimits | None = None,
                 head_pose_fn=None):
        self._mini = mini
        self.limits = limits or MotionLimits()
        # Injectable for offline tests; defaults to the SDK's builder.
        self._head_pose_fn = head_pose_fn
        self._lock = threading.Lock()
        self.head_yaw = 0.0
        self.head_pitch = 0.0   # positive = up (tool convention)
        self.body_yaw = 0.0

    # --- internals ------------------------------------------------------

    def _goto(self, head_yaw: float | None = None, head_pitch: float | None = None,
              body_yaw: float | None = None, duration: float = 0.5,
              commit=None) -> None:
        with self._lock:
            if self._head_pose_fn is None:
                from reachy_mini.utils import create_head_pose
                self._head_pose_fn = create_head_pose

            head = None
            if head_yaw is not None:
                # Ask for more than we want, so we get what we want. Everything
                # outside this line — limits, tracking, the follow controller —
                # is in true angles. Build this under the same lock as the SDK
                # call so another caller cannot change body_yaw between them.
                lm = self.limits
                commanded_yaw = self.body_yaw + head_yaw / max(0.1, lm.yaw_command_gain)
                commanded_pitch = (head_pitch or 0.0) / max(0.1, lm.pitch_command_gain)
                # SDK pitch is positive-down; tools speak positive-up.
                head = self._head_pose_fn(yaw=commanded_yaw, pitch=-commanded_pitch,
                                          degrees=True)
            self._mini.goto_target(
                head=head,
                body_yaw=None if body_yaw is None else math.radians(body_yaw),
                duration=duration,
            )
            # Commit the tracked angles INSIDE the same lock that serialises
            # the SDK call. Publishing them after the lock lets two callers
            # (the follow loop's head thread, the Gemini executor, and
            # /api/motion/look are three) finish out of order, so posture()
            # would report an angle the head is not at — and the follow
            # controller steers the wheels off exactly that number.
            if commit is not None:
                commit()

    def posture(self) -> dict:
        # Posture drives the next closed-loop correction. Do not sample an
        # SDK interpolation halfway through another thread's goto_target and
        # then steer back toward that transient pose. Also do not block the
        # perception loop behind that interpolation: while it is in flight,
        # report the last stable measurement and let the latest-wins slot
        # coalesce the next correction.
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return {"head_yaw": round(self.head_yaw, 1),
                    "head_pitch": round(self.head_pitch, 1),
                    "body_yaw": round(self.body_yaw, 1)}
        try:
            if hasattr(self._mini, "get_current_head_pose"):
                pose = self._mini.get_current_head_pose()
                joints, _ = self._mini.get_current_joint_positions()
                absolute_yaw = math.degrees(math.atan2(pose[1][0], pose[0][0]))
                body = math.degrees(joints[0])
                pitch = math.degrees(math.atan2(
                    pose[2][0], math.hypot(pose[0][0], pose[1][0])))
                relative = (absolute_yaw - body + 180.0) % 360.0 - 180.0
                if not all(math.isfinite(v) for v in (relative, pitch, body)):
                    raise ValueError("invalid measured robot posture")
                self.head_yaw, self.head_pitch, self.body_yaw = relative, pitch, body
            return {"head_yaw": round(self.head_yaw, 1),
                    "head_pitch": round(self.head_pitch, 1),
                    "body_yaw": round(self.body_yaw, 1)}
        finally:
            self._lock.release()

    _posture = posture  # internal alias

    # --- tiers ----------------------------------------------------------

    def look(self, yaw: Any = 0.0, pitch: Any = 0.0) -> dict:
        """Tier 1: point the head. Absolute degrees relative to the shell."""
        self.posture()
        lm = self.limits
        yaw_v, yaw_clamped = clamp_angle(yaw, -lm.head_yaw, lm.head_yaw)
        pitch_v, pitch_clamped = clamp_angle(
            pitch, -lm.head_pitch_down, lm.head_pitch_up)
        moved = max(abs(yaw_v - self.head_yaw), abs(pitch_v - self.head_pitch))
        if moved < 0.5:
            # Search and follow run faster than the blocking SDK actuator.
            # Re-sending an already-reached pose makes the neck audibly hunt
            # around encoder noise and was the direct cause of the rattling.
            out = {"status": "ok", **self._posture(), "unchanged": True}
            if yaw_clamped or pitch_clamped:
                out["note"] = "requested head pose is already at its safe limit"
            return out
        def commit():
            self.head_yaw, self.head_pitch = yaw_v, pitch_v

        self._goto(head_yaw=yaw_v, head_pitch=pitch_v, body_yaw=None,
                   duration=motion_duration(max(moved, 20.0), self.limits),
                   commit=commit)
        out = {"status": "ok", **self._posture()}
        if yaw_clamped:
            out["note"] = ("head yaw is at its limit — call turn_body to face "
                           "further that way")
        elif pitch_clamped:
            out["note"] = "head pitch limit reached"
        return out

    def turn_body(self, degrees: Any) -> dict:
        """Tier 2: rotate the body shell by `degrees` (positive = left)."""
        self.posture()
        lm = self.limits
        step, _ = clamp_angle(degrees, -2 * lm.body_yaw, 2 * lm.body_yaw)
        target = max(-lm.body_yaw, min(lm.body_yaw, self.body_yaw + step))
        applied = target - self.body_yaw
        if abs(applied) >= 0.5:
            self._goto(head_yaw=self.head_yaw + applied * self.limits.yaw_command_gain,
                       head_pitch=self.head_pitch, body_yaw=target,
                       duration=motion_duration(applied, self.limits),
                       commit=lambda: setattr(self, "body_yaw", target))
        out = {"status": "ok", "turned": round(applied, 1), **self._posture(),
               # Remaining sweep each way, so the model can plan one big
               # rotation instead of discovering the limit by bumping it.
               "can_turn_left": round(lm.body_yaw - self.body_yaw, 1),
               "can_turn_right": round(lm.body_yaw + self.body_yaw, 1)}
        if abs(applied - step) >= 0.5:
            out["note"] = ("body shell is at its rotation limit — use the "
                           "drive tool (rotate_ccw for left, rotate_cw for "
                           "right) to keep turning with the wheels")
        return out

    def look_with_body(self, yaw: float, pitch: float, delta: float) -> dict:
        """Transfer gaze to the shell without rotating the camera twice.

        yaw is relative to the OLD shell. Send that absolute gaze and the
        new shell angle in one SDK interpolation; the neck compensates.
        """
        self.posture()
        lm = self.limits
        target = max(-lm.body_yaw, min(lm.body_yaw, self.body_yaw + delta))
        applied = target - self.body_yaw
        pitch, _ = clamp_angle(pitch, -lm.head_pitch_down, lm.head_pitch_up)
        relative = max(-lm.head_yaw, min(lm.head_yaw, yaw - applied))
        # _goto adds the old shell angle, so compensate any neck-limit clamp.
        def commit():
            self.body_yaw = target
            self.head_yaw, self.head_pitch = relative, pitch
        self._goto(head_yaw=relative + applied, head_pitch=pitch,
                   body_yaw=target, duration=motion_duration(max(abs(applied), 20), lm),
                   commit=commit)
        return {"status": "ok", **self.posture()}

    def center(self) -> dict:
        """Head and body shell back to neutral, facing wheel-forward."""
        def commit():
            self.head_yaw = self.head_pitch = self.body_yaw = 0.0

        self.posture()
        self._goto(head_yaw=-self.body_yaw * self.limits.yaw_command_gain,
                   head_pitch=0.0, body_yaw=0.0, duration=0.8,
                   commit=commit)
        return {"status": "ok", **self._posture()}
