"""RobotMotion (head/body tiers) + hierarchy dispatch. Offline — no SDK.

The SDK trap this suite guards hardest: ``goto_target`` defaults
``body_yaw=0.0``, so any head-only move that forgets ``body_yaw=None``
silently spins the body shell back to centre mid-conversation.
"""

from __future__ import annotations

import math
import os
import time

import pytest
import subprocess
import sys
from pathlib import Path

from reachy_wheels_app.voice.motion import (
    MotionLimits,
    RobotMotion,
    clamp_angle,
    motion_duration,
)
from reachy_wheels_app.voice.tools import (
    MOTION_TOOL_DECLS,
    dispatch_tool,
    response_scheduling,
)

APP_ROOT = Path(__file__).resolve().parents[1]


def _fake_head_pose(yaw=0.0, pitch=0.0, degrees=True):
    return {"yaw": yaw, "pitch": pitch, "degrees": degrees}


class FakeMini:
    def __init__(self):
        self.gotos = []

    def goto_target(self, head=None, body_yaw=0.0, duration=0.5, **kw):
        self.gotos.append({"head": head, "body_yaw": body_yaw,
                           "duration": duration})


def make_motion():
    mini = FakeMini()
    return mini, RobotMotion(mini, head_pose_fn=_fake_head_pose)


def test_imports_stay_light():
    """Importing the app must not drag in the SDK or numpy.

    Checked in a fresh interpreter rather than against this process's
    ``sys.modules``: other tests in the suite legitimately import numpy, and
    the property we care about is the *app's* import graph — what the robot
    pays for at start-up — not what pytest happens to have loaded.
    """
    probe = (
        "import sys;"
        "import reachy_wheels_app.api, reachy_wheels_app.voice.motion,"
        " reachy_wheels_app.voice.tools, reachy_wheels_app.tracking;"
        "print(','.join(m for m in ('reachy_mini', 'numpy', 'cv2', 'onnxruntime',"
        " 'google.genai') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(APP_ROOT)})
    assert out.stdout.strip() == "", f"heavy imports at module load: {out.stdout}"


def test_clamp_angle_handles_junk():
    assert clamp_angle(999, -40, 40) == (40.0, True)
    assert clamp_angle("sideways", -40, 40) == (0.0, False)
    assert clamp_angle(-10, -40, 40) == (-10.0, False)


def test_motion_duration_bounds():
    lm = MotionLimits()
    assert motion_duration(1, lm) == lm.min_duration
    assert motion_duration(100000, lm) == lm.max_duration


def test_look_keeps_body_yaw_and_negates_pitch():
    # Gain 1.0 so this pins the frame conventions, not the neck calibration
    # (which is exercised separately below).
    mini = FakeMini()
    motion = RobotMotion(mini, limits=MotionLimits(yaw_command_gain=1.0),
                         head_pose_fn=_fake_head_pose)
    out = motion.look(yaw=20, pitch=10)
    assert out["status"] == "ok"
    assert (out["head_yaw"], out["head_pitch"]) == (20.0, 10.0)
    goto = mini.gotos[-1]
    assert goto["body_yaw"] is None            # the body_yaw=0.0 reset trap
    assert goto["head"]["yaw"] == 20.0
    assert goto["head"]["pitch"] == -10.0      # tool up(+) → SDK down(+)


def test_look_at_yaw_limit_points_to_turn_body():
    _, motion = make_motion()
    out = motion.look(yaw=90)
    assert out["head_yaw"] == motion.limits.head_yaw
    assert "turn_body" in out["note"]
    # A limit note must reach the model, not stay silent.
    assert response_scheduling("look", out) == "WHEN_IDLE"


def test_turn_body_is_relative_and_tracked():
    mini, motion = make_motion()
    assert motion.turn_body(45)["body_yaw"] == 45.0
    out = motion.turn_body(-30)
    assert out["body_yaw"] == 15.0 and out["turned"] == -30.0
    assert "note" not in out
    # Envelope awareness: remaining sweep both ways, so the model can plan
    # one big rotation instead of probing for the limit.
    assert out["can_turn_left"] == 105.0 and out["can_turn_right"] == 135.0
    # Absolute target reaches the SDK in radians — degrees never leak through.
    assert math.isclose(mini.gotos[-1]["body_yaw"], math.radians(15))
    assert response_scheduling("turn_body", out) == "SILENT"


def test_turn_body_limit_escalates_to_wheels():
    _, motion = make_motion()
    motion.turn_body(100)
    out = motion.turn_body(100)                 # would pass +120 limit
    assert out["body_yaw"] == motion.limits.body_yaw
    assert out["turned"] == 20.0
    assert "drive" in out["note"] and "rotate_ccw" in out["note"]
    assert response_scheduling("turn_body", out) == "WHEN_IDLE"


def test_center_resets_everything():
    mini, motion = make_motion()
    motion.look(yaw=30, pitch=-10)
    motion.turn_body(-60)
    out = motion.center()
    assert (out["head_yaw"], out["head_pitch"], out["body_yaw"]) == (0, 0, 0)
    goto = mini.gotos[-1]
    assert goto["body_yaw"] == 0.0 and goto["head"]["yaw"] == 0.0


def test_motion_decls_are_non_blocking():
    assert {d["name"] for d in MOTION_TOOL_DECLS} == {"look", "turn_body", "center"}
    assert all(d["behavior"] == "NON_BLOCKING" for d in MOTION_TOOL_DECLS)


def test_dispatch_routes_motion_tools():
    _, motion = make_motion()
    out = dispatch_tool(None, "look", {"yaw": 15}, motion=motion)
    assert out["status"] == "ok" and out["head_yaw"] == 15.0
    out = dispatch_tool(None, "turn_body", {"degrees": -20}, motion=motion)
    assert out["status"] == "ok" and out["turned"] == -20.0
    assert dispatch_tool(None, "center", {}, motion=motion)["status"] == "ok"


def test_dispatch_without_motion_reports_unavailable():
    out = dispatch_tool(None, "look", {"yaw": 15}, motion=None)
    assert out["status"] == "error" and "not available" in out["error"]
    assert response_scheduling("look", out) == "INTERRUPT"


def test_dispatch_survives_a_raising_mini():
    class ExplodingMini:
        def goto_target(self, **kw):
            raise RuntimeError("servo fault")

    motion = RobotMotion(ExplodingMini(), head_pose_fn=_fake_head_pose)
    out = dispatch_tool(None, "look", {"yaw": 5}, motion=motion)
    assert out["status"] == "error"


def test_head_commands_are_scaled_up_to_hit_the_angle_asked_for():
    """The neck under-travels (~0.91x on this robot). Without compensation
    the head never reaches the target it is centring, and the tracked angle
    the follow controller steers the wheels from overstates reality."""
    mini = FakeMini()
    motion = RobotMotion(mini, limits=MotionLimits(yaw_command_gain=0.909),
                         head_pose_fn=_fake_head_pose)
    motion.look(yaw=10.0, pitch=0.0)
    sent = mini.gotos[-1]["head"]["yaw"]
    assert sent == pytest.approx(11.0, abs=0.05)   # 10 / 0.909
    # …but the TRUE angle is what gets tracked and reported.
    assert motion.head_yaw == pytest.approx(10.0)
    assert motion.posture()["head_yaw"] == pytest.approx(10.0)


def test_a_gain_of_one_sends_the_angle_unchanged():
    mini = FakeMini()
    motion = RobotMotion(mini, limits=MotionLimits(yaw_command_gain=1.0),
                         head_pose_fn=_fake_head_pose)
    motion.look(yaw=10.0, pitch=0.0)
    assert mini.gotos[-1]["head"]["yaw"] == pytest.approx(10.0)


def test_repeated_measured_head_pose_is_a_noop_not_another_interpolation():
    mini = MeasuredMini(head=10, body=0)
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    out = motion.look(yaw=10.2, pitch=0.0)
    assert out["unchanged"] is True
    assert mini.gotos == []


def test_posture_does_not_block_or_sample_during_an_interpolation():
    mini = MeasuredMini(head=15, body=0)
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    assert motion.posture()["head_yaw"] == 15
    motion._lock.acquire()
    try:
        mini.head = 7  # transient SDK pose must not leak into control
        started = time.monotonic()
        posture = motion.posture()
        assert time.monotonic() - started < 0.05
        assert posture["head_yaw"] == 15
    finally:
        motion._lock.release()


class MeasuredMini(FakeMini):
    """SDK semantics: head pose is absolute, NOT relative to body yaw."""
    def __init__(self, head=60, body=30):
        super().__init__()
        self.head, self.body = head, body

    def goto_target(self, head=None, body_yaw=None, **kw):
        super().goto_target(head=head, body_yaw=body_yaw, **kw)
        if head is not None:
            self.head = head["yaw"]
        if body_yaw is not None:
            self.body = math.degrees(body_yaw)

    def get_current_head_pose(self):
        r = math.radians(self.head)
        return [[math.cos(r), -math.sin(r), 0, 0],
                [math.sin(r), math.cos(r), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def get_current_joint_positions(self):
        return [math.radians(self.body)] + [0] * 6, [0, 0]


def test_measured_posture_and_shell_relative_command_conversion():
    mini = MeasuredMini()
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    assert motion.posture()["head_yaw"] == 30
    motion.look(10, 0)
    assert mini.head == 40  # 30-degree shell + 10-degree neck
    mini.head = 45  # external motion, not one of our cached commands
    assert motion.posture()["head_yaw"] == 15


def test_shell_transfer_keeps_absolute_gaze_and_reduces_neck_offset():
    mini = MeasuredMini()
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    motion.look_with_body(30, 0, 6)
    assert mini.head == 60
    assert mini.body == 36
    assert motion.posture()["head_yaw"] == 24
    assert len(mini.gotos) == 1


def test_center_with_rotated_shell_commands_absolute_zero():
    mini = MeasuredMini()
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    motion.center()
    assert mini.head == 0 and mini.body == 0


def test_body_search_carries_camera_with_shell_at_neck_limit():
    mini = MeasuredMini(head=40, body=0)
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    motion.look_with_body(46, 0, 6)
    assert (mini.head, mini.body) == pytest.approx((46, 6))
    assert motion.posture()["head_yaw"] == 40


def test_voice_body_turn_preserves_relative_neck_angle():
    mini = MeasuredMini(head=40, body=30)
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    motion.turn_body(10)
    assert mini.head == 50 and mini.body == 40
    assert motion.posture()["head_yaw"] == 10


@pytest.mark.parametrize("side", [-1, 1])
def test_closed_loop_shell_transfer_then_chassis_alignment_then_approach(side):
    from reachy_wheels_app.tracking.follow import FollowController
    from reachy_wheels_app.tracking.types import Detection, FrameInfo
    frame = FrameInfo(width=640, height=480)
    mini = MeasuredMini(head=60 * side, body=30 * side)
    motion = RobotMotion(mini, head_pose_fn=_fake_head_pose)
    ctrl = FollowController()
    target_world, chassis = 60 * side, 0.0
    transferred = approached = False
    for i in range(150):
        posture = motion.posture()
        bearing = chassis + mini.head - target_world
        assert abs(bearing) < frame.hfov_deg / 2, "lost target during coordinated motion"
        cx = frame.width / 2 + frame.focal_px * math.tan(math.radians(bearing))
        target = Detection(bbox=(cx-30, 190, cx+30, 290), score=.9,
                           class_name="person", track_id=1)
        cmd = ctrl.step(target, frame, now=i*.2, **posture)
        yaw = cmd.head_yaw if cmd.head_yaw is not None else posture["head_yaw"]
        if cmd.body_yaw_delta:
            transferred = True
            assert cmd.omega == 0
            motion.look_with_body(yaw, 0, cmd.body_yaw_delta)
        elif cmd.head_yaw is not None:
            motion.look(yaw, 0)
        chassis += cmd.omega * 80 * .2
        approached |= cmd.vx > 0
    assert transferred and approached
    assert abs(cmd.yaw_error_deg) < ctrl.config.approach_gate_deg
