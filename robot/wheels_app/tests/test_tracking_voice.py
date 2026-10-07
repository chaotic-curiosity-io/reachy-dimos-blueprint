"""The voice agent's follow/stop_following tools and their prompt."""

from __future__ import annotations

from reachy_wheels_app.voice.gemini_live import build_system_prompt
from reachy_wheels_app.voice.tools import (
    DRIVE_TOOL_DECLS,
    FOLLOW_TOOL_DECLS,
    DriveLimits,
    describe_tool_result,
    dispatch_tool,
    response_scheduling,
)


class FakeWheels:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1
        return {"ok": True}


class FakeFollow:
    def __init__(self, active=True):
        self.started: list[tuple] = []
        self.stopped: list[str] = []
        self.active = active

    def start(self, target, distance_cm=None):
        self.started.append((target, distance_cm))
        return {"status": "ok", "action": f"following {target}",
                "target": target, "labels": ["person"]}

    def stop(self, reason="stopped"):
        self.stopped.append(reason)
        was, self.active = self.active, False
        return {"status": "ok", "was_active": was}


# --- declarations ---------------------------------------------------------

def test_follow_tools_are_non_blocking_like_the_rest():
    assert {d["name"] for d in FOLLOW_TOOL_DECLS} == {"follow", "stop_following"}
    for decl in FOLLOW_TOOL_DECLS:
        assert decl["behavior"] == "NON_BLOCKING"


def test_follow_requires_a_target_argument():
    follow = next(d for d in FOLLOW_TOOL_DECLS if d["name"] == "follow")
    assert follow["parameters"]["required"] == ["target"]
    assert "distance_cm" in follow["parameters"]["properties"]


def test_tool_names_do_not_collide_with_the_drive_tools():
    drive = {d["name"] for d in DRIVE_TOOL_DECLS}
    assert drive.isdisjoint({d["name"] for d in FOLLOW_TOOL_DECLS})


# --- dispatch -------------------------------------------------------------

def test_follow_call_reaches_the_manager():
    follow = FakeFollow()
    out = dispatch_tool(FakeWheels(), "follow", {"target": "the dog"},
                        follow=follow)
    assert out["status"] == "ok"
    assert follow.started == [("the dog", None)]


def test_follow_forwards_a_requested_distance():
    follow = FakeFollow()
    dispatch_tool(FakeWheels(), "follow",
                  {"target": "me", "distance_cm": 90}, follow=follow)
    assert follow.started == [("me", 90.0)]


def test_a_junk_distance_is_dropped_rather_than_crashing():
    follow = FakeFollow()
    dispatch_tool(FakeWheels(), "follow",
                  {"target": "me", "distance_cm": "close-ish"}, follow=follow)
    assert follow.started == [("me", None)]


def test_stop_following_ends_the_session():
    follow = FakeFollow()
    assert dispatch_tool(FakeWheels(), "stop_following", {},
                         follow=follow)["status"] == "ok"
    assert follow.stopped


def test_plain_stop_also_ends_a_follow():
    # "Stop" from an alarmed user has to mean everything, not just the
    # chassis — the follow loop would re-command it a tick later.
    wheels, follow = FakeWheels(), FakeFollow(active=True)
    out = dispatch_tool(wheels, "stop", {}, follow=follow)
    assert out["status"] == "ok"
    assert wheels.stops == 1
    assert follow.stopped
    assert "also ended the follow" in out["note"]


def test_plain_stop_without_a_follow_is_unchanged():
    wheels = FakeWheels()
    out = dispatch_tool(wheels, "stop", {})
    assert out == {"status": "ok", "action": "stopped"}
    assert wheels.stops == 1


def test_follow_tools_report_clearly_when_unavailable():
    for name in ("follow", "stop_following"):
        out = dispatch_tool(FakeWheels(), name, {"target": "me"})
        assert out["status"] == "error"
        assert "not available" in out["error"]


def test_a_manager_error_comes_back_as_a_tool_error():
    class Refusing(FakeFollow):
        def start(self, target, distance_cm=None):
            return {"status": "error", "error": "no class for unicorn"}

    out = dispatch_tool(FakeWheels(), "follow", {"target": "unicorn"},
                        follow=Refusing())
    assert out["status"] == "error"
    assert response_scheduling("follow", out) == "INTERRUPT"


def test_follow_outcomes_are_delivered_rather_than_absorbed():
    # A follow changes what the robot does for the next minute; the model
    # should acknowledge it out loud instead of swallowing it.
    ok = {"status": "ok", "action": "following me"}
    assert response_scheduling("follow", ok) == "WHEN_IDLE"
    assert response_scheduling("stop_following", ok) == "WHEN_IDLE"


def test_transcript_line_names_the_target():
    line = describe_tool_result("follow", {"target": "the dog"},
                                {"status": "ok", "target": "the dog"})
    assert "follow the dog" in line


# --- the prompt -----------------------------------------------------------

def test_prompt_teaches_the_tracker_only_when_it_exists():
    tracked = build_system_prompt(DriveLimits(), can_follow=True)
    manual = build_system_prompt(DriveLimits(), can_follow=False)
    assert "`follow`" in tracked and "stop_following" in tracked
    assert "short legs" in manual and "`follow`" not in manual


def test_prompt_forbids_hand_steering_over_the_tracker():
    tracked = build_system_prompt(DriveLimits(), can_follow=True)
    assert "fight your own tracker" in tracked


# --- the robot's mounting on the chassis ---------------------------------

def test_primitives_pass_through_an_unrotated_mount():
    from reachy_wheels_app.voice.tools import remap_primitive
    for name in ("forward", "reverse", "strafe_left", "diagonal_fr"):
        assert remap_primitive(name, 0.0) == name


def test_forward_becomes_a_right_strafe_on_a_90_degree_mount():
    from reachy_wheels_app.voice.tools import remap_primitive
    assert remap_primitive("forward", -90.0) == "strafe_right"
    assert remap_primitive("reverse", -90.0) == "strafe_left"
    assert remap_primitive("strafe_left", -90.0) == "forward"
    assert remap_primitive("strafe_right", -90.0) == "reverse"


def test_diagonals_rotate_with_the_mount():
    from reachy_wheels_app.voice.tools import remap_primitive
    assert remap_primitive("diagonal_fl", -90.0) == "diagonal_fr"


def test_rotations_are_mount_independent():
    # Spinning about the vertical axis turns the robot the same way no
    # matter which way it is bolted on.
    from reachy_wheels_app.voice.tools import remap_primitive
    for yaw in (0.0, -90.0, 90.0, 180.0):
        assert remap_primitive("rotate_cw", yaw) == "rotate_cw"
        assert remap_primitive("rotate_ccw", yaw) == "rotate_ccw"


def test_a_drive_call_reaches_the_board_in_chassis_axes():
    """The model says 'forward' meaning the robot's forward; with the robot
    mounted 90° round the board must receive a strafe."""
    class Recorder(FakeWheels):
        def __init__(self):
            super().__init__()
            self.sent = []

        def command(self, command, speed=None, duration=None, **extra):
            self.sent.append(command)
            return {"ok": True}

    wheels = Recorder()
    out = dispatch_tool(wheels, "drive", {"direction": "forward"},
                        limits=DriveLimits(mount_yaw_deg=-90.0))
    assert out["status"] == "ok"
    assert wheels.sent == ["strafe_right"]
    # …but the model is told what it asked for, in its own frame.
    assert out["direction"] == "forward"
