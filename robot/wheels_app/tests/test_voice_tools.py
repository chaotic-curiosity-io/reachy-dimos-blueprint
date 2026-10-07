"""Voice tool dispatch: clamps, mapping to WheelsClient, error surfacing.

Pure offline — no google-genai, no board, no SDK.
"""

from __future__ import annotations

import sys

import pytest

from reachy_wheels_app.voice.tools import (
    DRIVE_TOOL_DECLS,
    DriveLimits,
    clamp_drive_args,
    describe_tool_result,
    dispatch_tool,
    resolve_drive,
    response_scheduling,
)
from reachy_wheels_app.wheels_client import PRIMITIVES, WheelsError


class FakeWheels:
    def __init__(self):
        self.calls = []
        self.reachable = True

    def command(self, command, speed=None, duration=None, **extra):
        if not self.reachable:
            raise WheelsError("chassis unreachable")
        self.calls.append((command, speed, duration))
        return {"ok": True}

    def stop(self):
        if not self.reachable:
            raise WheelsError("chassis unreachable")
        self.calls.append(("stop", None, None))
        return {"ok": True}

    def state(self):
        return {"moving": False, "last_command": "stop", "wheels": {}}


def test_imports_stay_light():
    # The tools module must not drag in google-genai or the robot SDK.
    assert "google.genai" not in sys.modules
    assert "reachy_mini" not in sys.modules


def test_decls_cover_every_primitive():
    drive = next(d for d in DRIVE_TOOL_DECLS if d["name"] == "drive")
    assert drive["parameters"]["properties"]["direction"]["enum"] == list(PRIMITIVES)


def test_decls_are_non_blocking():
    # Latency contract: the model must be able to keep talking while a tool
    # runs; a decl silently reverting to BLOCKING re-introduces the dead air.
    assert all(d["behavior"] == "NON_BLOCKING" for d in DRIVE_TOOL_DECLS)


def test_response_scheduling_policy():
    assert response_scheduling("drive", {"status": "ok"}) == "SILENT"
    assert response_scheduling("stop", {"status": "ok"}) == "SILENT"
    assert response_scheduling("wheels_state", {"status": "ok"}) == "WHEN_IDLE"
    # Any failure must be spoken immediately, whatever the tool.
    for name in ("drive", "stop", "wheels_state"):
        assert response_scheduling(name, {"status": "error", "error": "x"}) \
            == "INTERRUPT"


def test_clamps():
    limits = DriveLimits()
    d, s, t = clamp_drive_args({"direction": "forward", "speed": 9, "duration": 60}, limits)
    assert (d, s, t) == ("forward", 1.0, 4.0)
    d, s, t = clamp_drive_args({"direction": "ROTATE_CW", "speed": 0.01, "duration": 0.01}, limits)
    assert (d, s, t) == ("rotate_cw", 0.2, 0.2)
    d, s, t = clamp_drive_args({"direction": "reverse"}, limits)
    assert (s, t) == (limits.default_speed, limits.default_duration)
    # Non-numeric junk from the model falls back to defaults, not a crash.
    d, s, t = clamp_drive_args({"direction": "forward", "speed": "fast", "duration": None}, limits)
    assert (s, t) == (limits.default_speed, limits.default_duration)
    with pytest.raises(ValueError):
        clamp_drive_args({"direction": "warp"}, limits)


def test_resolve_drive_from_distance():
    # 60 cm at default 0.8 (30 cm/s ref rate) → 2.0 s, no truncation note.
    lm = DriveLimits()
    d, s, t, extras = resolve_drive({"direction": "forward", "distance_cm": 60}, lm)
    assert (d, s, t) == ("forward", 0.8, 2.0)
    assert extras["estimated_cm"] == 60.0
    assert "note" not in extras


def test_resolve_drive_truncates_and_says_so():
    lm = DriveLimits()  # max 4 s → 120 cm at default speed
    _, _, t, extras = resolve_drive(
        {"direction": "forward", "distance_cm": 300}, lm)
    assert t == 4.0
    assert extras["estimated_cm"] == 120.0
    assert "call drive again" in extras["note"]
    # The note must reach the model so it chains the next leg.
    result = {"status": "ok", **extras}
    assert response_scheduling("drive", result) == "WHEN_IDLE"


def test_resolve_drive_degrees_for_rotation():
    lm = DriveLimits()  # 80 deg/s at ref 0.8
    d, s, t, extras = resolve_drive(
        {"direction": "rotate_ccw", "degrees": 90}, lm)
    assert d == "rotate_ccw" and abs(t - 90 / 80) < 1e-9
    assert extras["estimated_deg"] == 90.0


def test_resolve_drive_scales_rate_with_speed():
    lm = DriveLimits()
    _, _, t, _ = resolve_drive(
        {"direction": "forward", "distance_cm": 60, "speed": 0.4}, lm)
    assert t == 4.0  # half speed → 15 cm/s → 60 cm again hits the 4 s cap


def test_resolve_drive_explicit_duration_wins():
    lm = DriveLimits()
    _, _, t, extras = resolve_drive(
        {"direction": "forward", "distance_cm": 300, "duration": 1.0}, lm)
    assert t == 1.0 and "note" not in extras
    assert extras["estimated_cm"] == 30.0


def test_drive_decl_offers_physical_units():
    drive = next(d for d in DRIVE_TOOL_DECLS if d["name"] == "drive")
    props = drive["parameters"]["properties"]
    assert "distance_cm" in props and "degrees" in props


def test_dispatch_drive_and_stop_and_state():
    bot = FakeWheels()
    out = dispatch_tool(bot, "drive", {"direction": "strafe_left", "duration": 2})
    assert out["status"] == "ok" and out["direction"] == "strafe_left"
    assert bot.calls[-1] == ("strafe_left", 0.8, 2.0)

    assert dispatch_tool(bot, "stop", {})["status"] == "ok"
    assert bot.calls[-1][0] == "stop"

    st = dispatch_tool(bot, "wheels_state", {})
    assert st["status"] == "ok" and st["state"]["last_command"] == "stop"


def test_dispatch_never_raises():
    bot = FakeWheels()
    assert dispatch_tool(bot, "drive", {"direction": "warp"})["status"] == "error"
    assert dispatch_tool(bot, "selfdestruct", {})["status"] == "error"
    bot.reachable = False
    out = dispatch_tool(bot, "drive", {"direction": "forward"})
    assert out["status"] == "error" and "unreachable" in out["error"]


def test_describe_tool_result_lines():
    ok = dispatch_tool(FakeWheels(), "drive", {"direction": "forward"})
    assert describe_tool_result("drive", {"direction": "forward"}, ok) \
        == "forward 1.5s @0.8 → ok"
    bad = {"status": "error", "error": "chassis unreachable"}
    assert "unreachable" in describe_tool_result("stop", {}, bad)
