"""The system prompt must carry the robot's real numbers, not vibes."""

from __future__ import annotations

import sys

from reachy_wheels_app.voice.gemini_live import SYSTEM_PROMPT, build_system_prompt
from reachy_wheels_app.voice.motion import MotionLimits
from reachy_wheels_app.voice.tools import DriveLimits


def test_importing_driver_module_stays_light():
    assert "google.genai" not in sys.modules
    assert "reachy_mini" not in sys.modules


def test_prompt_carries_envelope_and_rates():
    p = build_system_prompt(DriveLimits(), MotionLimits())
    assert "±40°" in p and "±120°" in p and "±160°" in p
    assert "30 cm per second" in p and "80° per second" in p
    assert "120 cm or 320°" in p          # 4 s cap × calibration
    assert "distance_cm" in p and "no odometry" in p
    # Mount realities from the physical build (top-heavy, countertop, blind
    # near-field) must stay in the prompt.
    assert "TOP-HEAVY" in p and "cliff" in p and "20 cm" in p


def test_prompt_tracks_configured_calibration():
    p = build_system_prompt(
        DriveLimits(cm_per_s=50.0, deg_per_s=120.0, max_duration=3.0),
        MotionLimits(head_yaw=30.0, body_yaw=90.0))
    assert "±30°" in p and "±90°" in p and "±120°" in p
    assert "50 cm per second" in p and "150 cm or 360°" in p


def test_default_prompt_is_prebuilt():
    assert SYSTEM_PROMPT == build_system_prompt(DriveLimits())
