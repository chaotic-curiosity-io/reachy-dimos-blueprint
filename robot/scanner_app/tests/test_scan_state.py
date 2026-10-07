"""Smoke tests for the scan-state pose tracker.

Pure-Python; no robot or websocket needed. Run with::

    pip install -e '.[dev]'
    pytest
"""

from __future__ import annotations

import numpy as np

from dimos_scanner.core.scan_state import (
    MAX_BODY_YAW_DEG,
    MAX_HEAD_YAW_DEG,
    ScanState,
)


def test_initial_pose_is_identity_rotation() -> None:
    s = ScanState()
    pose = s.head_pose()
    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_array_equal(pose[3, :], [0, 0, 0, 1])


def test_pitch_clamps_to_limit() -> None:
    s = ScanState()
    for _ in range(100):
        s.apply("pitch_up", step_deg=10.0)
    assert s.head_pitch_deg == -25.0  # MAX_HEAD_PITCH_DEG flipped (pitch_up rotates -)


def test_yaw_left_bleeds_into_body_yaw_past_head_limit() -> None:
    # MAX_HEAD_YAW_DEG=35°; with 10° steps, head crosses the limit on step 4
    # (30° → would-be 40°, clamps to 35°, body absorbs the 5° remainder).
    s = ScanState(step_deg=10.0)
    for _ in range(3):
        s.apply("yaw_left")
    assert s.head_yaw_deg == 30.0
    assert s.body_yaw_deg == 0.0
    s.apply("yaw_left")
    assert s.head_yaw_deg == MAX_HEAD_YAW_DEG
    assert s.body_yaw_deg == 5.0
    # Further nudges go fully to body (head is now saturated).
    s.apply("yaw_left")
    assert s.head_yaw_deg == MAX_HEAD_YAW_DEG
    assert s.body_yaw_deg == 15.0


def test_body_yaw_clamps_at_max() -> None:
    s = ScanState()
    for _ in range(200):
        s.apply("yaw_left", step_deg=10.0)
    assert s.body_yaw_deg == MAX_BODY_YAW_DEG


def test_reset_zeros_all_axes() -> None:
    s = ScanState(head_yaw_deg=20, head_pitch_deg=10, body_yaw_deg=30)
    s.apply("reset")
    assert (s.head_yaw_deg, s.head_pitch_deg, s.head_roll_deg, s.body_yaw_deg) == (0, 0, 0, 0)
