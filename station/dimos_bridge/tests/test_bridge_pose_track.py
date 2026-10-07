"""Phase-A pose-hygiene tests for the dimos bridge + server.

Covers the two drift fixes that dominate the chair smear:

  1. Timestamped pose interpolation (``Bridge.pose_at``) — a frame captured
     mid-rotation must be paired with the pose at *its* instant, not whatever
     pose arrived last (which trails by up to a frame period during a pan).
  2. The head->camera lever arm (``server._T_HEAD_CAM``) — a pure head yaw must
     translate the camera on an arc, not leave it pinned at the head pivot.

Deterministic, offline, numpy-only — no robot, no dimos import, no depth model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from station.dimos_bridge.bridge import Bridge, _slerp_3x3  # noqa: E402
from station.dimos_bridge.server import _T_HEAD_CAM, _make_network_video_source  # noqa: E402
from station.dimos_bridge.protocol import encode_pose  # noqa: E402


def _yaw(theta: float, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    m[:3, 3] = t
    return m


# --------------------------------------------------------------------------- #
# pose_at interpolation                                                        #
# --------------------------------------------------------------------------- #

def _feed_pose(bridge: Bridge, ts: float, m: np.ndarray) -> None:
    bridge._ingest_pose(encode_pose(m.flatten(), ts_ns=int(ts * 1e9)))


def test_pose_at_interpolates_rotation_midway():
    b = Bridge()
    _feed_pose(b, 100.0, _yaw(0.0))
    _feed_pose(b, 100.1, _yaw(np.deg2rad(10.0)))
    # Halfway in time between the two samples -> 5 deg (slerp of pure yaw).
    mid = b.pose_at(100.05)
    assert mid is not None
    assert np.allclose(mid[:3, :3], _yaw(np.deg2rad(5.0))[:3, :3], atol=1e-6)


def test_pose_at_interpolates_translation_midway():
    b = Bridge()
    _feed_pose(b, 0.0, _yaw(0.0, t=(0.0, 0.0, 0.0)))
    _feed_pose(b, 1.0, _yaw(0.0, t=(1.0, 2.0, -3.0)))
    q = b.pose_at(0.25)
    assert q is not None
    assert np.allclose(q[:3, 3], [0.25, 0.5, -0.75], atol=1e-9)


def test_pose_at_clamps_out_of_range():
    b = Bridge()
    _feed_pose(b, 10.0, _yaw(np.deg2rad(3.0)))
    _feed_pose(b, 11.0, _yaw(np.deg2rad(9.0)))
    assert np.allclose(b.pose_at(5.0), _yaw(np.deg2rad(3.0)))    # before first
    assert np.allclose(b.pose_at(99.0), _yaw(np.deg2rad(9.0)))   # after last


def test_pose_at_is_the_fix_for_latest_pose_aliasing():
    """The core drift bug: pairing a frame with the LATEST pose misassigns
    rotation during a pan. With a frame captured at t=100.05 while poses stream
    at 100.0 and 100.1, ``latest_pose`` (100.1) is 5 deg off; ``pose_at`` nails
    the mid angle."""
    b = Bridge()
    _feed_pose(b, 100.0, _yaw(np.deg2rad(0.0)))
    _feed_pose(b, 100.1, _yaw(np.deg2rad(10.0)))
    latest = b.latest_pose()
    interp = b.pose_at(100.05)
    # latest is the 10-deg pose; interpolated is 5 deg — a 5-deg correction.
    ang_latest = np.rad2deg(np.arccos(np.clip((np.trace(latest[:3, :3]) - 1) / 2, -1, 1)))
    ang_interp = np.rad2deg(np.arccos(np.clip((np.trace(interp[:3, :3]) - 1) / 2, -1, 1)))
    # (arccos-of-trace is numerically soft near small angles — 1e-4 deg is ample.)
    assert abs(ang_latest - 10.0) < 1e-4
    assert abs(ang_interp - 5.0) < 1e-4


def test_pose_track_resets_on_backwards_timestamp():
    """A robot reconnect can send an older ts; the buffer must stay sorted so
    the bisect in pose_at stays valid."""
    b = Bridge()
    _feed_pose(b, 500.0, _yaw(np.deg2rad(20.0)))
    _feed_pose(b, 501.0, _yaw(np.deg2rad(25.0)))
    _feed_pose(b, 3.0, _yaw(np.deg2rad(0.0)))    # reconnect, clock reset
    _feed_pose(b, 3.05, _yaw(np.deg2rad(2.0)))
    mid = b.pose_at(3.025)
    assert mid is not None
    assert np.allclose(mid[:3, :3], _yaw(np.deg2rad(1.0))[:3, :3], atol=1e-6)


# --------------------------------------------------------------------------- #
# Lever arm (head yaw -> camera translation arc)                              #
# --------------------------------------------------------------------------- #

def test_lever_arm_matches_sdk_constant():
    # Same numbers as the SDK's ReachyMini.T_head_cam.
    assert np.allclose(_T_HEAD_CAM[:3, 3], [0.0437, 0.0, 0.0512], atol=1e-9)
    # Rotation is a proper optical<->body permutation.
    R = _T_HEAD_CAM[:3, :3]
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])   # optical +Z (fwd) -> body +X


def test_head_yaw_swings_camera_on_an_arc():
    """A pure head yaw about the body Z must MOVE the camera optical center
    (the lever arm), not leave it at the head pivot. Rotation-only would give
    zero translation change; the lever arm gives a centimeters-scale arc."""
    c2w_0 = _yaw(0.0) @ _T_HEAD_CAM
    c2w_30 = _yaw(np.deg2rad(30.0)) @ _T_HEAD_CAM
    shift = np.linalg.norm(c2w_30[:3, 3] - c2w_0[:3, 3])
    # A yaw is about body Z, so only the lever arm's XY projection sweeps an
    # arc (the 51.2 mm up component is on the axis). Radius = |[0.0437, 0]|.
    r_xy = np.linalg.norm(_T_HEAD_CAM[:2, 3])
    expect = 2.0 * r_xy * np.sin(np.deg2rad(15.0))
    assert shift > 0.02                       # centimeters, not zero
    assert np.isclose(shift, expect, atol=1e-6)


def test_video_source_pairs_frame_ts_with_interpolated_pose(monkeypatch):
    """End-to-end of the server's pose path: the NetworkVideoSource must query
    ``pose_at(frame_ts)`` and compose the lever arm, not read ``latest_pose``."""
    b = Bridge()
    # Two poses bracketing the frame instant: 0 deg @ t=50.0, 20 deg @ t=50.2.
    _feed_pose(b, 50.0, _yaw(np.deg2rad(0.0)))
    _feed_pose(b, 50.2, _yaw(np.deg2rad(20.0)))

    class _SF:
        def __init__(self, color_bgr, ts, frame_idx, c2w):
            self.c2w = c2w

    NVS = _make_network_video_source(b, _SF, pose_time_offset=0.0)
    # Bypass __init__ (which blocks on a real frame); exercise _c2w_from_pose.
    src = NVS.__new__(NVS)
    src._pose_inv0 = None
    # First call anchors the world to identity at this frame's interpolated pose.
    anchor = src._c2w_from_pose(50.0)      # 0 deg
    assert anchor is not None and np.allclose(anchor, np.eye(4), atol=1e-9)
    # Frame captured at 50.1 -> pose interpolates to 10 deg (relative to anchor).
    c2w_mid = src._c2w_from_pose(50.1)
    ang_mid = np.rad2deg(np.arccos(np.clip((np.trace(c2w_mid[:3, :3]) - 1) / 2, -1, 1)))
    assert abs(ang_mid - 10.0) < 1e-3
    # Frame at the 20-deg end: relative rotation is +20 deg about body yaw.
    c2w_end = src._c2w_from_pose(50.2)
    rel_ang = np.rad2deg(np.arccos(np.clip((np.trace(c2w_end[:3, :3]) - 1) / 2, -1, 1)))
    assert abs(rel_ang - 20.0) < 1e-3
    # The lever arm makes the camera translate between the frames (not pinned).
    assert np.linalg.norm(c2w_end[:3, 3]) > 0.01


def test_slerp_3x3_matches_endpoints():
    R0 = _yaw(np.deg2rad(0.0))[:3, :3]
    R1 = _yaw(np.deg2rad(40.0))[:3, :3]
    assert np.allclose(_slerp_3x3(R0, R1, 0.0), R0, atol=1e-9)
    assert np.allclose(_slerp_3x3(R0, R1, 1.0), R1, atol=1e-9)
    assert np.allclose(_slerp_3x3(R0, R1, 0.5), _yaw(np.deg2rad(20.0))[:3, :3], atol=1e-6)
