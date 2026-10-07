"""Wire-format checks for the station protocol (offline, stdlib + numpy).

The station decodes what ``robot/scanner_app`` encodes. When the robot-side
copy of the protocol is present in this checkout, byte-level round trips are
checked in both directions so the two copies cannot silently drift.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from station.dimos_bridge import protocol as station  # noqa: E402

_ROBOT_PROTOCOL = _REPO / "robot" / "scanner_app" / "src" / "dimos_scanner" / "io" / "protocol.py"


def _robot():
    if not _ROBOT_PROTOCOL.exists():
        pytest.skip("robot/scanner_app protocol not present in this checkout")
    spec = importlib.util.spec_from_file_location("robot_protocol", _ROBOT_PROTOCOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["robot_protocol"] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def test_fixed_sizes():
    assert station.HEADER_SIZE == 24
    assert station.IMU_SIZE == 56
    assert station.POSE_SIZE == 76


def test_frame_roundtrip():
    payload = station.encode_frame(b"\xff\xd8jpeg", 640, 480, ts_ns=123)
    msg = station.decode_frame(payload)
    assert (msg.ts_ns, msg.width, msg.height, msg.jpeg) == (123, 640, 480, b"\xff\xd8jpeg")
    assert station.peek_msg_type(payload) == station.FRAME_MSG_TYPE


def test_pose_roundtrip_row_major():
    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    msg = station.decode_pose(station.encode_pose(m.flatten(), ts_ns=7))
    assert msg.ts_ns == 7
    assert np.allclose(np.asarray(msg.matrix).reshape(4, 4), m)


def test_xr_messages_removed():
    for name in ("set_mode", "look_world", "robot_relocalize", "app_status", "xr_status"):
        assert not hasattr(station, name)


def test_robot_encodes_station_decodes():
    robot = _robot()
    assert robot.DEFAULT_BRIDGE_PORT == station.DEFAULT_BRIDGE_PORT
    for a, b in (("FRAME_MSG_TYPE",) * 2, ("IMU_MSG_TYPE",) * 2, ("POSE_MSG_TYPE",) * 2,
                 ("HEADER_FMT",) * 2, ("IMU_FMT",) * 2, ("POSE_FMT",) * 2):
        assert getattr(robot, a) == getattr(station, b)
    f = station.decode_frame(robot.encode_frame(b"abc", 2, 3, ts_ns=5))
    assert (f.width, f.height, f.jpeg, f.ts_ns) == (2, 3, b"abc", 5)
    i = station.decode_imu(robot.encode_imu((1, 2, 3), (4, 5, 6), (1, 0, 0, 0), 30.0, ts_ns=9))
    assert i.accel == (1, 2, 3) and i.quat == (1, 0, 0, 0)
    p = station.decode_pose(robot.encode_pose([float(v) for v in range(16)], ts_ns=11))
    assert list(p.matrix) == [float(v) for v in range(16)]
    assert json.loads(robot.relocalize_request())["type"] == "relocalize_request"
    assert json.loads(robot.hello("robot", "x"))["role"] == "robot"
