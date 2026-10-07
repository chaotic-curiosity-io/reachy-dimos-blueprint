"""Wire protocol shared with the station bridge.

Vendored from the station-side protocol module in ``../../station/`` so this
app stays installable on the robot on its own (the robot never needs the
station code). Keep the two copies in sync; the station copy is the source of
truth.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from typing import Literal


Role = Literal["robot", "controller", "bridge"]
ControlAction = Literal[
    "yaw_left", "yaw_right", "pitch_up", "pitch_down",
    "reset", "wake_up", "sleep",
]


def hello(role: Role, name: str = "", config: dict | None = None) -> str:
    msg = {"type": "hello", "role": role, "name": name}
    if config:
        msg["config"] = config
    return json.dumps(msg)


def control(action: ControlAction, step_deg: float = 5.0) -> str:
    return json.dumps({"type": "control", "action": action, "step_deg": step_deg})


def relocalize_request() -> str:
    """Ask the Mac to relocalize the robot against its saved reference map."""
    return json.dumps({"type": "relocalize_request"})


def parse_text(payload: str) -> dict:
    return json.loads(payload)


FRAME_MSG_TYPE = 0x01
HEADER_FMT = "<BBHQIII"  # 24 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 24


@dataclass(slots=True)
class FrameMsg:
    ts_ns: int
    width: int
    height: int
    jpeg: bytes

    @property
    def ts(self) -> float:
        return self.ts_ns / 1e9


def encode_frame(jpeg: bytes, width: int, height: int, ts_ns: int | None = None) -> bytes:
    if ts_ns is None:
        ts_ns = time.time_ns()
    header = struct.pack(
        HEADER_FMT, FRAME_MSG_TYPE, 0, 0, ts_ns, width, height, len(jpeg)
    )
    return header + jpeg


def decode_frame(payload: bytes) -> FrameMsg:
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"binary payload too short: {len(payload)} < {HEADER_SIZE}")
    msg_type, _r1, _r2, ts_ns, width, height, jpeg_len = struct.unpack(
        HEADER_FMT, payload[:HEADER_SIZE]
    )
    if msg_type != FRAME_MSG_TYPE:
        raise ValueError(f"unexpected msg_type=0x{msg_type:02x}")
    jpeg = payload[HEADER_SIZE : HEADER_SIZE + jpeg_len]
    if len(jpeg) != jpeg_len:
        raise ValueError(
            f"truncated frame: header says {jpeg_len} bytes, got {len(jpeg)}"
        )
    return FrameMsg(ts_ns=ts_ns, width=width, height=height, jpeg=jpeg)


DEFAULT_BRIDGE_PORT = 9876
DEFAULT_BRIDGE_HOST = "0.0.0.0"


# --- Binary IMU codec ------------------------------------------------------
# Mirrored from the station-side protocol module. Keep in sync.

IMU_MSG_TYPE = 0x02
IMU_FMT = "<BBHQfffffffffff"
IMU_SIZE = struct.calcsize(IMU_FMT)
assert IMU_SIZE == 56


@dataclass(slots=True)
class ImuMsg:
    ts_ns: int
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    temp_c: float

    @property
    def ts(self) -> float:
        return self.ts_ns / 1e9


def encode_imu(
    accel: tuple[float, float, float],
    gyro: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    temp_c: float,
    ts_ns: int | None = None,
) -> bytes:
    if ts_ns is None:
        ts_ns = time.time_ns()
    ax, ay, az = accel
    gx, gy, gz = gyro
    qw, qx, qy, qz = quat
    return struct.pack(
        IMU_FMT, IMU_MSG_TYPE, 0, 0, ts_ns,
        ax, ay, az, gx, gy, gz, qw, qx, qy, qz, temp_c,
    )


def decode_imu(payload: bytes) -> ImuMsg:
    if len(payload) < IMU_SIZE:
        raise ValueError(f"imu payload too short: {len(payload)} < {IMU_SIZE}")
    fields = struct.unpack(IMU_FMT, payload[:IMU_SIZE])
    msg_type = fields[0]
    if msg_type != IMU_MSG_TYPE:
        raise ValueError(f"unexpected msg_type=0x{msg_type:02x}")
    ts_ns = fields[3]
    ax, ay, az, gx, gy, gz, qw, qx, qy, qz, temp_c = fields[4:]
    return ImuMsg(
        ts_ns=ts_ns,
        accel=(ax, ay, az),
        gyro=(gx, gy, gz),
        quat=(qw, qx, qy, qz),
        temp_c=temp_c,
    )


# --- Binary head-pose codec ------------------------------------------------
# Mirrored from the station-side protocol module. Keep in sync. The robot streams its
# current head pose (forward kinematics, body/FLU convention: reachy_base ->
# head; the Mac applies the head->camera extrinsic) so the Mac can drive the
# spatial pipeline from kinematics instead of monocular VO (which degenerates
# on the in-place rotation a fixed-base scanner does). ts_ns is the FK read
# instant on the robot clock — the Mac interpolates poses to frame timestamps.

POSE_MSG_TYPE = 0x03
POSE_FMT = "<BBHQ16f"
POSE_SIZE = struct.calcsize(POSE_FMT)
assert POSE_SIZE == 76


@dataclass(slots=True)
class PoseMsg:
    ts_ns: int
    matrix: tuple  # 16 floats, row-major 4x4

    @property
    def ts(self) -> float:
        return self.ts_ns / 1e9


def encode_pose(matrix, ts_ns: int | None = None) -> bytes:
    """``matrix`` is any iterable of 16 floats (row-major 4x4). numpy callers
    pass ``m.flatten()`` — this module stays numpy-free."""
    if ts_ns is None:
        ts_ns = time.time_ns()
    vals = [float(x) for x in matrix]
    if len(vals) != 16:
        raise ValueError(f"pose matrix must have 16 elements, got {len(vals)}")
    return struct.pack(POSE_FMT, POSE_MSG_TYPE, 0, 0, ts_ns, *vals)


def decode_pose(payload: bytes) -> PoseMsg:
    if len(payload) < POSE_SIZE:
        raise ValueError(f"pose payload too short: {len(payload)} < {POSE_SIZE}")
    fields = struct.unpack(POSE_FMT, payload[:POSE_SIZE])
    msg_type = fields[0]
    if msg_type != POSE_MSG_TYPE:
        raise ValueError(f"unexpected msg_type=0x{msg_type:02x}")
    return PoseMsg(ts_ns=fields[3], matrix=fields[4:])


def peek_msg_type(payload: bytes) -> int:
    return payload[0] if payload else 0
