"""WebSocket message protocol between the on-robot scanner app and the station bridge.

Wire-compatible with ``robot/scanner_app/src/dimos_scanner/io/protocol.py``
(the robot-side copy); change both together.

Two message kinds travel over the WS:

  * TEXT (JSON)   — control plane: hello, control commands, status
  * BINARY        — data plane: camera frames + IMU samples (binary structs)

Binary frame wire format (little-endian for simplicity, fixed header):

    offset  size  field
    0       1     msg_type            (0x01 = frame)
    1       1     reserved            (0)
    2       2     reserved            (0)
    4       8     ts_unix_ns          uint64
    12      4     width               uint32
    16      4     height              uint32
    20      4     jpeg_len            uint32
    24      N     jpeg bytes          (N == jpeg_len)

Binary IMU wire format (fixed 56 bytes, no trailing payload):

    offset  size  field
    0       1     msg_type            (0x02 = imu)
    1       1     reserved            (0)
    2       2     reserved            (0)
    4       8     ts_unix_ns          uint64
    12      12    accel xyz           3 * float32 (m/s^2)
    24      12    gyro  xyz           3 * float32 (rad/s)
    36      16    quat  wxyz          4 * float32
    52      4     temp_c              float32

Binary readers MUST switch on ``payload[0]`` to pick the right decoder.

Keeping the protocol explicit (no msgpack/protobuf dependency) keeps both ends
trivial to debug with `xxd` or a 20-line test client.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from typing import Literal

# --- Roles -----------------------------------------------------------------

Role = Literal["robot", "controller", "bridge"]

# --- Control actions -------------------------------------------------------

# Direction names match arrow-key semantics from the operator's POV.
ControlAction = Literal[
    "yaw_left",     # pan head/body left
    "yaw_right",    # pan head/body right
    "pitch_up",     # tilt head up
    "pitch_down",   # tilt head down
    "reset",        # return to neutral pose
    "wake_up",
    "sleep",
]

# --- JSON helpers ----------------------------------------------------------


def hello(role: Role, name: str = "", config: dict | None = None) -> str:
    msg = {"type": "hello", "role": role, "name": name}
    if config:
        msg["config"] = config
    return json.dumps(msg)


def control(action: ControlAction, step_deg: float = 5.0) -> str:
    return json.dumps({"type": "control", "action": action, "step_deg": step_deg})


def relocalize_request() -> str:
    """Robot -> station: request a one-shot relocalization against a reference map.

    Kept for wire compatibility with the robot app's web button; this slimmed
    station does not implement relocalization and only logs the request."""
    return json.dumps({"type": "relocalize_request"})


def relocalize_status(status: dict) -> str:
    """Station -> robot: relocalization progress/result for the web app to display."""
    return json.dumps({"type": "relocalize_status", **status})


def parse_text(payload: str) -> dict:
    return json.loads(payload)


# --- Binary frame codec ----------------------------------------------------

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


# --- Binary IMU codec ------------------------------------------------------

IMU_MSG_TYPE = 0x02
# msg_type(1) + reserved(3) + ts_ns(8) + accel(3*f4) + gyro(3*f4) + quat(4*f4) + temp(f4)
IMU_FMT = "<BBHQfffffffffff"
IMU_SIZE = struct.calcsize(IMU_FMT)
assert IMU_SIZE == 56


@dataclass(slots=True)
class ImuMsg:
    ts_ns: int
    accel: tuple[float, float, float]   # m/s^2
    gyro: tuple[float, float, float]    # rad/s
    quat: tuple[float, float, float, float]  # w, x, y, z
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
# The robot streams its current head pose (forward kinematics, body/FLU
# convention: reachy_base -> head; the station composes the head->camera extrinsic)
# so the station can drive the spatial pipeline from kinematics instead of
# monocular VO — VO degenerates on in-place rotation, which is exactly how a
# fixed-base scanner moves. ts_ns is the FK read instant on the robot clock;
# the station interpolates poses to frame capture timestamps.
#
#     offset  size  field
#     0       1     msg_type            (0x03 = pose)
#     1       3     reserved            (0)
#     4       8     ts_unix_ns          uint64
#     12      64    matrix              16 * float32, row-major 4x4 (body->head_optical)

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
    """``matrix`` is any iterable of 16 floats (row-major 4x4). Callers with a
    numpy array should pass ``m.flatten()`` — this module stays numpy-free."""
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
    """Look at byte 0 to dispatch to the right decoder. Returns 0 on empty input."""
    return payload[0] if payload else 0


# --- Defaults --------------------------------------------------------------

DEFAULT_BRIDGE_PORT = 9876
DEFAULT_BRIDGE_HOST = "0.0.0.0"
