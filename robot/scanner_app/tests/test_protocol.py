"""Round-trip the wire codec without touching the network."""

from __future__ import annotations

from dimos_scanner.io.protocol import (
    HEADER_SIZE,
    control,
    decode_frame,
    encode_frame,
    hello,
    parse_text,
)


def test_frame_roundtrip_preserves_metadata() -> None:
    jpeg = b"\xff\xd8\xff\xe0" + b"x" * 1000
    payload = encode_frame(jpeg, width=320, height=240, ts_ns=42_000_000_000)
    msg = decode_frame(payload)
    assert msg.jpeg == jpeg
    assert msg.width == 320 and msg.height == 240
    assert msg.ts_ns == 42_000_000_000
    assert len(payload) == HEADER_SIZE + len(jpeg)


def test_hello_and_control_serialise_to_known_shape() -> None:
    assert parse_text(hello("robot", name="r1")) == {
        "type": "hello", "role": "robot", "name": "r1",
    }
    assert parse_text(control("yaw_left", step_deg=7.5)) == {
        "type": "control", "action": "yaw_left", "step_deg": 7.5,
    }
