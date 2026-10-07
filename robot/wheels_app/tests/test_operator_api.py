"""External-operator HTTP surface: /api/motion/* and /api/camera.

These endpoints let a station-side agent or script use the same motion
tiers and camera the voice agent has. Offline: fake motion + frame bytes.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_wheels_app import api as api_mod
from reachy_wheels_app.api import WheelsLink, wire_routes
from reachy_wheels_app.voice.motion import RobotMotion


def _fake_head_pose(yaw=0.0, pitch=0.0, degrees=True):
    return {"yaw": yaw, "pitch": pitch}


class FakeMini:
    def goto_target(self, **kw):
        pass


@pytest.fixture()
def operator(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH",
                        tmp_path / "state.json")
    link = WheelsLink(state={"host": "h", "port": 80, "token": "",
                             "default_speed": 0.8})
    motion = RobotMotion(FakeMini(), head_pose_fn=_fake_head_pose)
    frames = {"jpeg": b"\xff\xd8fakejpeg"}
    app = FastAPI()
    wire_routes(app, link, motion=motion,
                frame_provider=lambda: frames["jpeg"])
    return TestClient(app), motion, frames


def test_posture_reports_angles_and_limits(operator):
    client, motion, _ = operator
    motion.look(yaw=10, pitch=5)
    out = client.get("/api/motion").json()
    assert out["head_yaw"] == 10.0 and out["head_pitch"] == 5.0
    assert out["limits"]["body_yaw"] == motion.limits.body_yaw


def test_look_turn_center_roundtrip(operator):
    client, motion, _ = operator
    out = client.post("/api/motion/look", json={"yaw": 30, "pitch": -10}).json()
    assert out["status"] == "ok" and out["head_yaw"] == 30.0

    out = client.post("/api/motion/turn", json={"degrees": -45}).json()
    assert out["status"] == "ok" and out["body_yaw"] == -45.0
    assert out["can_turn_left"] == 165.0

    out = client.post("/api/motion/center").json()
    assert (out["head_yaw"], out["body_yaw"]) == (0, 0)


def test_camera_serves_jpeg(operator):
    client, _, frames = operator
    resp = client.get("/api/camera")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content.startswith(b"\xff\xd8")

    frames["jpeg"] = None
    assert client.get("/api/camera").status_code == 503


def test_endpoints_503_without_motion_or_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH",
                        tmp_path / "state.json")
    app = FastAPI()
    wire_routes(app, WheelsLink(state={"host": "h", "port": 80, "token": "",
                                       "default_speed": 0.8}))
    client = TestClient(app)
    assert client.get("/api/motion").status_code == 503
    assert client.post("/api/motion/look", json={"yaw": 1}).status_code == 503
    assert client.get("/api/camera").status_code == 503


def test_sdk_fault_becomes_500_json(operator, monkeypatch):
    client, motion, _ = operator

    def explode(**kw):
        raise RuntimeError("servo fault")

    monkeypatch.setattr(motion._mini, "goto_target", explode)
    resp = client.post("/api/motion/look", json={"yaw": 5})
    assert resp.status_code == 500 and "servo fault" in resp.json()["error"]
