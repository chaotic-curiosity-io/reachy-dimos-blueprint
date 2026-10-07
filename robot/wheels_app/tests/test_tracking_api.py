"""The /api/track/* surface, against a fake FollowManager."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_wheels_app import api as api_mod
from reachy_wheels_app.api import WheelsLink, wire_routes
from reachy_wheels_app.tracking.vocab import COCO_CLASSES


class FakeWheels:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1
        return {"ok": True, "command": "stop"}

    def state(self):
        return {"last_command": "stop", "wheels": {}, "moving": False}


class FakeFollow:
    def __init__(self):
        self.started: list[tuple] = []
        self.stopped: list[str] = []
        self.active = False
        self.jpeg = b""
        self.start_result = None

    def start(self, target, distance_cm=None):
        self.started.append((target, distance_cm))
        if self.start_result is not None:
            return self.start_result
        self.active = True
        return {"status": "ok", "target": target, "labels": ["person"],
                "detector": "onnx", "tracker": "roboflow:SORTTracker"}

    def stop(self, reason="stopped"):
        self.stopped.append(reason)
        was, self.active = self.active, False
        return {"status": "ok", "was_active": was}

    def status(self):
        return {"active": self.active, "phase": "following", "detail": "",
                "target": "person", "events": []}

    def preview(self):
        return self.jpeg


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH", tmp_path / "state.json")
    link = WheelsLink(state={"host": "192.0.2.10", "port": 80, "token": "",
                             "default_speed": 0.8, "track_detector": "onnx",
                             "track_remote_url": "", "track_algorithm": "sort"})
    wheels = FakeWheels()
    link.client = wheels
    follow = FakeFollow()
    app = FastAPI()
    wire_routes(app, link, follow_manager=follow)
    return TestClient(app), follow, wheels, link


def test_start_passes_the_phrase_through(harness):
    client, follow, _, _ = harness
    out = client.post("/api/track/start", json={"target": "the dog"}).json()
    assert out["status"] == "ok"
    assert follow.started == [("the dog", None)]


def test_start_forwards_a_requested_distance(harness):
    client, follow, _, _ = harness
    client.post("/api/track/start", json={"target": "me", "distance_cm": 120})
    assert follow.started == [("me", 120.0)]


def test_a_refused_start_is_a_400_with_the_reason(harness):
    client, follow, _, _ = harness
    follow.start_result = {"status": "error", "error": "no class for unicorn"}
    resp = client.post("/api/track/start", json={"target": "unicorn"})
    assert resp.status_code == 400
    assert "unicorn" in resp.json()["error"]


def test_stop_reports_whether_something_was_running(harness):
    client, follow, _, _ = harness
    client.post("/api/track/start", json={"target": "me"})
    assert client.post("/api/track/stop").json()["was_active"] is True
    assert client.post("/api/track/stop").json()["was_active"] is False


def test_the_big_stop_button_also_cancels_a_follow(harness):
    # Otherwise STOP would last exactly one tick of the follow loop.
    client, follow, wheels, _ = harness
    client.post("/api/track/start", json={"target": "me"})
    out = client.post("/api/stop").json()
    assert out["following_cancelled"] is True
    assert follow.stopped and wheels.stops >= 1
    assert follow.active is False


def test_stop_still_halts_the_chassis_when_nothing_is_following(harness):
    client, _, wheels, _ = harness
    assert client.post("/api/stop").json()["following_cancelled"] is False
    assert wheels.stops == 1


def test_status_merges_settings_with_live_state(harness):
    client, _, _, _ = harness
    out = client.get("/api/track/status").json()
    assert out["available"] is True
    assert out["detector"] == "onnx"
    assert out["phase"] == "following"


def test_preview_returns_jpeg_when_there_is_one(harness):
    client, follow, _, _ = harness
    assert client.get("/api/track/preview").status_code == 503
    follow.jpeg = b"\xff\xd8jpeg"
    resp = client.get("/api/track/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"\xff\xd8jpeg"


def test_vocabulary_is_closed_for_onnx_and_open_for_remote(harness):
    client, _, _, link = harness
    out = client.get("/api/track/vocabulary").json()
    assert out["open_vocabulary"] is False
    assert out["classes"] == list(COCO_CLASSES)

    link.reconfigure(track_detector="remote")
    out = client.get("/api/track/vocabulary").json()
    assert out["open_vocabulary"] is True and out["classes"] == []


def test_config_rejects_an_unknown_backend(harness):
    client, _, _, _ = harness
    resp = client.post("/api/track/config", json={"track_detector": "magic"})
    assert resp.status_code == 400


def test_config_clamps_dangerous_values(harness):
    client, _, _, link = harness
    out = client.post("/api/track/config", json={
        "track_drive_speed": 9.0, "track_min_score": 0.0,
        "track_rate_hz": 500.0, "track_imgsz": 4}).json()
    assert out["track_drive_speed"] == 1.0
    assert out["track_min_score"] == 0.05
    assert out["track_rate_hz"] == 30.0
    assert out["track_imgsz"] == 96
    assert link.snapshot()["track_drive_speed"] == 1.0


def test_empty_config_patch_is_refused(harness):
    client, _, _, _ = harness
    assert client.post("/api/track/config", json={}).status_code == 400


def test_routes_degrade_when_no_follow_manager_is_wired(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH", tmp_path / "s.json")
    link = WheelsLink(state={"host": "192.0.2.10", "port": 80, "token": "",
                             "default_speed": 0.8})
    link.client = FakeWheels()
    app = FastAPI()
    wire_routes(app, link)                      # no follow_manager
    client = TestClient(app)

    assert client.post("/api/track/start", json={"target": "me"}).status_code == 503
    assert client.get("/api/track/status").json()["available"] is False
    # STOP must keep working regardless.
    assert client.post("/api/stop").json()["ok"] is True


# --- mount detection endpoints -------------------------------------------

class FakeMount:
    def __init__(self):
        self.state = {"state": "on_wheels", "rssi_dbm": -39.0,
                      "smoothed_dbm": -39.0, "calibrated": True,
                      "on_dbm": -39.0, "off_dbm": -60.0,
                      "threshold_dbm": -49.5, "confidence": 0.95,
                      "freq_mhz": 2462, "misses": 0}
        self.collected = []
        self.samples = [-39.0, -39.0, -38.0]

    def snapshot(self):
        return dict(self.state)

    def collect(self, samples=8, timeout=45.0):
        self.collected.append(samples)
        return list(self.samples)


@pytest.fixture()
def mounted(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH", tmp_path / "s.json")
    link = WheelsLink(state={"host": "192.0.2.10", "port": 80, "token": "",
                             "default_speed": 0.8, "mount_enabled": True,
                             "mount_ssid": "mecanum-beacon"})
    link.client = FakeWheels()
    mount = FakeMount()
    app = FastAPI()
    wire_routes(app, link, mount_monitor=mount)
    return TestClient(app), mount, link


def test_mount_status_reports_the_live_state(mounted):
    client, _, _ = mounted
    out = client.get("/api/mount/status").json()
    assert out["available"] is True
    assert out["state"] == "on_wheels"
    assert out["threshold_dbm"] == -49.5


def test_calibrating_records_the_median_for_that_state(mounted):
    client, mount, link = mounted
    out = client.post("/api/mount/calibrate", json={"state": "on"}).json()
    assert out["ok"] is True
    assert out["median_dbm"] == -39.0
    assert link.snapshot()["mount_rssi_on"] == -39.0


def test_calibrating_the_off_state_completes_the_pair(mounted):
    client, mount, link = mounted
    client.post("/api/mount/calibrate", json={"state": "on"})
    mount.samples = [-61.0, -60.0, -59.0]
    out = client.post("/api/mount/calibrate", json={"state": "off"}).json()
    assert out["calibrated"] is True
    assert out["threshold_dbm"] == pytest.approx(-49.5)
    assert out["note"] is None


def test_a_pair_that_barely_differs_is_reported_as_not_calibrated(mounted):
    client, mount, _ = mounted
    client.post("/api/mount/calibrate", json={"state": "on"})
    mount.samples = [-40.0]          # lifting it off barely moved the signal
    out = client.post("/api/mount/calibrate", json={"state": "off"}).json()
    assert out["calibrated"] is False
    assert "dB" in out["note"]


def test_calibrating_with_no_beacon_says_so(mounted):
    client, mount, _ = mounted
    mount.samples = []
    resp = client.post("/api/mount/calibrate", json={"state": "on"})
    assert resp.status_code == 502
    assert "chassis" in resp.json()["error"]


def test_an_unknown_calibration_state_is_refused(mounted):
    client, _, _ = mounted
    assert client.post("/api/mount/calibrate",
                       json={"state": "sideways"}).status_code == 400


def test_mount_routes_degrade_when_the_monitor_is_not_running(harness):
    client, _, _, _ = harness          # wired without a mount monitor
    out = client.get("/api/mount/status").json()
    assert out["available"] is False
    assert client.post("/api/mount/calibrate",
                       json={"state": "on"}).status_code == 503
