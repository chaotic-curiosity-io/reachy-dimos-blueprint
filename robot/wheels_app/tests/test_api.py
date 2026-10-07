"""App HTTP surface, backed by a fake WheelsClient (no board, no SDK)."""

from __future__ import annotations

import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_wheels_app import api as api_mod
from reachy_wheels_app.api import WheelsLink, wire_routes
from reachy_wheels_app.wheels_client import WheelsError


class FakeWheels:
    def __init__(self):
        self.calls = []
        self.reachable = True

    def _check(self):
        if not self.reachable:
            raise WheelsError("/cmd: chassis at http://x unreachable")

    def command(self, command, speed=None, duration=None, **extra):
        self._check()
        self.calls.append({"command": command, "speed": speed,
                           "duration": duration, **extra})
        return {"ok": True, "command": command, "state": {"moving": True}}

    def stop(self):
        self._check()
        self.calls.append({"command": "stop"})
        return {"ok": True, "command": "stop"}

    def state(self):
        self._check()
        self.state_calls = getattr(self, "state_calls", 0) + 1
        return {"last_command": "stop", "wheels": {"front_left": 0.0},
                "moving": False, "stops_in": None}

    def log(self):
        self._check()
        return ["1 listening"]


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH",
                        tmp_path / "state.json")
    link = WheelsLink(state={"host": "192.0.2.10", "port": 80,
                             "token": "", "default_speed": 0.8})
    fake = FakeWheels()
    link.client = fake
    app = FastAPI()
    wire_routes(app, link)
    return TestClient(app), fake, link


def test_import_does_not_pull_in_reachy_mini():
    assert "reachy_mini" not in sys.modules


def test_status_online_and_offline(harness):
    client, fake, link = harness
    ok = client.get("/api/status").json()
    assert ok["connected"] is True and ok["state"]["last_command"] == "stop"

    fake.reachable = False
    link._state_cache = None  # expire the telemetry cache: outage shows now
    down = client.get("/api/status").json()
    assert down["connected"] is False and "unreachable" in down["error"]


def test_status_board_reads_are_cached(harness):
    # Telemetry polls must coalesce — the real board serves one connection at
    # a time and UI polls were competing with drive commands for it.
    client, fake, _ = harness
    for _ in range(3):
        assert client.get("/api/status").json()["connected"] is True
    assert fake.state_calls == 1


def test_status_errors_are_not_cached(harness):
    client, fake, link = harness
    fake.reachable = False
    assert client.get("/api/status").json()["connected"] is False
    fake.reachable = True
    assert client.get("/api/status").json()["connected"] is True
    # And a stale cache from before an outage doesn't mask recovery forever.
    assert 0 < link.STATE_CACHE_TTL <= 2.0


def test_cmd_forwards_primitives(harness):
    client, fake, _ = harness
    resp = client.post("/api/cmd", json={"command": "strafe_left", "speed": 0.6})
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake.calls[-1]["command"] == "strafe_left"
    assert fake.calls[-1]["speed"] == 0.6


def test_cmd_move_carries_velocities(harness):
    client, fake, _ = harness
    client.post("/api/cmd", json={"command": "move", "vx": 1, "omega": -0.5})
    assert fake.calls[-1] == {"command": "move", "speed": None, "duration": None,
                              "vx": 1.0, "vy": 0.0, "omega": -0.5}


def test_cmd_rejects_unknown_command(harness):
    client, fake, _ = harness
    resp = client.post("/api/cmd", json={"command": "selfdestruct"})
    assert resp.status_code == 400
    assert fake.calls == []


def test_cmd_unreachable_maps_to_502(harness):
    client, fake, _ = harness
    fake.reachable = False
    resp = client.post("/api/cmd", json={"command": "forward"})
    assert resp.status_code == 502 and resp.json()["ok"] is False


def test_stop_endpoint(harness):
    client, fake, _ = harness
    assert client.post("/api/stop").json()["ok"] is True
    assert fake.calls[-1]["command"] == "stop"


def test_config_roundtrip_never_echoes_token(harness):
    client, _, link = harness
    cfg = client.get("/api/config").json()
    assert cfg["host"] == "192.0.2.10" and cfg["token"] is False

    out = client.post("/api/config", json={
        "host": "192.0.2.99", "token": "secret", "default_speed": 5.0}).json()
    assert out["host"] == "192.0.2.99"
    assert out["token"] is True            # presence only, not the value
    assert out["default_speed"] == 1.0     # clamped
    assert link.client.config.host == "192.0.2.99"       # client rebuilt
    assert link.client.config.token == "secret"


def test_config_empty_patch_is_an_error(harness):
    client, _, _ = harness
    assert client.post("/api/config", json={}).status_code == 400


def test_board_log_passthrough(harness):
    client, fake, _ = harness
    assert client.get("/api/log").json()["lines"] == ["1 listening"]
    fake.reachable = False
    down = client.get("/api/log").json()
    assert down["connected"] is False and down["lines"] == []


def test_app_side_settings_do_not_rebuild_the_board_client(harness):
    """Only host/port/token may replace the client.

    Rebuilding on every setting change quietly swapped an injected fake for
    a real client aimed at whatever address was in the state — which made a
    test capable of driving the chassis on the bench. It also rewrote the
    settings file on every repeat of a held command.
    """
    _, fake, link = harness
    link.reconfigure(default_speed=0.5)
    assert link.client is fake

    link.reconfigure(host="192.0.2.99")
    assert link.client is not fake
    assert link.client.config.host == "192.0.2.99"


def test_reconfiguring_with_unchanged_values_does_not_rewrite_state(harness, tmp_path):
    _, _, link = harness
    link.reconfigure(default_speed=0.5)
    written = (tmp_path / "state.json").stat().st_mtime_ns
    link.reconfigure(default_speed=0.5)
    assert (tmp_path / "state.json").stat().st_mtime_ns == written
