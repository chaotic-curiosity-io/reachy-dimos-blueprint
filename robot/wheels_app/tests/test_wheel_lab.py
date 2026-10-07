"""The wheel lab: per-wheel mixes, presets, trim persistence.

Everything here runs against a fake chassis — the point of the lab is that a
human drives it, so what the tests can defend is the plumbing: mixes reach
the board unrenormalised, trims survive a board reset because the app kept
them, and a typo cannot make it as far as the motors.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_wheels_app import api as api_mod
from reachy_wheels_app import wheel_lab
from reachy_wheels_app.api import WheelsLink, wire_routes
from reachy_wheels_app.wheels_client import WheelsError


class FakeBoard:
    def __init__(self):
        self.calls = []
        self.tuning = {n: {"invert": True, "trim": 1.0}
                       for n in wheel_lab.WHEEL_NAMES}
        self.reachable = True

    def _check(self):
        if not self.reachable:
            raise WheelsError("/cmd: chassis unreachable")

    def command(self, command, speed=None, duration=None, **extra):
        self._check()
        self.calls.append({"command": command, "speed": speed,
                           "duration": duration, **extra})
        return {"ok": True, "command": command, "state": {"moving": True}}

    def set_wheels(self, speeds, speed=None, duration=None):
        return self.command("wheels", speed=speed, duration=duration,
                            speeds=dict(speeds))

    def tune(self, wheel, trim=None, invert=None, flip=False):
        self._check()
        entry = self.tuning[wheel]
        if trim is not None:
            entry["trim"] = trim
        if invert is not None:
            entry["invert"] = invert
        self.calls.append({"command": "tune", "wheel": wheel,
                           "trim": trim, "invert": invert})
        return {"ok": True, "wheel": wheel, **entry}

    def stop(self):
        self._check()
        return {"ok": True, "command": "stop"}

    def state(self):
        self._check()
        return {"last_command": "wheels", "wheels": {}, "moving": False,
                "stops_in": None, "tuning": self.tuning}


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH", tmp_path / "state.json")
    link = WheelsLink(state={"host": "192.0.2.10", "port": 80, "token": "",
                             "default_speed": 0.8})
    board = FakeBoard()
    link.client = board
    app = FastAPI()
    wire_routes(app, link)
    return TestClient(app), board, link


# --- mixes --------------------------------------------------------------

def test_a_mix_reaches_the_board_with_every_corner_named(harness):
    client, board, _ = harness
    resp = client.post("/api/wheels/mix", json={
        "speeds": {"front_left": 1.0, "rear_right": -0.6}, "speed": 0.7})
    assert resp.status_code == 200
    call = board.calls[-1]
    assert call["command"] == "wheels"
    # Corners left out of the request are commanded to rest, not omitted:
    # a wheel the board never hears about keeps whatever it was doing.
    assert call["speeds"] == {"front_left": 1.0, "front_right": 0.0,
                              "rear_left": 0.0, "rear_right": -0.6}
    assert call["speed"] == 0.7


def test_an_asymmetric_mix_is_not_renormalised(harness):
    """The whole reason the lab exists — `move()` would even these out."""
    client, board, _ = harness
    client.post("/api/wheels/mix", json={
        "speeds": {"front_left": 1.0, "front_right": -0.4,
                   "rear_left": 0.6, "rear_right": -1.0}})
    assert board.calls[-1]["speeds"]["front_right"] == -0.4
    assert board.calls[-1]["speeds"]["rear_left"] == 0.6


def test_mix_values_are_clamped(harness):
    client, board, _ = harness
    client.post("/api/wheels/mix", json={"speeds": {"front_left": 4.0,
                                                    "rear_left": -9.0}})
    assert board.calls[-1]["speeds"]["front_left"] == 1.0
    assert board.calls[-1]["speeds"]["rear_left"] == -1.0


def test_a_misspelled_wheel_never_reaches_the_board(harness):
    client, board, _ = harness
    resp = client.post("/api/wheels/mix", json={"speeds": {"front-left": 1.0}})
    assert resp.status_code == 400
    assert "front-left" in resp.json()["error"]
    assert board.calls == []


def test_pulse_duration_is_capped_below_the_boards_own_limit(harness):
    client, board, _ = harness
    client.post("/api/wheels/mix", json={"speeds": {"front_left": 1.0},
                                         "duration": 90.0})
    assert board.calls[-1]["duration"] == 10.0


def test_a_hold_sends_no_duration_so_the_deadman_is_the_limit(harness):
    client, board, _ = harness
    client.post("/api/wheels/mix", json={"speeds": {"front_left": 1.0}})
    assert board.calls[-1]["duration"] is None


def test_the_last_mix_survives_a_reload(harness):
    client, _, link = harness
    client.post("/api/wheels/mix", json={
        "speeds": {"front_left": 0.8, "front_right": -0.8}, "speed": 0.6})
    lab = client.get("/api/wheels/lab").json()
    assert lab["mix"]["front_left"] == 0.8
    assert lab["speed"] == 0.6
    assert link.snapshot()["wheel_lab_mix"]["front_right"] == -0.8


def test_an_unreachable_board_is_a_502_not_a_crash(harness):
    client, board, _ = harness
    board.reachable = False
    resp = client.post("/api/wheels/mix", json={"speeds": {"front_left": 1.0}})
    assert resp.status_code == 502
    assert resp.json()["ok"] is False


# --- trim ---------------------------------------------------------------

def test_trim_is_pushed_to_the_board_and_remembered(harness):
    client, board, link = harness
    resp = client.post("/api/wheels/tune",
                       json={"wheel": "rear_left", "trim": 0.85})
    assert resp.status_code == 200
    assert board.tuning["rear_left"]["trim"] == 0.85
    assert link.snapshot()["wheel_trim"]["rear_left"] == 0.85


def test_trim_is_clamped_to_the_band_the_board_accepts(harness):
    client, board, _ = harness
    client.post("/api/wheels/tune", json={"wheel": "front_left", "trim": 9.0})
    assert board.tuning["front_left"]["trim"] == wheel_lab.TRIM_MAX


def test_remembered_trims_can_be_re_pushed_after_a_board_reset(harness):
    """The board forgets /tune on reset; this is what puts it back."""
    client, board, _ = harness
    client.post("/api/wheels/tune", json={"wheel": "front_left", "trim": 0.9})
    client.post("/api/wheels/tune", json={"wheel": "rear_right", "invert": False})
    board.tuning = {n: {"invert": True, "trim": 1.0}
                    for n in wheel_lab.WHEEL_NAMES}          # reset

    resp = client.post("/api/wheels/tune/apply")
    assert resp.json()["applied"] == ["front_left", "rear_right"]
    assert board.tuning["front_left"]["trim"] == 0.9
    assert board.tuning["rear_right"]["invert"] is False


def test_tuning_a_wheel_that_does_not_exist_is_rejected(harness):
    client, board, _ = harness
    resp = client.post("/api/wheels/tune", json={"wheel": "spare", "trim": 1.0})
    assert resp.status_code == 400
    assert not [c for c in board.calls if c["command"] == "tune"]


def test_tune_with_nothing_to_change_is_rejected(harness):
    client, _, _ = harness
    assert client.post("/api/wheels/tune",
                       json={"wheel": "front_left"}).status_code == 400


# --- the lab's own view -------------------------------------------------

def test_lab_reports_the_boards_live_tuning_over_the_remembered_copy(harness):
    client, board, link = harness
    link.reconfigure(wheel_trim={"front_left": 0.5})
    board.tuning["front_left"]["trim"] = 1.2      # someone tuned it elsewhere
    lab = client.get("/api/wheels/lab").json()
    assert lab["connected"] is True
    assert lab["board_tuning"]["front_left"]["trim"] == 1.2


def test_lab_still_answers_when_the_chassis_is_off(harness):
    client, board, _ = harness
    board.reachable = False
    lab = client.get("/api/wheels/lab").json()
    assert lab["connected"] is False
    assert lab["presets"] and lab["wheels"]


def test_presets_are_all_valid_full_mixes():
    for preset in wheel_lab.PRESETS:
        assert set(preset["mix"]) == set(wheel_lab.WHEEL_NAMES)
        assert all(-1.0 <= v <= 1.0 for v in preset["mix"].values())
        assert preset["note"]


def test_the_all_four_preset_is_what_the_board_calls_rotate_cw():
    """`move(omega=-1)` → fl +1, fr -1, rl +1, rr -1; the baseline to beat."""
    assert wheel_lab.PRESETS_BY_KEY["all_four"]["mix"] == {
        "front_left": 1.0, "front_right": -1.0,
        "rear_left": 1.0, "rear_right": -1.0}


def test_mirroring_a_mix_reverses_the_turn():
    mix = wheel_lab.PRESETS_BY_KEY["diag_fl_rr"]["mix"]
    assert wheel_lab.mirrored(mix) == {"front_left": -1.0, "front_right": 0.0,
                                       "rear_left": 0.0, "rear_right": 1.0}


def test_describe_is_a_line_a_human_can_paste_back():
    line = wheel_lab.describe({"front_left": 1.0, "rear_right": -0.6}, 0.8)
    assert line == "FL +1.00 · FR +0.00 · RL +0.00 · RR -0.60 @ speed 0.80"


def test_pins_snippet_names_every_corner_with_its_trim():
    snippet = wheel_lab.pins_snippet({"front_left": 0.9}, {"rear_left": False})
    assert '"front_left":' in snippet and '"trim": 0.90' in snippet
    assert '"invert": False' in snippet          # rear_left, flipped
    assert snippet.count('"trim"') == 4
