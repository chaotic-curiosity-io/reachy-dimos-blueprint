"""Voice endpoints + VoiceBoard behavior (no board hardware, no Gemini)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_wheels_app import api as api_mod
from reachy_wheels_app.api import WheelsLink, wire_routes
from reachy_wheels_app.voice.board import VoiceBoard


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH", tmp_path / "state.json")
    link = WheelsLink(state={"host": "192.0.2.10", "port": 80, "token": "",
                             "default_speed": 0.8, "voice_enabled": True,
                             "gemini_api_key": "", "gemini_model": "m",
                             "gemini_voice": "Aoede"})
    board = VoiceBoard()
    app = FastAPI()
    wire_routes(app, link, voice_board=board)
    return TestClient(app), board, link


def test_status_without_key(harness):
    client, board, _ = harness
    st = client.get("/api/voice/status").json()
    assert st["voice_enabled"] is True
    assert st["key_present"] is False
    assert st["connected"] is False
    assert st["transcript"] == []


def test_setting_key_flips_key_present_but_never_echoes(harness):
    client, _, link = harness
    out = client.post("/api/voice/config", json={"gemini_api_key": "sk-123"}).json()
    assert out["ok"] is True and out["key_present"] is True
    assert "sk-123" not in str(out)
    assert link.snapshot()["gemini_api_key"] == "sk-123"
    assert client.get("/api/voice/status").json()["key_present"] is True


def test_max_drive_seconds_clamped(harness):
    client, _, link = harness
    client.post("/api/voice/config", json={"max_drive_seconds": 99})
    assert link.snapshot()["max_drive_seconds"] == 10.0


def test_transcript_flows_through_status(harness):
    client, board, _ = harness
    board.set_status(True, "listening")
    board.push("user", "roll forward")
    board.push("tool", "forward 1.5s @0.8 → ok")
    st = client.get("/api/voice/status").json()
    assert st["connected"] is True and st["detail"] == "listening"
    assert [l["who"] for l in st["transcript"]] == ["user", "tool"]


def test_board_glues_streamed_fragments():
    board = VoiceBoard()
    board.push("reachy", "Rolling")
    board.push("reachy", "forward now!")
    board.push("user", "thanks")
    lines = board.snapshot()["transcript"]
    assert [l["text"] for l in lines] == ["Rolling forward now!", "thanks"]


def test_status_without_voice_board(monkeypatch, tmp_path):
    monkeypatch.setattr(api_mod.config_store, "STATE_PATH", tmp_path / "s.json")
    link = WheelsLink(state=dict(api_mod.config_store.DEFAULTS))
    app = FastAPI()
    wire_routes(app, link)  # no voice_board (e.g. tests, headless)
    st = TestClient(app).get("/api/voice/status").json()
    assert st["connected"] is False and "not available" in st["detail"]
