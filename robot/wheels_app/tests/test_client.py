"""WheelsClient against a fake ESP32 board (threaded http.server).

Covers: primitive/move payload shapes, deadman-relevant fields passing
through, /stop, error surfacing, and the busy-board retry (the real chassis
serves one connection at a time and refuses the rest).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reachy_wheels_app.wheels_client import (
    PRIMITIVES,
    WheelsClient,
    WheelsConfig,
    WheelsError,
)


class FakeBoard(BaseHTTPRequestHandler):
    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.server.fail_next > 0:
            self.server.fail_next -= 1
            self.connection.close()
            return
        if self.path == "/state":
            self._reply(200, {"last_command": "stop", "wheels": {},
                              "moving": False, "stops_in": None})
        elif self.path == "/log":
            self._reply(200, {"lines": ["1 boot", "2 listening"]})
        else:
            self._reply(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.server.requests.append((self.path, body))
        if self.path == "/stop":
            self._reply(200, {"ok": True, "command": "stop"})
        elif self.path == "/cmd":
            if body.get("command") == "explode":
                self._reply(400, {"ok": False, "error": "unknown command 'explode'"})
            else:
                self._reply(200, {"ok": True, "command": body.get("command"),
                                  "state": {"moving": True}})
        elif self.path == "/tune":
            if body.get("wheel") not in ("front_left", "front_right",
                                         "rear_left", "rear_right"):
                self._reply(400, {"ok": False,
                                  "error": "unknown wheel %r" % body.get("wheel")})
            else:
                self._reply(200, {"ok": True, "wheel": body["wheel"],
                                  "trim": body.get("trim", 1.0),
                                  "invert": body.get("invert", True)})
        else:
            self._reply(404, {"ok": False, "error": "not found"})

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture()
def board():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBoard)
    server.requests = []
    server.fail_next = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture()
def client(board):
    return WheelsClient(WheelsConfig(host="127.0.0.1", port=board.server_address[1],
                                     timeout=2.0, retry_delay=0.01))


def test_every_primitive_posts_its_name(client, board):
    for name in PRIMITIVES:
        getattr(client, name)(speed=0.5)
    sent = [body["command"] for path, body in board.requests if path == "/cmd"]
    assert sent == list(PRIMITIVES)
    assert all(body["speed"] == 0.5 for _, body in board.requests)


def test_duration_and_default_speed_pass_through(client, board):
    client.forward(duration=1.5)
    _, body = board.requests[-1]
    assert body == {"command": "forward", "duration": 1.5}  # no speed → board default


def test_move_sends_velocity_mix(client, board):
    client.move(vx=1.0, vy=-0.5, omega=0.25, speed=0.7)
    _, body = board.requests[-1]
    assert body == {"command": "move", "vx": 1.0, "vy": -0.5,
                    "omega": 0.25, "speed": 0.7}


def test_stop_uses_dedicated_endpoint(client, board):
    assert client.stop()["ok"] is True
    assert board.requests[-1][0] == "/stop"


def test_state_and_log_and_ping(client):
    assert client.state()["last_command"] == "stop"
    assert client.log() == ["1 boot", "2 listening"]
    assert client.ping() is True


def test_board_rejection_raises(client):
    with pytest.raises(WheelsError, match="explode"):
        client.command("explode")


def test_busy_board_is_retried(client, board):
    board.fail_next = 2  # two dropped connections, third attempt lands
    assert client.ping() is True


def test_unreachable_board_raises_after_retries():
    dead = WheelsClient(WheelsConfig(host="127.0.0.1", port=1,  # nothing listens
                                     timeout=0.2, retries=2, retry_delay=0.01))
    with pytest.raises(WheelsError, match="unreachable"):
        dead.state()
    assert dead.ping() is False


def test_missing_host_fails_fast_with_a_hint():
    # No hardcoded chassis address: an unconfigured client never guesses.
    bot = WheelsClient(WheelsConfig(host="", retries=1))
    with pytest.raises(WheelsError, match="WHEELS_HOST"):
        bot.state()
    assert bot.ping() is False


def test_set_wheels_sends_a_per_wheel_mix(client, board):
    client.set_wheels({"front_left": 1, "rear_right": -0.6}, speed=0.7,
                      duration=1.5)
    path, body = board.requests[-1]
    assert path == "/cmd"
    assert body["command"] == "wheels"
    assert body["speeds"] == {"front_left": 1.0, "rear_right": -0.6}
    assert body["speed"] == 0.7 and body["duration"] == 1.5


def test_tune_pushes_trim_and_polarity(client, board):
    reply = client.tune("rear_left", trim=0.85, invert=False)
    path, body = board.requests[-1]
    assert path == "/tune"
    assert body == {"wheel": "rear_left", "trim": 0.85, "invert": False}
    assert reply["trim"] == 0.85


def test_tune_raises_on_an_unknown_wheel(client, board):
    with pytest.raises(WheelsError):
        client.tune("spare", trim=1.0)
