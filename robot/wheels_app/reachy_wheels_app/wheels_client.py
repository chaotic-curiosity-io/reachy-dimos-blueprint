"""HTTP client for the Wheels mecanum chassis.

The chassis is an ESP32 running MicroPython (separate Wheels project) that
exposes movement primitives over JSON:

    POST /cmd    {"command": "strafe_left", "speed": 0.6, "duration": 1.5}
    POST /cmd    {"command": "move", "vx": 1, "vy": 0.5, "omega": 0}
    POST /cmd    {"command": "wheels", "speeds": {"front_left": 1, ...}}
    POST /tune   {"wheel": "front_left", "trim": 0.9}
    POST /stop
    GET  /state
    GET  /log

Safety model (enforced board-side, mirrored here so callers can rely on it):
every command arms a deadman timer. A command with an explicit ``duration``
stops at the end of it; one without stops after the board's COMMAND_TIMEOUT
(2s) unless another command arrives first. A crashed caller therefore never
leaves the chassis driving.

This module is deliberately stdlib-only so it can be vendored byte-for-byte
into apps that must install standalone on the robot.

The chassis address has no built-in default: set ``WHEELS_HOST`` (the ESP32's
LAN IP, e.g. ``192.0.2.10``) in the environment, or pass ``host=`` explicitly.
A client without a host raises :class:`WheelsError` on its first request.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# The ESP32's LAN IP / hostname. Deliberately no hardcoded default — every
# network is different. Set WHEELS_HOST or pass host= explicitly.
DEFAULT_HOST = os.environ.get("WHEELS_HOST", "")
DEFAULT_PORT = int(os.environ.get("WHEELS_PORT", "80"))

# Named single-speed commands understood by the board (device/server.py).
PRIMITIVES = (
    "forward", "reverse", "strafe_left", "strafe_right",
    "rotate_cw", "rotate_ccw",
    "diagonal_fl", "diagonal_fr", "diagonal_rl", "diagonal_rr",
)

# The four corners, in the order everything downstream displays them.
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


class WheelsError(Exception):
    """The chassis rejected a command or could not be reached."""


@dataclass
class WheelsConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # Board round-trips ~300ms average over wifi (power save disabled), with
    # occasional slow outliers; 3s leaves margin without hanging a UI.
    timeout: float = 3.0
    # The ESP32 serves one connection at a time and refuses requests that
    # arrive while it is busy — that is normal, not a fault. Retry through it.
    retries: int = 3
    retry_delay: float = 0.15
    # X-Token header for /put and /reset; empty when the board runs open.
    token: str = ""
    extra_headers: dict = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class WheelsClient:
    """Thin, synchronous wrapper over the chassis HTTP API.

    >>> bot = WheelsClient()
    >>> bot.strafe_left(speed=0.6, duration=1.5)
    >>> bot.stop()
    """

    def __init__(self, config: WheelsConfig | None = None, **overrides):
        self.config = config or WheelsConfig(**overrides)

    # --- transport ------------------------------------------------------

    def _request(self, path: str, payload: dict | None = None,
                 method: str | None = None) -> dict:
        if not self.config.host:
            raise WheelsError(
                f"{path}: no chassis host configured — set WHEELS_HOST or pass "
                "host= (the ESP32's LAN IP)")
        url = self.config.base_url + path
        data = None
        headers = {"Accept": "application/json", **self.config.extra_headers}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        method = method or ("POST" if payload is not None else "GET")
        if self.config.token:
            headers["X-Token"] = self.config.token

        last_exc: Exception | None = None
        for attempt in range(max(1, self.config.retries)):
            if attempt:
                time.sleep(self.config.retry_delay)
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    body = resp.read()
                return json.loads(body) if body else {}
            except urllib.error.HTTPError as exc:
                # The board answered: 400/401/404 carry a JSON error body and
                # retrying will not change the outcome.
                try:
                    detail = json.loads(exc.read()).get("error", "")
                except Exception:
                    detail = ""
                raise WheelsError(
                    f"{path}: HTTP {exc.code}" + (f" — {detail}" if detail else "")
                ) from exc
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                # Refused/reset while the single-connection server is busy, or
                # the board is genuinely off the network. Retry, then report.
                last_exc = exc
        raise WheelsError(
            f"{path}: chassis at {self.config.base_url} unreachable ({last_exc})"
        ) from last_exc

    # --- queries --------------------------------------------------------

    def ping(self) -> bool:
        """True when the chassis answers /state."""
        try:
            self.state()
            return True
        except WheelsError:
            return False

    def state(self) -> dict:
        """Wheel speeds, last command, seconds until the deadman fires."""
        return self._request("/state")

    def log(self) -> list:
        """The board's in-memory log ring buffer (survives without serial)."""
        return self._request("/log").get("lines", [])

    def files(self) -> list:
        return self._request("/ls").get("files", [])

    # --- commands -------------------------------------------------------

    def command(self, command: str, speed: float | None = None,
                duration: float | None = None, **extra) -> dict:
        """Send one raw command dict; raises WheelsError if rejected."""
        payload: dict = {"command": command, **extra}
        if speed is not None:
            payload["speed"] = float(speed)
        if duration is not None:
            payload["duration"] = float(duration)
        reply = self._request("/cmd", payload)
        if not reply.get("ok"):
            raise WheelsError(f"{command}: {reply.get('error', 'rejected')}")
        return reply

    def move(self, vx: float = 0.0, vy: float = 0.0, omega: float = 0.0,
             speed: float | None = None, duration: float | None = None) -> dict:
        """Body-frame velocity mix: +vx forward, +vy left, +omega CCW."""
        return self.command("move", speed=speed, duration=duration,
                            vx=float(vx), vy=float(vy), omega=float(omega))

    def set_wheels(self, speeds: dict, speed: float | None = None,
                   duration: float | None = None) -> dict:
        """Drive an explicit per-wheel mix; wheels left out of it coast.

        Unlike :meth:`move`, nothing is renormalised — an asymmetric mix
        reaches the motors as asked. This is the hook for tuning a
        rotate-in-place, where each corner's contribution is the thing under
        investigation rather than an implementation detail.
        """
        return self.command("wheels", speed=speed, duration=duration,
                            speeds={str(k): float(v) for k, v in speeds.items()})

    def wheel(self, name: str, speed: float | None = None,
              duration: float | None = None) -> dict:
        """Spin one wheel alone (identifying corners, checking polarity)."""
        return self.command("wheel", speed=speed, duration=duration, wheel=name)

    def tune(self, wheel: str, trim: float | None = None,
             invert: bool | None = None, flip: bool = False) -> dict:
        """Adjust one wheel's output scale / polarity on the live board.

        Runtime only: the board keeps it until the next reset, so anything
        worth keeping belongs in the chassis's own ``pins.py``.
        """
        payload: dict = {"wheel": wheel}
        if trim is not None:
            payload["trim"] = float(trim)
        if invert is not None:
            payload["invert"] = bool(invert)
        if flip:
            payload["flip"] = True
        reply = self._request("/tune", payload)
        if not reply.get("ok"):
            raise WheelsError(f"tune {wheel}: {reply.get('error', 'rejected')}")
        return reply

    def stop(self) -> dict:
        """Coast to a stop immediately (dedicated endpoint, never queued)."""
        return self._request("/stop", method="POST")

    def brake(self) -> dict:
        """Short the motors and stop hard."""
        return self.command("brake")


def _add_primitive(name: str) -> None:
    def primitive(self: WheelsClient, speed: float | None = None,
                  duration: float | None = None) -> dict:
        return self.command(name, speed=speed, duration=duration)

    primitive.__name__ = name
    primitive.__doc__ = f"Named primitive `{name}`; deadman-limited like every command."
    setattr(WheelsClient, name, primitive)


for _name in PRIMITIVES:
    _add_primitive(_name)
