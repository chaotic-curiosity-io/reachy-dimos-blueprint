"""HTTP command server for the mecanum chassis.

Exposes the movement primitives over JSON so a planning agent can drive the
chassis without knowing anything about GPIO pins:

    POST /cmd    {"command": "strafe_left", "speed": 0.6, "duration": 1.5}
    POST /cmd    {"command": "move", "vx": 1, "vy": 0.5, "omega": 0, "speed": 0.7}
    POST /cmd    {"command": "wheels", "speeds": {"front_left": 1, "rear_right": -0.6}}
    POST /stop
    GET  /state
    GET  /              -- browser control page

Every command arms a deadman timer. A command with an explicit `duration` stops
at the end of it; a command without one stops after netcfg.COMMAND_TIMEOUT
unless another command arrives first. Either way the chassis cannot keep
driving after the controlling agent goes away.
"""

import asyncio
import json
import os
import sys
import time

import machine

import netcfg
import pins
from page import PAGE
import wifi
from motors import MecanumDrive

# Commands that take a single `speed` argument.
_PRIMITIVES = (
    "forward", "reverse", "strafe_left", "strafe_right",
    "rotate_cw", "rotate_ccw",
    "diagonal_fl", "diagonal_fr", "diagonal_rl", "diagonal_rr",
)

_bot = None
_deadline = None        # ticks_ms value, or None when idle
_last_command = "stop"

MAX_BODY = 65536
_LOG = []               # ring buffer so logs survive without a serial cable
_LOG_MAX = 200


def log(*parts):
    """Print, and keep a copy retrievable over the network via GET /log."""
    line = " ".join(str(p) for p in parts)
    print(line)
    _LOG.append("%d %s" % (time.ticks_ms(), line))
    if len(_LOG) > _LOG_MAX:
        del _LOG[0:len(_LOG) - _LOG_MAX]


def _authorised(token):
    return not netcfg.API_TOKEN or token == netcfg.API_TOKEN


def _safe_name(name):
    """Only flat filenames -- no traversal, no subdirectories."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    return name


def _arm(duration):
    """Set the point in time at which the drive must stop by itself."""
    global _deadline
    seconds = netcfg.COMMAND_TIMEOUT if duration is None else float(duration)
    seconds = max(0.0, min(30.0, seconds))
    _deadline = time.ticks_add(time.ticks_ms(), int(seconds * 1000))


def _disarm():
    global _deadline
    _deadline = None


async def _watchdog():
    """Stop the chassis the moment its deadline passes."""
    while True:
        if _deadline is not None and time.ticks_diff(_deadline, time.ticks_ms()) <= 0:
            _bot.stop()
            _disarm()
        await asyncio.sleep_ms(50)


def _dispatch(req):
    """Apply one command dict. Returns (ok, message)."""
    global _last_command

    command = req.get("command", "")
    speed = float(req.get("speed", pins.DEFAULT_SPEED))
    duration = req.get("duration")

    if command in ("stop", "brake"):
        (_bot.brake if command == "brake" else _bot.stop)()
        _disarm()
        _last_command = command
        return True, command

    if command == "move":
        _bot.move(vx=float(req.get("vx", 0.0)),
                  vy=float(req.get("vy", 0.0)),
                  omega=float(req.get("omega", 0.0)),
                  speed=speed)
        _arm(duration)
        _last_command = "move"
        return True, "move"

    if command == "wheels":
        # An explicit per-wheel mix. The wheel lab uses this to hunt for a
        # rotate-in-place that pivots instead of walking; nothing here is
        # renormalised, so an asymmetric mix survives to the motors.
        mix = req.get("speeds") or {}
        for name in mix:
            if name not in _bot.wheels:
                return False, "unknown wheel %r" % name
        _bot.set_wheels(mix, speed=speed)
        _arm(duration)
        _last_command = "wheels"
        return True, "wheels"

    if command == "wheel":
        # One wheel alone, for identifying corners and checking polarity.
        name = req.get("wheel")
        if name not in _bot.wheels:
            return False, "unknown wheel %r" % name
        _bot.stop()
        _bot.wheels[name].drive(speed)
        _arm(duration)
        _last_command = "wheel:" + name
        return True, _last_command

    if command == "channel":
        # Address an L298N channel by its GPIO pair, bypassing the wheel map.
        # Calibration needs this: the map itself is what is in question.
        pair = (int(req.get("a", -1)), int(req.get("b", -1)))
        for name, motor in _bot.wheels.items():
            if tuple(pins.WHEELS[name]["pins"]) == pair:
                _bot.stop()
                motor.drive(speed)
                _arm(duration)
                _last_command = "channel:%d/%d" % pair
                return True, _last_command
        return False, "no channel on GPIO %d/%d" % pair

    if command in _PRIMITIVES:
        getattr(_bot, command)(speed=speed)
        _arm(duration)
        _last_command = command
        return True, command

    return False, "unknown command %r" % command


def _state():
    remaining = None
    if _deadline is not None:
        remaining = max(0, time.ticks_diff(_deadline, time.ticks_ms())) / 1000
    return {
        "last_command": _last_command,
        "wheels": _bot.state(),
        "moving": any(v != 0 for v in _bot.state().values()),
        "tuning": _bot.tuning(),
        "stops_in": remaining,
    }


_REASON = {200: b"OK", 400: b"Bad Request", 401: b"Unauthorized",
           404: b"Not Found"}


async def _send(writer, status, body, ctype="application/json"):
    if not isinstance(body, (bytes, bytearray)):
        body = body.encode()
    writer.write(b"HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
                 b"Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                 % (status, _REASON.get(status, b"OK"), ctype.encode(), len(body)))
    writer.write(body)
    await writer.drain()


async def _handle(reader, writer):
    try:
        line = await reader.readline()
        parts = line.split()
        if len(parts) < 2:
            return
        method, path = parts[0].decode(), parts[1].decode()

        query = {}
        if "?" in path:
            path, _, qs = path.partition("?")
            for pair in qs.split("&"):
                if "=" in pair:
                    k, _, v = pair.partition("=")
                    query[k] = v

        length = 0
        token = ""
        while True:
            header = await reader.readline()
            if not header or header == b"\r\n":
                break
            low = header.lower()
            if low[:15] == b"content-length:":
                length = int(header.split(b":", 1)[1])
            elif low[:8] == b"x-token:":
                token = header.split(b":", 1)[1].strip().decode()

        body = b""
        while len(body) < length:
            chunk = await reader.read(length - len(body))
            if not chunk:
                break
            body += chunk

        if path == "/" and method == "GET":
            await _send(writer, 200, PAGE, "text/html")

        elif path == "/log":
            await _send(writer, 200, json.dumps({"lines": _LOG}))

        elif path == "/ls":
            files = []
            for f in os.listdir("/"):
                try:
                    files.append({"name": f, "size": os.stat(f)[6]})
                except OSError:
                    pass
            await _send(writer, 200, json.dumps({"files": files}))

        elif path == "/put" and method == "POST":
            name = _safe_name(query.get("path"))
            if not _authorised(token):
                await _send(writer, 401, json.dumps({"ok": False, "error": "bad token"}))
            elif not name:
                await _send(writer, 400, json.dumps({"ok": False, "error": "bad path"}))
            elif len(body) > MAX_BODY:
                await _send(writer, 400, json.dumps({"ok": False, "error": "too large"}))
            else:
                with open(name, "wb") as fh:
                    fh.write(body)
                log("put", name, len(body), "bytes")
                await _send(writer, 200, json.dumps(
                    {"ok": True, "path": name, "size": len(body)}))

        elif path == "/reset" and method == "POST":
            if not _authorised(token):
                await _send(writer, 401, json.dumps({"ok": False, "error": "bad token"}))
            else:
                log("soft reset requested")
                _bot.stop()
                await _send(writer, 200, json.dumps({"ok": True, "resetting": True}))
                asyncio.create_task(_delayed_reset())

        elif path == "/tune" and method == "POST":
            try:
                req = json.loads(body) if body else {}
            except ValueError:
                await _send(writer, 400, json.dumps({"ok": False, "error": "bad json"}))
            else:
                name = req.get("wheel")
                if name not in _bot.wheels:
                    await _send(writer, 400, json.dumps(
                        {"ok": False, "error": "unknown wheel %r" % name}))
                else:
                    motor = _bot.wheels[name]
                    if "flip" in req:
                        motor.invert = not motor.invert
                    if "invert" in req:
                        motor.invert = bool(req["invert"])
                    if "trim" in req:
                        motor.trim = max(0.3, min(1.5, float(req["trim"])))
                    if req.get("unbias"):
                        for m in _bot.wheels.values():
                            m.invert = False
                    await _send(writer, 200, json.dumps({
                        "ok": True, "wheel": name,
                        "invert": motor.invert, "trim": motor.trim}))

        elif path == "/rssi":
            # Ranging telemetry. `rssi` is this board's link to the router, not
            # to the robot -- the robot-facing measurement is made at the robot,
            # by scanning for `beacon_ssid`.
            await _send(writer, 200, json.dumps({
                "rssi": wifi.rssi(),
                "channel": wifi.channel(),
                "beacon_ssid": wifi.beacon_ssid(),
                "t_ms": time.ticks_ms(),
            }))

        elif path == "/state":
            await _send(writer, 200, json.dumps(_state()))

        elif path == "/stop":
            _bot.stop()
            _disarm()
            await _send(writer, 200, json.dumps({"ok": True, "command": "stop"}))

        elif path == "/cmd" and method == "POST":
            try:
                req = json.loads(body) if body else {}
            except ValueError:
                await _send(writer, 400, json.dumps({"ok": False, "error": "bad json"}))
            else:
                ok, message = _dispatch(req)
                payload = {"ok": ok, "state": _state()}
                if ok:
                    payload["command"] = message
                else:
                    payload["error"] = message
                await _send(writer, 200 if ok else 400, json.dumps(payload))
        else:
            await _send(writer, 404, json.dumps({"ok": False, "error": "not found"}))

    except Exception as exc:
        # A malformed request must never take the drive layer down with it.
        log("request failed:", exc)
        _bot.stop()
        _disarm()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _delayed_reset():
    # Let the HTTP response flush before the interpreter restarts.
    await asyncio.sleep(0.4)
    machine.soft_reset()


async def _serve(bot):
    global _bot
    _bot = bot
    ip = wifi.start()
    await asyncio.start_server(_handle, "0.0.0.0", netcfg.PORT)
    log("listening on http://%s:%d  (deadman %.1fs)" % (ip, netcfg.PORT, netcfg.COMMAND_TIMEOUT))
    if not netcfg.API_TOKEN:
        log("warning: API_TOKEN empty, /put and /reset are unauthenticated")
    asyncio.create_task(_watchdog())
    while True:
        await asyncio.sleep(1)


def run(bot=None):
    """Blocking. Ctrl-C at the serial REPL stops it and halts the wheels."""
    bot = bot or MecanumDrive()
    try:
        asyncio.run(_serve(bot))
    finally:
        bot.stop()
        _disarm()
        print("server stopped, wheels halted")
