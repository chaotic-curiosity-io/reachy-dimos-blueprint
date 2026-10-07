#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyserial"]
# ///
"""Keyboard teleoperation and per-wheel diagnostics for the mecanum chassis.

Talks to the MicroPython raw REPL on the ESP32 over USB, so it needs no wifi
and no running server -- just the cable.

    ./teleop.sh                      # or: uv run teleop.py
    ./teleop.sh --port /dev/cu.usbserial-XXXX --speed 0.5
    ./teleop.sh --host <board-ip>    # over wifi, no cable

Terminals do not report key-release, so holding a key works by relying on the
keyboard's auto-repeat: each repeat refreshes a deadline, and letting go lets it
lapse. The device arms its own hardware one-shot timer on every command, so the
wheels stop even if this script dies or the cable is pulled.
"""

import argparse
import glob
import json
import os
import select
import subprocess
import sys
import termios
import time
import tty

ROOT = os.path.dirname(os.path.abspath(__file__))
PINS_PATH = os.path.join(ROOT, "device", "pins.py")
DEPLOY_PATH = os.path.join(ROOT, "deploy.sh")
DEPLOY_WIFI_PATH = os.path.join(ROOT, "deploy_wifi.sh")
# USB-serial device names to try when --port is not given: CP210x / CH340
# adapters on macOS, then Linux. Exactly one match is used; zero or several is
# an error asking for --port.
PORT_GLOBS = ("/dev/cu.usbserial-*", "/dev/cu.wchusbserial-*",
              "/dev/cu.SLAB_USBtoUART*", "/dev/ttyUSB*", "/dev/ttyACM*")
IDLE_STOP = 0.6      # seconds of no keypress before we stop; must exceed the
                     # terminal's initial auto-repeat delay or holding stutters
DEVICE_DEADMAN_MS = 900

ORDER = ("front_left", "front_right", "rear_left", "rear_right")

SETUP = """
import motors
import pins
from machine import Timer
b = motors.MecanumDrive()
b.stop()
_pwms = []
for _w in b.wheels.values():
    _pwms.append(_w._a)
    _pwms.append(_w._b)
_t = Timer(0)
def _halt(t):
    # Runs in interrupt context: touch only preallocated objects, no allocation.
    # This zeroes the outputs without updating b.state(), so after a deadman
    # stop the reported speeds read stale. The duty cycles are the truth.
    for p in _pwms:
        p.duty_u16(0)
def _arm():
    _t.init(mode=Timer.ONE_SHOT, period=%d, callback=_halt)
def go(name, spd):
    getattr(b, name)(spd)
    _arm()
def wheel(name, spd):
    b.stop()
    b.wheels[name].drive(spd)
    _arm()
def unbias():
    # Calibration must observe raw motor polarity, so clear every invert flag.
    for _n in b.wheels:
        b.wheels[_n].invert = False
def chan(a, c, spd):
    # Drive one L298N channel addressed by its pin pair, independent of which
    # corner the map currently believes it reaches.
    b.stop()
    for _n, _cfg in pins.WHEELS.items():
        if tuple(_cfg["pins"]) == (a, c):
            b.wheels[_n].drive(spd)
            break
    _arm()
def flip(name):
    b.wheels[name].invert = not b.wheels[name].invert
def trim(name, v):
    b.wheels[name].trim = v
def halt():
    _t.deinit()
    b.stop()
""" % DEVICE_DEADMAN_MS

# Whole-chassis moves.
CHASSIS_KEYS = {
    "w": "forward",     "\x1b[A": "forward",
    "s": "reverse",     "\x1b[B": "reverse",
    "a": "rotate_ccw",  "\x1b[D": "rotate_ccw",
    "d": "rotate_cw",   "\x1b[C": "rotate_cw",
    "z": "strafe_left",
    "c": "strafe_right",
}

# One wheel at a time. Number = forward, shift+number = reverse, laid out so the
# digit matches the corner: 1 2 across the front, 3 4 across the back.
WHEEL_KEYS = {
    "1": ("front_left", 1),  "2": ("front_right", 1),
    "3": ("rear_left", 1),   "4": ("rear_right", 1),
    "!": ("front_left", -1), "@": ("front_right", -1),
    "#": ("rear_left", -1),  "$": ("rear_right", -1),
}

LABEL = {
    "forward": "forward", "reverse": "reverse",
    "rotate_ccw": "rotate left", "rotate_cw": "rotate right",
    "strafe_left": "strafe left", "strafe_right": "strafe right",
}

BANNER = """
  MECANUM TELEOP  --  hold a key to drive, release to stop

  PER-WHEEL   (this is the one for checking the map)
    1 2 3 4           drive ONE wheel forward
    ! @ # $           drive that same wheel backward   (shift + number)
                        1 = front-left    2 = front-right
                        3 = rear-left     4 = rear-right
    f                 FLIP the last wheel you spun (reverses its direction)
    [ / ]             slow down / speed up the last wheel you spun (trim)
    t                 narrated sweep: each wheel in turn, hands-free

  WHOLE CHASSIS
    arrows / w a s d  forward, reverse, rotate left, rotate right
    z / c             strafe left / right

    space  stop      + / -  speed      q  quit & save flips to pins.py
"""


class Board:
    """Minimal MicroPython raw-REPL client."""

    RESEND = 0.15       # ~5ms round trip, so we can afford to be chatty
    HOLD_TTL = None     # the device-side Timer one-shot already covers this

    def __init__(self, port, baud=115200):
        # Imported here, not at module scope: the wifi transport needs no serial
        # library, and should not fail to start because one is missing.
        import serial

        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.write(b"\r\x03\x03")          # interrupt whatever is running
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.write(b"\r\x01")              # ctrl-A: enter raw REPL
        if b">" not in self._read_until(b">"):
            raise RuntimeError("no raw REPL prompt; is the board connected?")

    def _read_until(self, token, timeout=6.0):
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            waiting = self.ser.in_waiting
            if waiting:
                buf += self.ser.read(waiting)
                if token in buf:
                    return buf
            else:
                time.sleep(0.004)
        return buf

    def run(self, code, timeout=6.0):
        self.ser.write(code.encode() + b"\x04")
        out = self._read_until(b"\x04>", timeout)
        if b"Traceback" in out or b"Error" in out:
            sys.stderr.write("\r\ndevice error: %s\r\n" % out.decode("utf8", "replace"))
        return out

    def close(self):
        try:
            self.run("halt()", timeout=2.0)
            self.ser.write(b"\r\x02")           # back to the friendly REPL
        except Exception:
            pass
        finally:
            self.ser.close()

    # --- verbs shared with HttpBoard -----------------------------------
    def setup(self):
        self.run(SETUP, timeout=8.0)

    def go(self, cmd, speed):
        self.run('go("%s",%.2f)' % (cmd, speed))

    def wheel(self, name, speed):
        self.run('wheel("%s",%.2f)' % (name, speed))

    def channel(self, a, c, speed):
        self.run('chan(%d,%d,%.2f)' % (a, c, speed))

    def flip(self, name):
        self.run('flip("%s")' % name)

    def set_trim(self, name, value):
        self.run('trim("%s",%.2f)' % (name, value))

    def unbias(self):
        self.run("unbias()")

    def halt(self):
        self.run("halt()")


class HttpBoard:
    """Same verbs as Board, spoken over the wifi HTTP API.

    Needs no REPL, so it does not interrupt the running server the way a serial
    connection does -- this is the transport to use once the cable is off.
    """

    RESEND = 0.30       # measured ~213ms median round trip; do not outrun it
    HOLD_TTL = 0.7      # every command self-expires, so a lost stop cannot run on

    def __init__(self, host, token=""):
        self.host = host
        self.token = token
        self.get("/state")          # fail fast if it is not answering

    def _call(self, path, payload=None):
        import urllib.request
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request("http://%s%s" % (self.host, path), data=data,
                                     method="POST" if data is not None else "GET")
        if self.token:
            req.add_header("X-Token", self.token)
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())

    def get(self, path):
        return self._call(path)

    def setup(self):
        pass                        # server.py is already running on the board

    def go(self, cmd, speed):
        self._call("/cmd", {"command": cmd, "speed": speed, "duration": self.HOLD_TTL})

    def wheel(self, name, speed):
        self._call("/cmd", {"command": "wheel", "wheel": name,
                            "speed": speed, "duration": self.HOLD_TTL})

    def channel(self, a, c, speed):
        self._call("/cmd", {"command": "channel", "a": a, "b": c, "speed": speed})

    def flip(self, name):
        self._call("/tune", {"wheel": name, "flip": True})

    def set_trim(self, name, value):
        self._call("/tune", {"wheel": name, "trim": value})

    def unbias(self):
        self._call("/tune", {"wheel": "front_left", "unbias": True})

    def halt(self):
        self._call("/stop")

    def close(self):
        try:
            self.halt()
        except Exception:
            pass


def read_key(timeout=0.1):
    """The most recent keypress, discarding any backlog behind it.

    Auto-repeat generates keys faster than a slow link can send commands, so
    they queue in the tty buffer. Processing that queue one key per iteration
    means a released key keeps being acted on for as long as the backlog lasts
    -- the chassis carries on driving after you let go. Collapsing the buffer to
    its newest key makes release immediate regardless of link latency.
    """
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    key = None
    while True:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            if select.select([sys.stdin], [], [], 0.02)[0]:
                ch += sys.stdin.read(1)
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    ch += sys.stdin.read(1)
        key = ch
        if not select.select([sys.stdin], [], [], 0)[0]:
            return key


def line(text):
    sys.stdout.write("\r\033[K  " + text)
    sys.stdout.flush()


def sweep(board, speed):
    """Drive each wheel in turn, announcing it live so you can watch along."""
    sys.stdout.write("\r\n")
    for name in ORDER:
        for sign, word in ((1, "FORWARD "), (-1, "BACKWARD")):
            end = time.time() + 2.0
            while time.time() < end:
                board.wheel(name, sign * speed)
                line("SWEEP  %-12s %s   (%.0fs)  -- any key aborts"
                     % (name, word, end - time.time()))
                if read_key(0.2) is not None:
                    board.halt()
                    line("sweep aborted")
                    sys.stdout.write("\r\n")
                    return
            board.halt()
            line("SWEEP  %-12s %s   done" % (name, word))
            time.sleep(0.9)
        sys.stdout.write("\r\n")
    line("sweep complete")
    sys.stdout.write("\r\n")


CORNERS = ("front_left", "front_right", "rear_left", "rear_right")
CORNER_PROMPT = """
    1  front-left      2  front-right
    3  rear-left       4  rear-right
"""


def ask(prompt, valid):
    while True:
        got = input(prompt).strip().lower()
        if got in valid:
            return got
        print("    please answer one of: %s" % ", ".join(valid))


def run_calibration(board, speed):
    channels = channel_list()
    print("""
  CALIBRATION

  1. Prop the chassis up so all four wheels spin freely.
  2. Point the robot AWAY from you. That direction is the FRONT.
     Stand behind it, so the robot's left and right match your own.

  For each channel I will spin one wheel. Tell me which corner moved, and
  whether the TOP of that wheel travelled toward the front or the back.
  (Watching the top edge is what makes the answer independent of which side
  of the robot you are standing on.)
""")
    input("  press enter when the chassis is propped and pointing away... ")

    board.unbias()
    result = {}
    for pins_pair, tag in channels:
        print("\n  ---- %s : GPIO %d / %d ----" % (tag, pins_pair[0], pins_pair[1]))
        while True:
            board.halt()
            time.sleep(0.2)
            end = time.time() + 2.5
            while time.time() < end:
                board.channel(pins_pair[0], pins_pair[1], speed)
                time.sleep(0.2)
            board.halt()
            again = ask("  spin again? [y/n] ", ("y", "n"))
            if again == "n":
                break

        print(CORNER_PROMPT)
        corner = CORNERS[int(ask("  which corner moved? [1-4] ", ("1", "2", "3", "4"))) - 1]
        top = ask("  did the TOP of the wheel go toward the front or back? [f/b] ", ("f", "b"))
        if corner in result:
            print("  !! %s was already claimed by another channel; restarting that one."
                  % corner)
            continue
        result[corner] = {"pins": pins_pair, "invert": top == "b", "trim": 1.0}

    return result


def channel_list():
    """The four L298N channels, read out of pins.py as the source of truth."""
    ns = {}
    exec(open(PINS_PATH).read(), ns)
    return [
        (tuple(ns["DRIVER1_A"]), "driver 1, IN1/IN2"),
        (tuple(ns["DRIVER1_B"]), "driver 1, IN3/IN4"),
        (tuple(ns["DRIVER2_A"]), "driver 2, IN1/IN2"),
        (tuple(ns["DRIVER2_B"]), "driver 2, IN3/IN4"),
    ]


def write_map(result):
    """Rewrite the generated block in pins.py with the calibrated mapping."""
    ns = {}
    exec(open(PINS_PATH).read(), ns)
    names = {tuple(ns[k]): k for k in ("DRIVER1_A", "DRIVER1_B", "DRIVER2_A", "DRIVER2_B")}

    lines = ["# BEGIN WHEEL MAP -- regenerated by `./teleop.sh --calibrate`, "
             "edits here are lost", "WHEELS = {"]
    for corner in ("front_left", "rear_left", "front_right", "rear_right"):
        entry = result[corner]
        lines.append('    "%s":%s{"pins": %s, "invert": %s, "trim": %s},'
                     % (corner, " " * (12 - len(corner)),
                        names[tuple(entry["pins"])], entry["invert"],
                        round(float(entry.get("trim", 1.0)), 2)))
    lines += ["}", "# END WHEEL MAP"]

    src = open(PINS_PATH).read()
    start = src.index("# BEGIN WHEEL MAP")
    end = src.index("# END WHEEL MAP") + len("# END WHEEL MAP")
    open(PINS_PATH, "w").write(src[:start] + "\n".join(lines) + src[end:])


def flip_front():
    """Move the front to the opposite end of the chassis.

    This is a 180-degree reframe, not a simple polarity flip. Turning the robot
    around swaps corners diagonally AND reverses each wheel's forward direction;
    doing only the latter would reverse rotation too, which a real 180 does not.
    Diagonal pairs map onto diagonal pairs, so the mecanum roller X survives.
    """
    ns = {}
    exec(open(PINS_PATH).read(), ns)
    old = ns["WHEELS"]
    swap = {"front_left": "rear_right", "front_right": "rear_left",
            "rear_left": "front_right", "rear_right": "front_left"}
    return {new: {"pins": tuple(old[src_]["pins"]),
                  "invert": not old[src_]["invert"],
                  "trim": old[src_].get("trim", 1.0)}
            for new, src_ in swap.items()}


def offer_deploy(args):
    """A saved map only takes effect once it is copied to the board.

    Splitting these was a foot-gun: the f key changes the device in RAM, so the
    session behaves correctly right up until you reboot and the old file loads.

    Deploys over whichever link this session used -- pushing over USB after a
    wifi session would fail on a cable that is not plugged in.
    """
    if args.host:
        script, env, human = DEPLOY_WIFI_PATH, {"HOST": args.host}, "./deploy_wifi.sh"
        if args.token:
            env["TOKEN"] = args.token
        where = "http://%s" % args.host
    else:
        port = args.port or detect_port()
        script, env, human = DEPLOY_PATH, {"PORT": port}, "./deploy.sh"
        where = port

    try:
        answer = input("\n  copy this to the board over %s now? [Y/n] " % where).strip().lower()
    except EOFError:
        answer = "n"
    if answer in ("", "y", "yes"):
        rc = subprocess.call(["bash", script], env=dict(os.environ, **env))
        print("  deployed\n" if rc == 0
              else "  deploy failed (rc=%d); run %s\n" % (rc, human))
    else:
        print("  skipped -- run %s before rebooting the board\n" % human)


def detect_port():
    """The single USB-serial device present, or exit asking for --port."""
    found = sorted({p for g in PORT_GLOBS for p in glob.glob(g)})
    if len(found) == 1:
        return found[0]
    if not found:
        sys.exit("no USB-serial port found; plug in the ESP32 or pass "
                 "--port /dev/cu.usbserial-XXXX")
    sys.exit("several serial ports found, pass one with --port: %s"
             % ", ".join(found))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="serial device, e.g. /dev/cu.usbserial-XXXX or "
                         "/dev/ttyUSB0 (auto-detected when there is exactly "
                         "one; ignored when --host is given)")
    ap.add_argument("--host", default=None,
                    help="drive over wifi instead of USB, e.g. 192.0.2.10")
    ap.add_argument("--token", default="", help="X-Token for --host, if set")
    ap.add_argument("--speed", type=float, default=None,
                    help="override the default speed from pins.py")
    ap.add_argument("--calibrate", action="store_true",
                    help="walk through each channel and rewrite the wheel map")
    ap.add_argument("--flip-front", action="store_true",
                    help="make the opposite end of the chassis the front")
    args = ap.parse_args()

    if args.flip_front:
        # Pure file surgery; no board involved.
        before = {}
        exec(open(PINS_PATH).read(), before)
        result = flip_front()
        write_map(result)
        print("\n  front moved to the opposite end of the chassis\n")
        for corner in ("front_left", "front_right", "rear_left", "rear_right"):
            e, o = result[corner], before["WHEELS"][corner]
            print("    %-12s GPIO %2d/%2d invert=%-5s   (was GPIO %2d/%2d invert=%s)"
                  % (corner, e["pins"][0], e["pins"][1], e["invert"],
                     o["pins"][0], o["pins"][1], o["invert"]))
        offer_deploy(args)
        return

    if args.host:
        print("connecting to http://%s ..." % args.host)
        board = HttpBoard(args.host, args.token)
    else:
        args.port = args.port or detect_port()
        print("connecting to %s ..." % args.port)
        board = Board(args.port)
    board.setup()

    if args.calibrate:
        try:
            result = run_calibration(board, max(0.3, min(1.0, args.speed or 0.6)))
            write_map(result)
        finally:
            board.halt()
            board.close()
        print("\n  wrote %s\n" % PINS_PATH)
        for corner in ("front_left", "front_right", "rear_left", "rear_right"):
            e = result[corner]
            print("    %-12s GPIO %2d/%2d   invert=%s"
                  % (corner, e["pins"][0], e["pins"][1], e["invert"]))
        offer_deploy(args)
        return

    print(BANNER)

    cfg = {}
    exec(open(PINS_PATH).read(), cfg)
    wheels_cfg = cfg["WHEELS"]
    inverts = {n: bool(wheels_cfg[n]["invert"]) for n in CORNERS}
    trims = {n: float(wheels_cfg[n].get("trim", 1.0)) for n in CORNERS}
    original = (dict(inverts), dict(trims))

    speed = max(0.2, min(1.0, args.speed if args.speed is not None
                         else cfg.get("DEFAULT_SPEED", 0.8)))
    last_wheel = None       # most recent wheel driven with 1-4, for the f key
    current = None          # (kind, payload) so repeats are detected correctly
    label = "idle"
    last_press = 0.0
    last_sent = 0.0

    def status():
        line("speed %.2f  |  %s" % (speed, label))

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        status()
        while True:
            key = read_key()
            now = time.time()

            if key is not None:
                if key in ("q", "Q"):
                    break
                if key == " ":
                    board.halt()
                    current, label = None, "idle"
                    status()
                    continue
                if key in ("+", "="):
                    speed = min(1.0, round(speed + 0.05, 2))
                    status()
                    continue
                if key in ("-", "_"):
                    speed = max(0.2, round(speed - 0.05, 2))
                    status()
                    continue
                if key in ("[", "]"):
                    if not last_wheel:
                        label = "press 1-4 first, then [ or ]"
                        status()
                        continue
                    step = 0.05 if key == "]" else -0.05
                    trims[last_wheel] = max(0.3, min(1.5,
                        round(trims[last_wheel] + step, 2)))
                    board.set_trim(last_wheel, trims[last_wheel])
                    label = "%s trim %.2f" % (last_wheel, trims[last_wheel])
                    status()
                    continue
                if key in ("f", "F"):
                    if not last_wheel:
                        label = "press 1-4 first, then f"
                        status()
                        continue
                    board.flip(last_wheel)
                    inverts[last_wheel] = not inverts[last_wheel]
                    # Re-spin it straight away so the new direction is visible.
                    end = time.time() + 1.2
                    while time.time() < end:
                        board.wheel(last_wheel, speed)
                        label = "FLIPPED %s -> invert=%s" % (last_wheel, inverts[last_wheel])
                        status()
                        time.sleep(0.2)
                    board.halt()
                    current = None
                    continue
                if key in ("t", "T"):
                    sweep(board, speed)
                    current, label = None, "idle"
                    status()
                    continue

                # (what to send, an identity for repeat-detection, display label)
                call = None
                if key in WHEEL_KEYS:
                    name, sign = WHEEL_KEYS[key]
                    last_wheel = name
                    call = (lambda n=name, s=sign: board.wheel(n, s * speed),
                            "wheel:%s:%+.2f" % (name, sign * speed),
                            "%-12s %s" % (name, "forward" if sign > 0 else "backward"))
                else:
                    cmd = CHASSIS_KEYS.get(key.lower() if len(key) == 1 else key)
                    if cmd:
                        call = (lambda c=cmd: board.go(c, speed),
                                "go:%s:%.2f" % (cmd, speed), LABEL[cmd])

                if call:
                    send, ident, label = call
                    if ident != current or now - last_sent > board.RESEND:
                        send()
                        last_sent = now
                    current = ident
                    last_press = now
                    status()

            elif current and now - last_press > IDLE_STOP:
                board.halt()
                current, label = None, "idle"
                status()

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        board.close()
        print("\nstopped, wheels halted.")
        if (inverts, trims) != original:
            write_map({n: {"pins": tuple(wheels_cfg[n]["pins"]),
                           "invert": inverts[n], "trim": trims[n]}
                       for n in CORNERS})
            print("\n  saved to %s:" % PINS_PATH)
            for n in CORNERS:
                mark = "  <- changed" if (inverts[n] != original[0][n]
                                          or trims[n] != original[1][n]) else ""
                print("    %-12s invert=%-5s trim=%.2f%s"
                      % (n, inverts[n], trims[n], mark))
            offer_deploy(args)


if __name__ == "__main__":
    main()
