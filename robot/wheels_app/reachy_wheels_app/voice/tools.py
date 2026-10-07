"""Gemini Live tool declarations for driving, and their dispatcher.

Pure module — no google.genai, no SDK — so the safety clamps and the
name/args → WheelsClient mapping are offline-testable. The Gemini provider
calls ``dispatch_tool`` (in an executor: WheelsClient is blocking HTTP) and
returns the resulting dict as the FunctionResponse payload, so the model
hears back whether the chassis actually moved.

Safety: a voice command is never allowed to run open-ended. Every drive gets
an explicit duration, clamped to ``DriveLimits.max_duration`` — on top of the
chassis's own 2 s deadman and 30 s board-side cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..wheels_client import PRIMITIVES, WheelsError

_DIRECTION_WORDS = {
    "forward": "gliding forward",
    "reverse": "backing up",
    "strafe_left": "sliding left",
    "strafe_right": "sliding right",
    "rotate_cw": "turning right",
    "rotate_ccw": "turning left",
    "diagonal_fl": "drifting front-left",
    "diagonal_fr": "drifting front-right",
    "diagonal_rl": "drifting back-left",
    "diagonal_rr": "drifting back-right",
}


# Each translating primitive as an angle in the ROBOT's frame, degrees CCW
# from "the way the robot faces". Rotations are mount-independent — spinning
# about the vertical axis turns the robot the same way however it is bolted
# on — so they are deliberately absent.
_PRIMITIVE_HEADINGS: dict[str, float] = {
    "forward": 0.0, "diagonal_fl": 45.0, "strafe_left": 90.0,
    "diagonal_rl": 135.0, "reverse": 180.0, "diagonal_rr": 225.0,
    "strafe_right": 270.0, "diagonal_fr": 315.0,
}
_HEADING_PRIMITIVES = {round(v) % 360: k for k, v in _PRIMITIVE_HEADINGS.items()}


def remap_primitive(direction: str, mount_yaw_deg: float) -> str:
    """Translate a robot-frame primitive into the chassis-frame one.

    The robot can be bolted onto the chassis facing any direction. The
    model — and the user — mean "forward" relative to the ROBOT; the board
    only knows its own axes. With the robot mounted facing the chassis's
    right (mount_yaw −90), "forward" has to leave as `strafe_right`.

    Rotations pass through untouched. Offsets are snapped to the nearest
    45°, which is all the eight named primitives can express.
    """
    heading = _PRIMITIVE_HEADINGS.get(direction)
    if heading is None:          # rotate_cw / rotate_ccw / anything else
        return direction
    snapped = round((heading + mount_yaw_deg) / 45.0) * 45
    return _HEADING_PRIMITIVES.get(snapped % 360, direction)


@dataclass(frozen=True)
class DriveLimits:
    max_duration: float = 4.0     # seconds per voice command
    default_duration: float = 1.5
    min_speed: float = 0.2
    max_speed: float = 1.0
    default_speed: float = 0.8    # board default; rotation needs the torque
    # Open-loop calibration (no odometry on the chassis): motion rates at
    # ``ref_speed``, scaled linearly for other speeds. Estimates until
    # measured on the floor — tune via cal_cm_per_s / cal_deg_per_s config.
    cm_per_s: float = 30.0        # forward/strafe at ref_speed
    deg_per_s: float = 80.0       # rotation in place at ref_speed
    ref_speed: float = 0.8
    # Angle from chassis-forward to robot-facing, degrees CCW. See
    # remap_primitive; 0 means the robot faces the way the chassis drives.
    mount_yaw_deg: float = 0.0

    def rate_for(self, direction: str, speed: float) -> tuple[float, str]:
        """(units per second, unit name) for a primitive at this speed."""
        scale = speed / self.ref_speed
        if direction.startswith("rotate"):
            return self.deg_per_s * scale, "deg"
        return self.cm_per_s * scale, "cm"


# All declarations are NON_BLOCKING: the model keeps talking while the
# chassis HTTP round-trip runs, instead of stalling its turn on our latency.
# The paired FunctionResponse carries a `scheduling` hint — see
# ``response_scheduling`` — so a success is absorbed silently while an error
# interrupts and gets spoken.
DRIVE_TOOL_DECLS: list[dict[str, Any]] = [
    {
        "name": "drive",
        "behavior": "NON_BLOCKING",
        "description": (
            "Move the wheeled base you are riding on. Runs for `duration` "
            "seconds then stops automatically. Call again for another leg; "
            "call `stop` to halt early. Executes in the background — keep "
            "talking while it runs."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "direction": {
                    "type": "STRING",
                    "enum": list(PRIMITIVES),
                    "description": (
                        "forward/reverse translate; strafe_* slide sideways "
                        "without turning; rotate_cw turns right, rotate_ccw "
                        "turns left; diagonal_* travel at 45 degrees."
                    ),
                },
                "distance_cm": {
                    "type": "NUMBER",
                    "description": (
                        "For translations: how far to travel, in cm. "
                        "Preferred over duration — the duration is computed "
                        "from calibration. Open-loop estimate: verify with "
                        "your camera. If one command can't cover it, the "
                        "reply says how far you got — call drive again."
                    ),
                },
                "degrees": {
                    "type": "NUMBER",
                    "description": (
                        "For rotate_cw/rotate_ccw: how many degrees to spin. "
                        "Preferred over duration. Same continuation rule as "
                        "distance_cm."
                    ),
                },
                "duration": {
                    "type": "NUMBER",
                    "description": "Raw seconds (0.2–4). Use only when distance_cm/degrees don't fit.",
                },
                "speed": {
                    "type": "NUMBER",
                    "description": "0.2–1.0. Default 0.8. Use lower speeds indoors near obstacles.",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "stop",
        "behavior": "NON_BLOCKING",
        "description": "Stop the wheeled base immediately. Use the instant anyone says stop, wait, or sounds alarmed.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "wheels_state",
        "behavior": "NON_BLOCKING",
        "description": "Read the wheeled base's current state: per-wheel speeds, whether it is moving, last command.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]


# Tier 1–2 declarations (head + body shell). Only offered to the model when
# a RobotMotion is wired in; same NON_BLOCKING contract as driving.
MOTION_TOOL_DECLS: list[dict[str, Any]] = [
    {
        "name": "look",
        "behavior": "NON_BLOCKING",
        "description": (
            "Point your head. Cheapest way to attend to something — use it "
            "freely while talking to glance at what you or the user mention. "
            "Angles are absolute, relative to your body's forward."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "yaw": {
                    "type": "NUMBER",
                    "description": "Degrees left(+)/right(-), about -40..40. 0 = straight ahead.",
                },
                "pitch": {
                    "type": "NUMBER",
                    "description": "Degrees up(+)/down(-), about -25..20. 0 = level.",
                },
            },
        },
    },
    {
        "name": "turn_body",
        "behavior": "NON_BLOCKING",
        "description": (
            "Rotate your body shell by a relative angle to face something "
            "beyond head range, or to square up before driving. The reply "
            "tells you if the shell hits its rotation limit — then use "
            "`drive` to keep turning with the wheels."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "degrees": {
                    "type": "NUMBER",
                    "description": "Relative degrees, left(+)/right(-). Shell range is about ±120 total.",
                },
            },
            "required": ["degrees"],
        },
    },
    {
        "name": "center",
        "behavior": "NON_BLOCKING",
        "description": (
            "Return head and body shell to neutral, facing the same way as "
            "the wheels. Do this when done attending to something, and "
            "before driving forward any real distance."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]

MOTION_TOOL_NAMES = {d["name"] for d in MOTION_TOOL_DECLS}


# Tier 3½: hand the wheels to the vision loop instead of steering by hand.
# Only offered when a FollowManager is wired in (camera + tracking stack).
FOLLOW_TOOL_DECLS: list[dict[str, Any]] = [
    {
        "name": "follow",
        "behavior": "NON_BLOCKING",
        "description": (
            "Lock your camera onto a person or object and keep facing and "
            "approaching it CONTINUOUSLY, on your own, until you are told "
            "to stop. This is the right tool any time someone says follow "
            "me, come with me, keep up, chase it, or stay on something — "
            "far better than repeated drive calls, because it corrects "
            "several times a second while you keep talking."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "target": {
                    "type": "STRING",
                    "description": (
                        "What to follow, in plain words: 'me', 'the person', "
                        "'the dog', 'the ball', 'the cup'. If the reply says "
                        "you have no detector for it, say so — do not "
                        "silently follow something else."
                    ),
                },
                "distance_cm": {
                    "type": "NUMBER",
                    "description": (
                        "Optional: roughly how far behind the target to hold "
                        "station, in cm. Defaults to a comfortable following "
                        "distance for that kind of object."
                    ),
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "stop_following",
        "behavior": "NON_BLOCKING",
        "description": (
            "End the visual follow and stand still. Use it when the user "
            "says stop following, that's enough, or stay here — `stop` also "
            "ends a follow, so use that one if they sound alarmed."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]

FOLLOW_TOOL_NAMES = {d["name"] for d in FOLLOW_TOOL_DECLS}


def response_scheduling(name: str, result: dict[str, Any]) -> str:
    """How the model should treat a NON_BLOCKING tool's late response.

    - errors INTERRUPT: the user believes the robot is moving — correct that
      out loud right now.
    - `wheels_state` answers a question, so deliver it WHEN_IDLE (after the
      current sentence finishes).
    - a success carrying a `note` (a motion tier hit its limit) lands
      WHEN_IDLE too — the model needs it to escalate to the next tier.
    - other successes are SILENT: the model already acknowledged the action
      while the call ran; a second confirmation is just chatter.
    """
    if result.get("status") != "ok":
        return "INTERRUPT"
    if name in ("wheels_state", "follow", "stop_following") or result.get("note"):
        # A follow's outcome changes what the robot will be doing for the
        # next minute; the model should acknowledge it rather than absorb it.
        return "WHEN_IDLE"
    return "SILENT"


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp_drive_args(args: dict[str, Any], limits: DriveLimits) -> tuple[str, float, float]:
    """Validate/clamp a `drive` call. Returns (direction, speed, duration).

    Raises ValueError on a direction the chassis doesn't know — that goes
    back to the model as an error payload, not to the board.
    """
    direction = str(args.get("direction", "")).strip().lower()
    if direction not in PRIMITIVES:
        raise ValueError(f"unknown direction {direction!r}")

    speed = _num(args.get("speed"), limits.default_speed)
    speed = max(limits.min_speed, min(limits.max_speed, speed))
    duration = _num(args.get("duration"), limits.default_duration)
    duration = max(0.2, min(limits.max_duration, duration))
    return direction, speed, duration


def resolve_drive(args: dict[str, Any], limits: DriveLimits) \
        -> tuple[str, float, float, dict[str, Any]]:
    """Ground a `drive` call in physical units.

    Returns (direction, speed, duration, extras). When the model asked in
    distance_cm/degrees (and gave no explicit duration), the duration comes
    from calibration; if the per-command cap truncates it, extras carries a
    `note` telling the model how far this leg goes so it can chain another
    call. extras always reports the estimated coverage of this leg.
    """
    direction, speed, duration = clamp_drive_args(args, limits)
    rate, unit = limits.rate_for(direction, speed)
    extras: dict[str, Any] = {}

    amount_arg = args.get("degrees") if unit == "deg" else args.get("distance_cm")
    if args.get("duration") is None and amount_arg is not None:
        want = abs(_num(amount_arg, 0.0))
        if want > 0 and rate > 0:
            need = want / rate
            duration = max(0.2, min(limits.max_duration, need))
            if need > limits.max_duration + 1e-6:
                extras["note"] = (
                    "one drive command covers only about {:.0f} {} of the "
                    "requested {:.0f} — call drive again to continue".format(
                        rate * duration, unit, want))
    extras[f"estimated_{unit}"] = round(rate * duration, 1)
    return direction, speed, duration, extras


def dispatch_tool(client, name: str, args: dict[str, Any],
                  limits: DriveLimits | None = None,
                  motion=None, follow=None) -> dict[str, Any]:
    """Execute one Gemini tool call against a WheelsClient / RobotMotion.

    Always returns a JSON-safe dict (the FunctionResponse body); never
    raises — the model should hear "the base is unreachable", not silence.
    """
    limits = limits or DriveLimits()
    try:
        if name in FOLLOW_TOOL_NAMES:
            if follow is None:
                return {"status": "error",
                        "error": "visual following is not available"}
            if name == "follow":
                return follow.start(str(args.get("target", "")),
                                    distance_cm=_optional_num(args.get("distance_cm")))
            return follow.stop("the user asked me to stop following")
        if name in MOTION_TOOL_NAMES:
            if motion is None:
                return {"status": "error",
                        "error": "head/body motion is not available"}
            if name == "look":
                return motion.look(args.get("yaw", 0.0), args.get("pitch", 0.0))
            if name == "turn_body":
                return motion.turn_body(args.get("degrees", 0.0))
            return motion.center()
        if name == "drive":
            direction, speed, duration, extras = resolve_drive(args, limits)
            # The model asked in the robot's frame; the board hears its own.
            on_board = remap_primitive(direction, limits.mount_yaw_deg)
            client.command(on_board, speed=speed, duration=duration)
            return {"status": "ok", "action": _DIRECTION_WORDS[direction],
                    "direction": direction, "speed": speed,
                    "duration": round(duration, 2), **extras}
        if name == "stop":
            # An active follow re-commands the base several times a second:
            # halting the chassis without ending the loop would last about
            # one tick. "Stop" always means both.
            was_following = False
            if follow is not None:
                was_following = bool(
                    follow.stop("the user said stop").get("was_active"))
            client.stop()
            return {"status": "ok", "action": "stopped",
                    **({"note": "also ended the follow"} if was_following else {})}
        if name == "wheels_state":
            return {"status": "ok", "state": client.state()}
        return {"status": "error", "error": f"unsupported tool {name!r}"}
    except (ValueError, WheelsError) as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — never stall the Live session
        return {"status": "error", "error": f"unexpected: {exc!r}"}


def describe_tool_result(name: str, args: dict[str, Any],
                         result: dict[str, Any]) -> str:
    """One transcript line for the UI, e.g. 'drive forward 1.5s @0.8 → ok'."""
    if name == "drive":
        detail = "{} {}s @{}".format(
            result.get("direction", args.get("direction", "?")),
            result.get("duration", "?"), result.get("speed", "?"))
    elif name == "look":
        detail = "look yaw={} pitch={}".format(
            result.get("head_yaw", args.get("yaw", "?")),
            result.get("head_pitch", args.get("pitch", "?")))
    elif name == "turn_body":
        detail = "turn body {}°".format(result.get("turned", args.get("degrees", "?")))
    elif name == "follow":
        detail = "follow {}".format(result.get("target", args.get("target", "?")))
    else:
        detail = name
    status = result.get("status", "?")
    if status != "ok":
        return f"{detail} → {result.get('error', 'error')}"
    return f"{detail} → ok"
