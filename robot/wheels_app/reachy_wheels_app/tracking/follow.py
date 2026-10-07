"""The follow controller: a tracked box in, a motion intent out.

Pure and synchronous — no SDK, no HTTP, no threads, no clock of its own
(``now`` is passed in). That is the point: the loop's actual behaviour
(when it turns, when it rolls, when it gives up) is decided here and is
therefore testable on a laptop, while ``session.py`` only has to move
bytes and joints.

## Control law

The head does the looking and wheels do the facing. Shell assistance transfers
neck offset into the shell while preserving absolute camera gaze. Chassis yaw
pauses during this transfer. Startup recentering is opt-in. Lost-target search
scans head up/down/left/right, then moves the shell and repeats, then uses
stopped wheel sectors with a fresh head scan between them.

  gaze   the head slews so the target sits at the centre of frame — this is
         cheap, fast and is what makes the robot look attentive
  yaw    the wheels rotate to drive the target's angle *relative to
         wheel-forward* (head yaw + body yaw − image bearing) to zero
  range  the wheels roll forward while the target's apparent size is below
         ``target_size``, gated on being roughly on-axis already

Both wheel axes leave as one ``move`` mix (``vx`` forward, ``omega`` CCW),
so turning and approaching happen in the same motion rather than in
alternating jerks.

## Sign conventions (they bite)

* image bearing: **positive = target is to the RIGHT** of frame centre
* robot yaw: **positive = LEFT** (matches ``voice/motion.py`` and the
  chassis's ``omega``), hence the negation when going from one to the other
* pitch: positive = up (tool convention, the SDK's negation lives in
  ``RobotMotion``)

## Safety

Every emitted command carries a short ``duration``; the chassis's own
deadman stops the base if this loop stalls or dies. Losing the target
walks through hold → search → give up, and the give-up is terminal
(``done``) rather than a robot that keeps rolling at nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Detection, FrameInfo
from .vocab import known_height_m

# Phases, in roughly the order a session moves through them.
ACQUIRING = "acquiring"
FOLLOWING = "following"
ARRIVED = "arrived"
HOLDING = "holding"
SEARCHING = "searching"
LOST = "lost"
TIMEOUT = "timeout"


@dataclass(frozen=True)
class FollowConfig:
    # --- gaze (head) ----------------------------------------------------
    head_yaw_limit: float = 40.0
    head_pitch_up: float = 20.0
    head_pitch_down: float = 25.0
    gaze_gain: float = 0.85       # <1 damps overshoot; goto_target blocks
    gaze_deadband_deg: float = 2.5
    gaze_step_limit: float = 12.0

    # --- yaw tier 2: the body shell ------------------------------------
    # The shell is the middle tier between a head glance and rolling the
    # wheels: quieter and more precise than driving, and it is what makes
    # the robot visibly *face* what it is following rather than staring
    # sideways at it. Once the head is carrying more than this much of the
    # offset, the shell rotates to absorb it and the head recentres.
    body_assist_deg: float = 12.0
    body_assist_enabled: bool = True
    body_step_limit: float = 6.0   # coordinated shell/neck transfer
    body_yaw_limit: float = 120.0   # shell travel each way (cables)

    # --- yaw tier 3: wheels rotate to face ------------------------------
    yaw_deadband_deg: float = 6.0
    yaw_gain: float = 0.015       # normalised omega per degree of error
    max_omega: float = 0.35

    # --- range (wheels roll to close) -----------------------------------
    # How far behind the target to hold station. Metres is the honest unit
    # now that the camera is calibrated: `target_size` below is a fraction
    # of the FRAME, so what it means in the room depends entirely on the
    # lens. On this robot's measured 42° vertical FOV, size 0.55 works out
    # to standing four metres back — which reads as the robot retreating.
    # When the class has a height prior, this wins and target_size is
    # derived from it; otherwise target_size is used as-is.
    follow_distance_m: float = 2.0
    # Fraction of frame the target should occupy when we are "there".
    # Fallback for classes with no height prior.
    target_size: float = 0.55
    size_deadband: float = 0.06   # hysteresis around target_size
    forward_gain: float = 2.2     # normalised vx per unit of size error
    max_vx: float = 0.8
    approach_gate_deg: float = 25.0  # must be this on-axis before rolling
    allow_reverse: bool = True    # back off when the target crowds us
    max_reverse_vx: float = 0.35

    # Snap head and shell to neutral before following. OFF by default: for
    # "follow me" the target is usually already in view, and recentring is
    # the fastest way to throw it out of frame before the loop ever sees
    # it. The wheel-frame yaw error already accounts for a rotated shell,
    # so centring buys nothing the controller needs.
    center_on_start: bool = False

    # --- chassis command shaping ----------------------------------------
    # Top-heavy mount: translate gently, rotation needs the torque.
    drive_speed: float = 0.5
    rotate_speed: float = 0.7
    command_duration: float = 0.7  # each command's own deadman window

    # --- losing the target ----------------------------------------------
    hold_seconds: float = 2.5      # pause at last gaze through detector gaps
    lock_confirm_frames: int = 2
    # How long to hunt for a target that has gone before giving up. Search
    # is deliberately sequential: a measured four-pose head scan, one shell
    # reposition followed by the same head scan, then short wheel turns with
    # a fresh head scan after each. This is long enough to complete that
    # ladder at the detector's measured ~1 Hz rate.
    search_seconds: float = 60.0
    # Finding the target the FIRST time is a different problem from
    # recovering one you had: the head may have to sweep its whole range,
    # and on-robot detection runs near 1 Hz, so six seconds is about five
    # attempts. Give acquisition its own, longer budget.
    acquire_seconds: float = 60.0
    search_omega: float = 0.18
    search_max_omega: float = 0.18  # also caps older persisted search settings
    search_step_limit: float = 20.0  # cap each head/shell search correction
    # Maximum time allowed for one requested head pose. The search advances
    # even if an asymmetric neck cannot quite attain an endpoint; otherwise
    # it can buzz forever asking for the same unreachable angle.
    search_waypoint_seconds: float = 5.0
    search_waypoint_hold_seconds: float = 0.35
    search_settle_deg: float = 2.0
    # Retained for old state files/API clients. The former blended triangle
    # wave used these; ordered search intentionally does not.
    search_sweep_period: float = 16.0
    search_widen_seconds: float = 6.0
    # The middle tier is one bounded shell reposition, then another complete
    # head scan from the new viewpoint. It is not blended into head motion.
    search_body_step_deg: float = 60.0
    search_body_seconds: float = 8.0
    # Compatibility floor for older saved settings. The state machine also
    # requires the head and shell stages to complete, so a persisted value of
    # six seconds can no longer make the wheels pre-empt them.
    search_wheels_after: float = 0.0
    # Once wheel search begins, rotate one sector, stop, and scan with the
    # head again. Continuous blind spinning made detections blurry and made
    # it look as though the robot knew only how to rotate.
    search_wheel_step_deg: float = 45.0
    # A search must eventually look EVERYWHERE, and head+shell only reach
    # ±160° — there is a blind wedge behind the robot that no amount of
    # twisting covers. So once the wheels join, the base turns steadily
    # through a full circle, carrying the sweeping head and shell with it,
    # and the union covers 360° several times over. This is the budget for
    # one complete revolution.
    search_full_turn_seconds: float = 25.0
    # Base spin rate at `rotate_speed`, deg/s — the same open-loop
    # calibration the voice drive tools use. It only sizes the search spin;
    # if it is off the sweep is slower or faster, never wrong.
    rotate_deg_per_s: float = 80.0
    max_session_seconds: float = 240.0

    def size_for_distance(self, class_name: str, distance_m: float,
                          frame: FrameInfo) -> float | None:
        """Apparent size that ``distance_m`` corresponds to, if we can tell.

        Lets a caller say "follow me at about a metre" and have it mean
        something, without the controller having to trust monocular metres
        as its control variable.
        """
        height_m = known_height_m(class_name)
        if not height_m or distance_m <= 0:
            return None
        half = math.tan(math.radians(frame.vfov_deg) / 2.0)
        if half <= 0:
            return None
        size = height_m / (2.0 * distance_m * half)
        # Cap below 1.0 with room for the deadband: a target size the box
        # can never reach (a close-up subject is cropped by the frame edge,
        # so apparent size saturates) would mean the robot never decides it
        # has arrived and keeps closing in.
        return max(0.05, min(0.85, size))


@dataclass(frozen=True)
class FollowCommand:
    """What the executor should do about this frame."""

    phase: str
    # Absolute head angles in degrees, or None to leave the head where it is.
    head_yaw: float | None = None
    head_pitch: float | None = None
    # Relative shell rotation, degrees, positive = left. None = leave it.
    body_yaw_delta: float | None = None
    # Body-frame velocity mix for the chassis, normalised to [-1, 1].
    vx: float = 0.0
    omega: float = 0.0
    speed: float = 0.5
    duration: float = 0.7
    stop: bool = False   # send an explicit /stop before anything else
    done: bool = False   # the session is over (lost / timed out)
    note: str = ""
    # Telemetry for the UI and the voice agent.
    bearing_deg: float | None = None
    yaw_error_deg: float | None = None
    size: float | None = None
    distance_m: float | None = None
    # How much azimuth a search has swept so far, degrees, capped at 360.
    search_covered_deg: float | None = None

    @property
    def moving(self) -> bool:
        return abs(self.vx) > 1e-3 or abs(self.omega) > 1e-3

    def to_dict(self) -> dict:
        out = {"phase": self.phase, "moving": self.moving,
               "vx": round(self.vx, 3), "omega": round(self.omega, 3)}
        if self.body_yaw_delta is not None:
            out["body_yaw_delta"] = round(self.body_yaw_delta, 1)
        for key in ("bearing_deg", "yaw_error_deg", "size", "distance_m",
                    "search_covered_deg"):
            value = getattr(self, key)
            if value is not None:
                out[key] = round(value, 2)
        if self.note:
            out["note"] = self.note
        return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apparent_size(det: Detection, frame: FrameInfo) -> float:
    """Fraction of the frame the box spans, on its larger axis.

    Height alone is the better range proxy for upright things, width for
    round ones; taking the max makes one number work for both, and degrades
    gracefully when a close-up target is cropped by the frame edge.
    """
    if frame.width <= 0 or frame.height <= 0:
        return 0.0
    return min(1.0, max(det.width / frame.width, det.height / frame.height))


def estimate_distance_m(det: Detection, frame: FrameInfo) -> float | None:
    """Rough metres to the target from its box height and a class prior.

    Monocular and unashamedly approximate — reported to the user, never
    used as the control variable (a cropped box would lie about it).
    """
    height_m = known_height_m(det.class_name)
    if not height_m or det.height <= 1 or frame.height <= 0:
        return None
    half = math.tan(math.radians(frame.vfov_deg) / 2.0)
    if half <= 0:
        return None
    return (height_m * frame.height) / (2.0 * det.height * half)


class FollowController:
    """Stateful across frames only for timing and search direction."""

    def __init__(self, config: FollowConfig | None = None, *,
                 started_at: float = 0.0):
        self.config = config or FollowConfig()
        self._started_at = started_at
        self._last_seen_at = started_at
        self._ever_seen = False
        self._last_yaw_error = 0.0
        self._reset_search()

    # --- lifecycle -------------------------------------------------------

    def note_start(self, now: float) -> None:
        self._started_at = now
        self._last_seen_at = now
        self._reset_search()

    def _reset_search(self) -> None:
        self._search_stage: str | None = None
        self._search_stage_started = 0.0
        self._search_waypoint = 0
        self._search_waypoint_started = 0.0
        self._search_waypoint_reached: float | None = None
        self._search_body_goal = 0.0
        self._search_direction = 1.0
        self._search_wheel_started: float | None = None
        self._search_wheel_turned = 0.0

    @property
    def ever_seen(self) -> bool:
        return self._ever_seen

    def seconds_since_seen(self, now: float) -> float:
        return max(0.0, now - self._last_seen_at)

    # --- the step --------------------------------------------------------

    def step(self, target: Detection | None, frame: FrameInfo, *,
             head_yaw: float = 0.0, head_pitch: float = 0.0,
             body_yaw: float = 0.0, now: float = 0.0) -> FollowCommand:
        cfg = self.config

        if now - self._started_at > cfg.max_session_seconds:
            return FollowCommand(
                phase=TIMEOUT, stop=True, done=True,
                note="follow session hit its {:.0f}s limit".format(
                    cfg.max_session_seconds))

        if target is None:
            return self._step_lost(frame, head_yaw=head_yaw,
                                   head_pitch=head_pitch,
                                   body_yaw=body_yaw, now=now)

        self._ever_seen = True
        self._last_seen_at = now
        self._reset_search()
        return self._step_visible(target, frame, head_yaw=head_yaw,
                                  head_pitch=head_pitch, body_yaw=body_yaw)

    # --- target in view --------------------------------------------------

    def _step_visible(self, target: Detection, frame: FrameInfo, *,
                      head_yaw: float, head_pitch: float,
                      body_yaw: float) -> FollowCommand:
        cfg = self.config
        cx, cy = target.center
        bearing = frame.bearing_deg(cx)        # + = target right of centre
        elevation = frame.elevation_deg(cy)    # + = target above centre

        # Gaze: image bearing is right-positive, robot yaw is left-positive.
        want_yaw = _clamp(head_yaw - _clamp(cfg.gaze_gain * bearing,
                          -cfg.gaze_step_limit, cfg.gaze_step_limit),
                          -cfg.head_yaw_limit, cfg.head_yaw_limit)
        want_pitch = _clamp(head_pitch + _clamp(cfg.gaze_gain * elevation,
                            -cfg.gaze_step_limit, cfg.gaze_step_limit),
                            -cfg.head_pitch_down, cfg.head_pitch_up)
        if abs(want_yaw - head_yaw) < cfg.gaze_deadband_deg:
            want_yaw = None
        if abs(want_pitch - head_pitch) < cfg.gaze_deadband_deg:
            want_pitch = None

        # Where the target sits relative to WHEEL-forward: the head is
        # already pointing `head_yaw` off the shell, the shell `body_yaw`
        # off the wheels, and the target `bearing` off the head's axis.
        yaw_error = head_yaw + body_yaw - bearing
        self._last_yaw_error = yaw_error

        # Tier 2: hand the head's accumulated offset to the shell, so the
        # robot turns to FACE the target instead of tracking it out of the
        # corner of its eye. The head recentres by itself from the next
        # frame, because the camera rides on the shell.
        body_delta = None
        if (cfg.body_assist_enabled
                and abs(bearing) <= max(cfg.gaze_deadband_deg * 2.0, 5.0)
                and abs(head_yaw) > cfg.body_assist_deg):
            wanted = _clamp(head_yaw, -cfg.body_step_limit, cfg.body_step_limit)
            room = (cfg.body_yaw_limit - body_yaw if wanted > 0
                    else cfg.body_yaw_limit + body_yaw)
            wanted = math.copysign(min(abs(wanted), max(0.0, room)), wanted)
            if abs(wanted) >= 1.0:
                body_delta = wanted

        omega = 0.0
        if abs(yaw_error) > cfg.yaw_deadband_deg:
            omega = _clamp(cfg.yaw_gain * yaw_error,
                           -cfg.max_omega, cfg.max_omega)
        if body_delta is not None:
            # Let the coordinated shell/neck transfer finish before rotating
            # the chassis. This reduces the neck offset rather than adding yaw.
            omega = 0.0

        size = apparent_size(target, frame)
        size_error = cfg.target_size - size
        vx = 0.0
        note = ""
        if abs(yaw_error) > cfg.approach_gate_deg:
            note = "turning to face the target before closing in"
        elif size_error > cfg.size_deadband:
            vx = _clamp(cfg.forward_gain * size_error, 0.0, cfg.max_vx)
        elif size_error < -cfg.size_deadband:
            if cfg.allow_reverse:
                vx = -_clamp(cfg.forward_gain * -size_error,
                             0.0, cfg.max_reverse_vx)
                note = "target is closer than the follow distance — easing back"
            else:
                note = "target is closer than the follow distance"

        if body_delta is not None:
            vx = 0.0
            note = "transferring gaze to body shell before approach"

        aligned = abs(yaw_error) <= cfg.yaw_deadband_deg
        at_range = abs(size_error) <= cfg.size_deadband
        phase = ARRIVED if (aligned and at_range) else FOLLOWING
        if phase == ARRIVED and not note:
            note = "holding station at the follow distance"

        return FollowCommand(
            phase=phase,
            head_yaw=want_yaw, head_pitch=want_pitch,
            body_yaw_delta=body_delta,
            vx=vx, omega=omega,
            speed=cfg.rotate_speed if abs(vx) < 0.05 else cfg.drive_speed,
            duration=cfg.command_duration,
            note=note,
            bearing_deg=bearing, yaw_error_deg=yaw_error,
            size=size, distance_m=estimate_distance_m(target, frame),
        )

    # --- target not in view ----------------------------------------------

    def _step_lost(self, frame: FrameInfo, *, head_yaw: float,
                   head_pitch: float = 0.0, body_yaw: float = 0.0,
                   now: float = 0.0) -> FollowCommand:
        cfg = self.config
        gone = now - self._last_seen_at

        if gone <= cfg.hold_seconds:
            # A single dropped detection is normal (motion blur, occlusion).
            # Emit nothing: the executor stops the base on the moving→still
            # edge and the board deadman is the backstop either way.
            return FollowCommand(phase=HOLDING, note="target briefly out of view")

        search_window = cfg.search_seconds if self._ever_seen else cfg.acquire_seconds
        if gone <= cfg.hold_seconds + search_window:
            elapsed = gone - cfg.hold_seconds
            return self._search_step(
                head_yaw=head_yaw, head_pitch=head_pitch,
                body_yaw=body_yaw, now=now, elapsed=elapsed)

        return FollowCommand(
            phase=LOST, stop=True, done=True,
            note="could not find the target again — stopped"
            if self._ever_seen else "never saw the target — stopped",
        )

    # --- ordered target search ------------------------------------------

    def _start_search_stage(self, stage: str, now: float) -> None:
        self._search_stage = stage
        self._search_stage_started = now
        self._search_waypoint = 0
        self._search_waypoint_started = now
        self._search_waypoint_reached = None

    def _search_prefix(self) -> str:
        return ("lost the target — reacquiring"
                if self._ever_seen else "looking for the target")

    def _search_step(self, *, head_yaw: float, head_pitch: float,
                     body_yaw: float, now: float,
                     elapsed: float) -> FollowCommand:
        cfg = self.config
        if self._search_stage is None:
            self._search_direction = 1.0 if self._last_yaw_error >= 0 else -1.0
            self._start_search_stage("head_initial", now)

        if self._search_stage in ("head_initial", "head_body", "head_wheel"):
            return self._search_head_step(
                head_yaw=head_yaw, head_pitch=head_pitch,
                body_yaw=body_yaw, now=now, elapsed=elapsed)

        if self._search_stage == "body_move":
            timed_out = now - self._search_stage_started >= cfg.search_body_seconds
            if timed_out:
                self._start_search_stage("head_body", now)
                return self._search_head_step(
                    head_yaw=head_yaw, head_pitch=head_pitch,
                    body_yaw=body_yaw, now=now, elapsed=elapsed)
            # Finish the head tier before the shell tier. Returning the neck
            # to neutral also gives the shell its full safe cable-limited arc.
            centred = (abs(head_yaw) <= cfg.search_settle_deg
                        and abs(head_pitch) <= cfg.search_settle_deg)
            if not centred:
                return self._search_head_command(
                    head_yaw, head_pitch, 0.0, 0.0,
                    f"{self._search_prefix()}: centering head before shell")

            remaining = self._search_body_goal - body_yaw
            if abs(remaining) <= cfg.search_settle_deg:
                self._start_search_stage("head_body", now)
                return self._search_head_step(
                    head_yaw=head_yaw, head_pitch=head_pitch,
                    body_yaw=body_yaw, now=now, elapsed=elapsed)
            delta = _clamp(
                remaining,
                -min(cfg.body_step_limit, cfg.search_step_limit),
                min(cfg.body_step_limit, cfg.search_step_limit))
            return FollowCommand(
                phase=SEARCHING, body_yaw_delta=delta,
                speed=cfg.rotate_speed, duration=cfg.command_duration,
                note=f"{self._search_prefix()}: repositioning body shell",
                search_covered_deg=self._search_covered())

        # Wheel turns are short, stopped sectors. A complete head scan runs
        # after each one instead of asking the detector to see through blur.
        if self._search_stage == "wheel_turn":
            centred = (abs(head_yaw) <= cfg.search_settle_deg
                        and abs(head_pitch) <= cfg.search_settle_deg)
            if (not centred and
                    now - self._search_stage_started < cfg.search_waypoint_seconds):
                return self._search_head_command(
                    head_yaw, head_pitch, 0.0, 0.0,
                    f"{self._search_prefix()}: centering head before wheel turn")
            if elapsed < max(0.0, cfg.search_wheels_after):
                return FollowCommand(
                    phase=SEARCHING, stop=True,
                    note=f"{self._search_prefix()}: upper tiers still have priority",
                    search_covered_deg=self._search_covered())

            spin_deg_per_s = 360.0 / max(1.0, cfg.search_full_turn_seconds)
            omega_mag = _clamp(
                spin_deg_per_s / max(1e-6, cfg.rotate_deg_per_s), 0.0,
                max(0.0, min(cfg.search_omega, cfg.search_max_omega,
                             cfg.max_omega)))
            effective_deg_per_s = omega_mag * cfg.rotate_deg_per_s
            if self._search_wheel_started is None:
                self._search_wheel_started = now
            turned = effective_deg_per_s * (now - self._search_wheel_started)
            if effective_deg_per_s <= 0.0 or turned >= cfg.search_wheel_step_deg:
                self._search_wheel_turned += min(cfg.search_wheel_step_deg, turned)
                self._search_wheel_started = None
                self._start_search_stage("head_wheel", now)
                return FollowCommand(
                    phase=SEARCHING, stop=True,
                    note=f"{self._search_prefix()}: wheel sector complete; scanning head",
                    search_covered_deg=self._search_covered())
            return FollowCommand(
                phase=SEARCHING,
                omega=self._search_direction * omega_mag,
                speed=cfg.rotate_speed, duration=cfg.command_duration,
                note=(f"{self._search_prefix()}: rotating wheels to the next "
                      f"viewpoint ({self._search_covered():.0f}° covered)"),
                search_covered_deg=self._search_covered())

        # Defensive fallback for a corrupted in-memory stage.
        self._start_search_stage("head_initial", now)
        return self._search_head_step(
            head_yaw=head_yaw, head_pitch=head_pitch,
            body_yaw=body_yaw, now=now, elapsed=elapsed)

    def _search_head_step(self, *, head_yaw: float, head_pitch: float,
                          body_yaw: float, now: float,
                          elapsed: float) -> FollowCommand:
        cfg = self.config
        # Exact operator-requested order. Positive yaw/pitch are left/up.
        waypoints = (
            (0.0, cfg.head_pitch_up, "up"),
            (0.0, -cfg.head_pitch_down, "down"),
            (cfg.head_yaw_limit, 0.0, "left"),
            (-cfg.head_yaw_limit, 0.0, "right"),
        )
        target_yaw, target_pitch, name = waypoints[self._search_waypoint]
        reached = (abs(target_yaw - head_yaw) <= cfg.search_settle_deg
                   and abs(target_pitch - head_pitch) <= cfg.search_settle_deg)
        if reached:
            if self._search_waypoint_reached is None:
                self._search_waypoint_reached = now
            ready = now - self._search_waypoint_reached >= cfg.search_waypoint_hold_seconds
        else:
            ready = now - self._search_waypoint_started >= cfg.search_waypoint_seconds

        if ready:
            self._search_waypoint += 1
            self._search_waypoint_started = now
            self._search_waypoint_reached = None
            if self._search_waypoint >= len(waypoints):
                finished = self._search_stage
                if finished == "head_initial":
                    room_goal = body_yaw + self._search_direction * cfg.search_body_step_deg
                    goal = _clamp(room_goal, -cfg.body_yaw_limit, cfg.body_yaw_limit)
                    if abs(goal - body_yaw) < cfg.search_settle_deg:
                        goal = _clamp(
                            body_yaw - self._search_direction * cfg.search_body_step_deg,
                            -cfg.body_yaw_limit, cfg.body_yaw_limit)
                        self._search_direction *= -1.0
                    self._search_body_goal = goal
                    self._start_search_stage("body_move", now)
                else:
                    self._start_search_stage("wheel_turn", now)
                return self._search_step(
                    head_yaw=head_yaw, head_pitch=head_pitch,
                    body_yaw=body_yaw, now=now, elapsed=elapsed)
            target_yaw, target_pitch, name = waypoints[self._search_waypoint]

        return self._search_head_command(
            head_yaw, head_pitch, target_yaw, target_pitch,
            f"{self._search_prefix()}: head scan {name}")

    def _search_head_command(self, head_yaw: float, head_pitch: float,
                             target_yaw: float, target_pitch: float,
                             note: str) -> FollowCommand:
        cfg = self.config
        yaw_error, pitch_error = target_yaw - head_yaw, target_pitch - head_pitch
        yaw = None if abs(yaw_error) <= cfg.search_settle_deg else \
            head_yaw + _clamp(yaw_error, -cfg.search_step_limit, cfg.search_step_limit)
        pitch = None if abs(pitch_error) <= cfg.search_settle_deg else \
            head_pitch + _clamp(pitch_error, -cfg.search_step_limit, cfg.search_step_limit)
        return FollowCommand(
            phase=SEARCHING, head_yaw=yaw, head_pitch=pitch,
            speed=cfg.rotate_speed, duration=cfg.command_duration,
            note=note, search_covered_deg=self._search_covered())

    def _search_covered(self) -> float:
        cfg = self.config
        upper = 2.0 * cfg.head_yaw_limit
        if self._search_stage not in (None, "head_initial"):
            upper += abs(self._search_body_goal)
        return min(360.0, upper + self._search_wheel_turned)
