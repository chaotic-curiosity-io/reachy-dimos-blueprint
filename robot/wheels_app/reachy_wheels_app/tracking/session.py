"""The follow loop: perception thread, actuator threads, and the manager.

Wiring only — every decision worth arguing about lives in ``follow.py``.

## Threads, and why there are three

``perceive``  grab frame → detect → track → pick the locked target → ask the
              controller what to do. Must never block on a motor or a
              socket, or the robot keeps acting on a stale view of a moving
              person.
``head``      applies gaze. ``goto_target`` *blocks* for the move duration,
              so it gets its own thread with a latest-wins slot: a queued
              head angle that has been superseded is simply dropped.
``wheels``    applies the drive mix. Chassis HTTP is ~300 ms per round trip
              and the ESP32 takes one connection at a time; same latest-wins
              slot, so a stop never waits behind a stale drive.

## Target lock

Detections are anonymous per frame; tracks are not. We lock one track id at
acquisition (biggest, most central, most confident) and follow *that* id,
so a second person walking through the frame can't steal the robot. If the
locked id ages out, we re-lock to the best candidate of the same class
rather than ending the session — people go behind furniture.

## Stopping

Three independent ways this stops, by design: the user says stop (or hits
the button), the controller gives up (lost / session timeout), or every
command's short ``duration`` expires and the board's own deadman halts the
base. The third one is the invariant — it holds even if this whole process
dies mid-drive.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from .annotate import annotate
from .detectors import DetectorUnavailable, build_detector, filter_detections
from .follow import FollowCommand, FollowConfig, FollowController
from .tracker import TargetTracker
from .types import Detection, FrameInfo, robot_to_chassis
from .vocab import COCO_CLASSES, known_height_m, resolve_target
from ..wheels_client import WheelsError

_log = logging.getLogger(__name__)


class TrackBoard:
    """Thread-safe status + event ring + latest preview frame."""

    def __init__(self, max_events: int = 40):
        self._lock = threading.Lock()
        self._status: dict = {"active": False, "phase": "idle", "detail": ""}
        self._events: deque[dict] = deque(maxlen=max_events)
        self._preview: bytes = b""

    def set_status(self, **fields) -> None:
        with self._lock:
            self._status.update(fields)

    def replace_status(self, status: dict) -> None:
        with self._lock:
            self._status = dict(status)

    def event(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            if self._events and self._events[-1]["text"] == text:
                return  # the loop repeats itself; the log shouldn't
            self._events.append({"t": time.time(), "text": text})

    def set_preview(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        with self._lock:
            self._preview = jpeg

    def preview(self) -> bytes:
        with self._lock:
            return self._preview

    def snapshot(self) -> dict:
        with self._lock:
            return {**self._status, "events": list(self._events)}


class _LatestSlot:
    """One worker thread applying only the most recent item handed to it.

    Coalescing is the point: a follow loop producing 8 head angles a second
    into an actuator that takes 400 ms to apply one must drop the stale
    ones, not queue them and lag further behind every second.
    """

    def __init__(self, name: str, apply, on_error=None):
        self._apply = apply
        self._on_error = on_error
        self._cv = threading.Condition()
        self._pending = None
        self._has_pending = False
        self._running = True
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, item) -> None:
        with self._cv:
            self._pending = item
            self._has_pending = True
            self._cv.notify()

    def close(self, timeout: float = 2.0) -> None:
        """Stop the worker, DISCARDING anything still queued.

        Applying one last command on the way out would mean a drive
        arriving after the user pressed stop — bounded by the deadman, but
        still motion nobody asked for.
        """
        with self._cv:
            self._running = False
            self._pending, self._has_pending = None, False
            self._cv.notify()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._running and not self._has_pending:
                    self._cv.wait(0.2)
                if not self._running and not self._has_pending:
                    return
                item, self._pending, self._has_pending = self._pending, None, False
            try:
                self._apply(item)
            except Exception as exc:  # noqa: BLE001 — an actuator fault must
                # not kill the worker; the next command gets a fresh attempt.
                _log.exception("actuator %s failed", self._thread.name)
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:  # noqa: BLE001
                        pass


def select_target(candidates: list[Detection], frame: FrameInfo) -> Detection | None:
    """Pick which of several matching objects to lock onto.

    Big and central beats small and peripheral: the thing the user pointed
    the robot at is almost always the one filling the middle of the view,
    and a large box is also the one whose range estimate is worth trusting.
    """
    if not candidates:
        return None

    def rank(det: Detection) -> float:
        cx, _ = det.center
        offset = abs(cx / frame.width - 0.5) * 2.0 if frame.width else 1.0
        area = det.area / max(1.0, frame.width * frame.height)
        return (area ** 0.5) * (1.0 - 0.5 * min(1.0, offset)) * max(0.05, det.score)

    return max(candidates, key=rank)


class FollowSession(threading.Thread):
    """One follow run, start to stop."""

    def __init__(self, *, frame_provider, wheels_provider, motion, detector,
                 board: TrackBoard, config: FollowConfig,
                 target_phrase: str, labels: set[str],
                 tracker: TargetTracker | None = None,
                 rate_hz: float = 8.0, preview_hz: float = 3.0,
                 relock_seconds: float = 1.5, hfov_deg: float = 70.0,
                 vfov_deg: float = 55.0, mount_yaw_deg: float = 0.0,
                 on_finish=None):
        super().__init__(name="reachy-wheels-follow", daemon=True)
        self._frames = frame_provider
        # A provider, not a client: WheelsLink rebuilds its client whenever
        # settings are saved, so a session holding the object would keep
        # commanding the OLD chassis address after the host is changed.
        self._wheels_provider = wheels_provider
        self._motion = motion
        self._detector = detector
        self._board = board
        self._config = config
        self._target_phrase = target_phrase
        self._labels = labels
        self._tracker = tracker or TargetTracker(frame_rate=rate_hz)
        self._period = 1.0 / max(1.0, rate_hz)
        self._preview_period = 1.0 / max(0.2, preview_hz)
        self._relock_seconds = relock_seconds
        self._hfov, self._vfov = float(hfov_deg), float(vfov_deg)
        self._mount_yaw = float(mount_yaw_deg)
        self._on_finish = on_finish

        self._stop_event = threading.Event()
        self._controller = FollowController(config)
        self._locked_id: int | None = None
        self._locked_lost_at: float | None = None
        self._last_preview = 0.0
        self._was_moving = False
        self._wheel_hold_until = 0.0
        self._fps = 0.0
        self._stop_reason = ""
        self._superseded = False
        self._last_size = (640, 480)   # until a real frame says otherwise
        self._confirm_id = None
        self._confirm_hits = 0
        self._confirm_seen_at = None

    # --- control ---------------------------------------------------------

    def request_stop(self, reason: str = "stopped by request") -> None:
        if not self._stop_reason:
            self._stop_reason = reason
        self._stop_event.set()

    @property
    def target_phrase(self) -> str:
        return self._target_phrase

    @property
    def tracker_backend(self) -> str:
        return self._tracker.backend

    # --- loop ------------------------------------------------------------

    def supersede(self) -> None:
        """Another session has taken over; stop writing to the shared board.

        A session whose teardown outran its join would otherwise stamp
        "stopped" over the status of the session that replaced it.
        """
        self._superseded = True

    def run(self) -> None:
        started = time.monotonic()
        self._controller.note_start(started)
        self._board.replace_status({
            "active": True, "phase": "acquiring", "detail": "",
            "target": self._target_phrase, "labels": sorted(self._labels),
            "detector": getattr(self._detector, "name", "?"),
            "tracker": self.tracker_backend,
            "started_at": time.time(),
        })
        self._board.event(f"following “{self._target_phrase}”")
        if self._motion is not None and self._config.center_on_start:
            # Opt-in only — see FollowConfig.center_on_start. Recentring
            # here used to throw the target out of frame before the loop
            # had seen it even once.
            try:
                self._motion.center()
            except Exception:  # noqa: BLE001
                _log.exception("could not centre before following")

        head = wheels = None
        try:
            head = _LatestSlot("follow-head", self._apply_head,
                               on_error=lambda exc: self.request_stop(f"head motion failed: {exc}"))
            wheels = _LatestSlot("follow-wheels", self._apply_wheels,
                                 on_error=lambda exc: self.request_stop(f"wheel motion failed: {exc}"))
            while not self._stop_event.is_set():
                tick = time.monotonic()
                command = self._tick(tick)
                if command is not None:
                    # A hold/lock command must also replace a queued search
                    # sweep, even though it contains no new head angles.
                    head.submit(command)
                    wheels.submit(command)
                    if command.done:
                        self._stop_reason = command.note or command.phase
                        break
                elapsed = time.monotonic() - tick
                self._fps = 1.0 / max(1e-3, elapsed) if elapsed > self._period \
                    else 1.0 / self._period
                time.sleep(max(0.0, self._period - elapsed))
        except Exception as exc:  # noqa: BLE001
            _log.exception("follow loop crashed")
            self._stop_reason = f"follow loop crashed: {exc!r}"
        finally:
            for slot in (head, wheels):
                if slot is not None:
                    slot.close()
            self._halt()
            reason = self._stop_reason or "stopped"
            if not self._superseded:
                self._board.set_status(active=False, phase="stopped",
                                       detail=reason, moving=False)
                self._board.event(reason)
            _log.info("follow session ended: %s", reason)
            if self._on_finish is not None:
                try:
                    self._on_finish(self)
                except Exception:  # noqa: BLE001
                    _log.exception("follow on_finish raised")

    def _blind_tick(self, now: float, detail: str) -> FollowCommand:
        """A tick where perception produced nothing usable.

        Routed through the controller as "target not visible" rather than
        returned early: hold → search → give up, and the session time
        limit, all live in there. Bailing out before `step()` would let a
        camera that never yields a frame, or a detector that raises every
        frame, spin an "active" session forever.
        """
        info = FrameInfo(width=self._last_size[0], height=self._last_size[1],
                         hfov_deg=self._hfov, vfov_deg=self._vfov)
        posture = self._posture()
        command = self._controller.step(
            None, info, head_yaw=posture["head_yaw"],
            head_pitch=posture["head_pitch"], body_yaw=posture["body_yaw"],
            now=now)
        # Camera/detector failure is NOT evidence that searching is useful.
        # Never rotate blind; still honor the controller's terminal timeout.
        from dataclasses import replace
        self._confirm_hits = 0
        command = replace(command, head_yaw=None, head_pitch=None,
                          body_yaw_delta=None, vx=0.0, omega=0.0, stop=True,
                          phase=command.phase if command.done else "holding",
                          note=detail or command.note)
        self._publish(command, [], None, now)
        return command

    def _tick(self, now: float) -> FollowCommand | None:
        frame = self._frames()
        if frame is None:
            return self._blind_tick(now, "waiting for a camera frame")

        h, w = frame.shape[:2]
        self._last_size = (w, h)
        info = FrameInfo(width=w, height=h,
                         hfov_deg=self._hfov, vfov_deg=self._vfov)
        try:
            raw = self._detector.detect(frame, tuple(self._labels))
        except DetectorUnavailable as exc:
            self._stop_reason = str(exc)
            self._stop_event.set()
            return None
        except Exception:  # noqa: BLE001 — one bad frame is not fatal
            _log.exception("detector raised — skipping frame")
            return self._blind_tick(now, "detector error on this frame")

        detections = self._tracker.update(filter_detections(raw, self._labels))
        target = self._pick(detections, info, now)

        posture = self._posture()
        command = self._controller.step(
            target, info,
            head_yaw=posture["head_yaw"], head_pitch=posture["head_pitch"],
            body_yaw=posture["body_yaw"], now=now)

        if target is not None:
            if (target.track_id != self._confirm_id or self._confirm_seen_at is None
                    or now - self._confirm_seen_at > self._config.hold_seconds):
                self._confirm_hits = 0
            self._confirm_id = target.track_id
            self._confirm_seen_at = now
            self._confirm_hits += 1
            if self._confirm_hits < max(2, self._config.lock_confirm_frames) and not command.done:
                # First sighting cancels all search motion. Observe again from
                # a settled camera before issuing an approach/turn command.
                command = FollowCommand(phase="acquiring", stop=True,
                                        note="target spotted — confirming lock")

        self._publish(command, detections, target, now, frame)
        return command

    # --- target lock -----------------------------------------------------

    def _pick(self, detections: list[Detection], info: FrameInfo,
              now: float) -> Detection | None:
        # -1 is the trackers library's "not confirmed yet" marker; every
        # other id is a real track. Note ids start at ZERO there, so this
        # must be `>= 0` — `> 0` would silently ignore the first object the
        # robot ever sees.
        confirmed = [d for d in detections
                     if d.track_id is not None and d.track_id >= 0]

        if self._locked_id is not None:
            for det in confirmed:
                if det.track_id == self._locked_id:
                    self._locked_lost_at = None
                    return det
            if self._locked_lost_at is None:
                self._locked_lost_at = now
            if now - self._locked_lost_at < self._relock_seconds:
                # Coast. This MUST apply from the very first missed frame:
                # an `elif` here would relock onto whatever else is in view
                # the instant one detection blinks — which at 8 Hz is a
                # routine motion-blur frame, and is exactly the "someone
                # walks past and steals the robot" failure this guard
                # exists to prevent.
                return None

        chosen = select_target(confirmed, info)
        if chosen is None:
            return None
        if chosen.track_id != self._locked_id:
            self._board.event(
                f"locked on {chosen.class_name or self._target_phrase} "
                f"#{chosen.track_id}")
            self._locked_id = chosen.track_id
        self._locked_lost_at = None
        return chosen

    # --- actuation -------------------------------------------------------

    def _posture(self) -> dict:
        if self._motion is None:
            return {"head_yaw": 0.0, "head_pitch": 0.0, "body_yaw": 0.0}
        # Never steer from fabricated zero angles when readback fails.
        # The session's exception handler halts the chassis instead.
        posture = self._motion.posture()
        return {"head_yaw": float(posture.get("head_yaw", 0.0)),
                "head_pitch": float(posture.get("head_pitch", 0.0)),
                "body_yaw": float(posture.get("body_yaw", 0.0))}

    def _apply_head(self, command: FollowCommand) -> None:
        """Both upper motion tiers, on one thread.

        Head and shell share the SDK's ``goto_target`` and its lock, so
        driving them from one worker keeps them serialised and keeps the
        ordering obvious: aim the head first (it is what the next frame is
        seen through), then swing the shell under it.
        """
        if self._motion is None:
            return
        posture = self._posture()
        if command.body_yaw_delta and hasattr(self._motion, "look_with_body"):
            yaw = command.head_yaw if command.head_yaw is not None else posture["head_yaw"]
            if command.phase == "searching":
                yaw += command.body_yaw_delta
            self._motion.look_with_body(
                yaw,
                command.head_pitch if command.head_pitch is not None else posture["head_pitch"],
                command.body_yaw_delta)
            return
        if command.head_yaw is not None or command.head_pitch is not None:
            self._motion.look(
                command.head_yaw if command.head_yaw is not None
                else posture["head_yaw"],
                command.head_pitch if command.head_pitch is not None
                else posture["head_pitch"],
            )
        if command.body_yaw_delta:
            self._motion.turn_body(command.body_yaw_delta)

    def _apply_wheels(self, command: FollowCommand) -> None:
        if command.stop or (self._was_moving and not command.moving):
            # Explicit halt on the moving → still edge: the deadman would
            # get there within `duration`, but a robot that keeps coasting
            # after its target stopped reads as broken.
            self._was_moving = False
            try:
                self._wheels_provider().stop()
            except WheelsError as exc:
                # The board may receive `/stop` but lose its acknowledgement
                # while its single-connection HTTP server or wlan0 is busy.
                # Every preceding move already has a short duration and the
                # board also has a 2 s deadman, so ending the whole tracking
                # session here is both unnecessary and what forced operators
                # to reset the app. Hold off new motion through the deadman,
                # report the degraded stop, and allow tracking to recover.
                self._wheel_hold_until = time.monotonic() + max(2.0, command.duration)
                self._board.event(
                    "Chassis stop acknowledgement missed; holding motion "
                    "while the hardware deadman stops the wheels")
                _log.warning("stop acknowledgement failed; relying on chassis "
                             "deadman before resuming: %s", exc)
            return
        if not command.moving:
            return
        if time.monotonic() < self._wheel_hold_until:
            return
        self._was_moving = True
        # The controller speaks in the robot's frame ("forward" = where the
        # camera looks); the board speaks in the chassis's. Rotate here, the
        # only place that knows how the robot is mounted.
        vx, vy = robot_to_chassis(command.vx, 0.0, self._mount_yaw)
        self._wheels_provider().move(
            vx=vx, vy=vy, omega=command.omega,
            speed=command.speed, duration=command.duration)

    def _halt(self) -> None:
        try:
            self._wheels_provider().stop()
        except Exception:  # noqa: BLE001 — board deadman is the backstop
            _log.warning("final stop failed; relying on the chassis deadman")

    # --- reporting -------------------------------------------------------

    def _publish(self, command: FollowCommand, detections: list[Detection],
                 target: Detection | None, now: float, frame=None) -> None:
        self._board.set_status(
            phase=command.phase, detail=command.note,
            moving=command.moving, fps=round(self._fps, 1),
            commanded_vx=command.vx, commanded_omega=command.omega,
            commanded_body_delta=command.body_yaw_delta,
            seen=len(detections), track_id=self._locked_id,
            lost_for=0.0 if target is not None
            else round(self._controller.seconds_since_seen(now), 1),
            # Explicit None when the target is not in view: merging only the
            # present keys would leave the UI showing the last known bearing
            # and distance as if they were current.
            **{k: getattr(command, k) for k in
               ("bearing_deg", "yaw_error_deg", "size", "distance_m",
                "search_covered_deg")},
            **self._posture(),
        )
        if command.note:
            self._board.event(command.note)
        if frame is not None and now - self._last_preview >= self._preview_period:
            # Annotate the frame these detections actually came from — a
            # freshly grabbed one would show boxes a frame or two stale.
            self._last_preview = now
            try:
                self._board.set_preview(
                    annotate(frame, detections, self._locked_id, command.phase))
            except Exception:  # noqa: BLE001 — the preview is cosmetic; a
                # missing encoder must not end a follow that is working.
                _log.exception("preview render failed — continuing without it")


class FollowManager:
    """Owns at most one session, and the (expensive to load) detector."""

    def __init__(self, *, frame_provider, wheels_provider, motion=None,
                 state_provider=None):
        self._frames = frame_provider
        # Callable for the same reason as in FollowSession: the link swaps
        # its client out from under us on every settings save.
        self._wheels = wheels_provider
        self._motion = motion
        self._state = state_provider or (lambda: {})
        # Two locks with different jobs. `_lock` guards the session/detector
        # fields and is only ever held for non-blocking work — a finishing
        # session takes it from its own thread to deregister itself, so
        # anything slow held under it deadlocks against that. `_transition`
        # serialises whole start/stop operations (which DO block, joining a
        # thread) without being in that path.
        self._transition = threading.Lock()
        self._lock = threading.Lock()
        self._session: FollowSession | None = None
        self._detector = None
        self._detector_key: tuple | None = None
        self.board = TrackBoard()

    # --- detector cache --------------------------------------------------

    def _detector_for(self, state: dict) -> tuple:
        """``(detector, displaced_or_None)`` for these settings.

        Order matters: build first, swap second, close last. Closing the
        old one up front would, on a build failure (switching to `remote`
        with no URL set — the default), leave a *running* session calling
        into a closed ONNX session and leave the dead detector cached under
        the old key for every future follow. Caller holds ``_transition``,
        so two concurrent starts cannot both build.
        """
        key = (state.get("track_detector"), state.get("track_model_path"),
               state.get("track_remote_url"), state.get("track_imgsz"),
               state.get("track_min_score"))
        if self._detector is not None and key == self._detector_key:
            return self._detector, None

        fresh = build_detector(state)      # may raise DetectorUnavailable
        previous = self._detector
        self._detector, self._detector_key = fresh, key
        # The displaced one goes back to the caller rather than being closed
        # here: a session may still be mid-frame inside it.
        return fresh, previous

    # --- API -------------------------------------------------------------

    def start(self, target: str, *, distance_cm: float | None = None) -> dict:
        state = dict(self._state())
        phrase = str(target or "").strip()
        if not phrase:
            return {"status": "error", "error": "say what to follow"}

        # Everything below runs under _transition: the detector swap and the
        # session handover have to be one atomic operation, or a second
        # caller can close a detector the first one is about to use.
        self._transition.acquire()
        try:
            return self._start_locked(phrase, state, distance_cm)
        finally:
            self._transition.release()

    def _start_locked(self, phrase: str, state: dict,
                      distance_cm: float | None) -> dict:
        # A detector that cannot be built must not disturb a running follow,
        # so this happens before anything is torn down.
        try:
            detector, displaced = self._detector_for(state)
        except DetectorUnavailable as exc:
            self.board.event(str(exc))
            return {"status": "error", "error": str(exc)}

        vocabulary = tuple(detector.vocabulary) or ()
        if vocabulary:
            labels, ok = resolve_target(phrase, vocabulary)
            if not ok:
                return {
                    "status": "error",
                    "error": (f"I don't have a detector class for “{phrase}”. "
                              "This backend knows COCO objects (person, dog, "
                              "cat, bottle, cup, chair, sports ball, …); "
                              "switch track_detector to \"remote\" for "
                              "arbitrary things."),
                }
        else:
            labels = {phrase.lower()}   # open vocabulary: the phrase IS the class

        config = follow_config_from_state(state)
        hfov, vfov = config_hfov(state), config_vfov(state)
        # No explicit request? Fall back to the configured follow distance
        # in metres, which means the same thing whatever lens is fitted.
        if not (distance_cm and distance_cm > 0) and config.follow_distance_m > 0:
            distance_cm = config.follow_distance_m * 100.0
        if distance_cm and distance_cm > 0:
            # "follow me at about a metre" → the apparent size that distance
            # would produce for this class. Only lands when we have a height
            # prior; otherwise the configured default size stands.
            # Prefer a label we hold a height prior for; alphabetical order
            # would otherwise decide the scale of the follow distance.
            priced = sorted(l for l in labels if known_height_m(l))
            size = config.size_for_distance(
                (priced or sorted(labels))[0], distance_cm / 100.0,
                FrameInfo(width=1, height=1, hfov_deg=hfov, vfov_deg=vfov))
            if size is not None:
                config = replace_config(config, target_size=size)

        self._end_current("switching target")
        if displaced is not None:
            # Safe now: whatever was using it has been joined.
            try:
                displaced.close()
            except Exception:  # noqa: BLE001
                pass

        session = FollowSession(
            frame_provider=self._frames, wheels_provider=self._wheels,
            motion=self._motion, detector=detector, board=self.board,
            config=config, target_phrase=phrase, labels=set(labels),
            tracker=TargetTracker(
                algorithm=str(state.get("track_algorithm") or "sort"),
                frame_rate=float(state.get("track_rate_hz", 8.0)),
                minimum_iou_threshold=float(state.get("track_iou", 0.25))),
            rate_hz=float(state.get("track_rate_hz", 8.0)),
            hfov_deg=hfov, vfov_deg=vfov,
            mount_yaw_deg=float(state.get("track_mount_yaw_deg", 0.0)),
            on_finish=self._forget,
        )
        with self._lock:
            self._session = session
        session.start()

        return {"status": "ok", "action": f"following {phrase}",
                "target": phrase, "labels": sorted(labels),
                "detector": detector.name, "tracker": session.tracker_backend,
                "follow_size": round(config.target_size, 2)}

    def stop(self, reason: str = "stopped by request") -> dict:
        with self._transition:
            session, clean = self._end_current(reason)
        if session is None:
            # Still halt the base: the caller said stop and may be reacting
            # to something we cannot see.
            try:
                self._wheels().stop()
            except Exception:  # noqa: BLE001
                pass
            return {"status": "ok", "action": "not following", "was_active": False}
        if not clean:
            return {"status": "ok", "action": "stopping",
                    "target": session.target_phrase, "was_active": True,
                    "note": "the session is still shutting down (the chassis "
                            "has been told to stop and its deadman bounds "
                            "any motion regardless)"}
        return {"status": "ok", "action": "stopped following",
                "target": session.target_phrase, "was_active": True}

    def _end_current(self, reason: str) -> tuple:
        """``(session_or_None, stopped_cleanly)``. Caller holds ``_transition``.

        Deliberately does NOT hold ``_lock`` across the join: the session
        deregisters itself from its own thread under that lock, so holding
        it here would block until the join timed out — seconds of a robot
        still following the previous target.

        The join can genuinely time out: teardown includes a final
        ``WheelsClient.stop()``, which retries for ~9 s against a busy or
        absent board. We say so rather than claiming success, and mark the
        session superseded so its late teardown cannot stamp "stopped" over
        whatever replaced it.
        """
        with self._lock:
            session = self._session
        if session is None or not session.is_alive():
            return None, True
        session.request_stop(reason)
        session.join(timeout=4.0)
        if session.is_alive():
            session.supersede()
            _log.warning("follow session did not stop within 4s — superseding "
                         "(the chassis deadman still bounds any motion)")
            return session, False
        return session, True

    def status(self) -> dict:
        with self._lock:
            session = self._session
        snapshot = self.board.snapshot()
        snapshot["active"] = bool(session is not None and session.is_alive())
        return snapshot

    def preview(self) -> bytes:
        return self.board.preview()

    def probe(self, target: str = "") -> dict:
        """One-shot: grab a frame, detect, report everything seen.

        The operator's answer to "why isn't it following anything?" — it
        exercises the exact frame source and detector instance the follow
        loop uses, so it distinguishes "no camera", "detector sees nothing",
        and "sees things but not the class you asked for".
        """
        state = dict(self._state())
        frame = self._frames()
        if frame is None:
            return {"status": "error", "error": "no camera frame available"}
        try:
            shape = list(frame.shape)
            mean = float(frame.mean())
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"unusable frame: {exc!r}"}

        with self._transition:
            try:
                detector, displaced = self._detector_for(state)
            except DetectorUnavailable as exc:
                return {"status": "error", "error": str(exc)}
            if displaced is not None:
                try:
                    displaced.close()
                except Exception:  # noqa: BLE001
                    pass

        labels: set[str] = set()
        if target.strip():
            vocabulary = tuple(detector.vocabulary) or ()
            labels = resolve_target(target, vocabulary)[0] if vocabulary \
                else {target.strip().lower()}

        started = time.monotonic()
        try:
            found = detector.detect(frame, tuple(labels))
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "frame": shape,
                    "error": f"detector raised: {exc!r}"}
        took_ms = round((time.monotonic() - started) * 1000, 1)

        unfiltered = []
        if not labels:
            unfiltered = found
        else:
            try:
                unfiltered = detector.detect(frame, ())
            except Exception:  # noqa: BLE001
                pass

        return {
            "status": "ok",
            "frame": {"shape": shape, "mean_pixel": round(mean, 1),
                      "dtype": str(getattr(frame, "dtype", "?"))},
            "detector": getattr(detector, "name", "?"),
            "detect_ms": took_ms,
            "asked_for": sorted(labels),
            "matching": [d.to_dict() for d in found],
            "everything_seen": [
                {"class_name": d.class_name, "score": round(d.score, 3)}
                for d in sorted(unfiltered, key=lambda d: -d.score)[:12]],
        }

    def shutdown(self) -> None:
        self.stop("app shutting down")
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:  # noqa: BLE001
                pass

    def _forget(self, session: FollowSession) -> None:
        with self._lock:
            if self._session is session:
                self._session = None


# --- config plumbing -----------------------------------------------------

def config_hfov(state: dict) -> float:
    return float(state.get("track_hfov_deg", 70.0))


def config_vfov(state: dict) -> float:
    return float(state.get("track_vfov_deg", 55.0))


def replace_config(config: FollowConfig, **updates) -> FollowConfig:
    import dataclasses
    return dataclasses.replace(config, **updates)


def follow_config_from_state(state: dict) -> FollowConfig:
    """Build a FollowConfig from the persisted settings, ignoring junk keys.

    Each field is read from ``track_<field>`` and coerced to the default's
    type; anything missing, unrecognised or uncoercible keeps the default,
    so a hand-edited settings file can't stop the robot from following.
    """
    import dataclasses

    defaults = FollowConfig()
    overrides = {}
    for field in dataclasses.fields(FollowConfig):
        value = state.get(f"track_{field.name}")
        if value is None:
            continue
        try:
            wanted = type(getattr(defaults, field.name))
            # bool("false") is True; only real booleans may set a flag.
            overrides[field.name] = (bool(value) if wanted is bool
                                     and isinstance(value, bool)
                                     else wanted(value))
        except (TypeError, ValueError):
            continue
    return FollowConfig(**overrides)


__all__ = ["FollowManager", "FollowSession", "TrackBoard", "select_target",
           "follow_config_from_state", "COCO_CLASSES"]
