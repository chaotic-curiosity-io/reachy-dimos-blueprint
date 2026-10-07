"""The follow session end to end, with fakes for camera, chassis and detector.

Threads are real here — the point is to prove the loop actually commands the
chassis, keeps its lock, and stops when told, without needing a robot.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from reachy_wheels_app.tracking.detectors import DetectorUnavailable
from reachy_wheels_app.tracking.session import (
    FollowManager,
    FollowSession,
    _LatestSlot,
    TrackBoard,
    follow_config_from_state,
    select_target,
)
from reachy_wheels_app.tracking.follow import FollowCommand
from reachy_wheels_app.wheels_client import WheelsError
from reachy_wheels_app.tracking.types import Detection, FrameInfo

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def box(cx, cy=240, w=100, h=200, name="person", score=0.9):
    return Detection(bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                     score=score, class_name=name)


class FakeWheels:
    def __init__(self):
        self.lock = threading.Lock()
        self.moves: list[dict] = []
        self.stops = 0

    def move(self, vx=0.0, vy=0.0, omega=0.0, speed=None, duration=None):
        with self.lock:
            self.moves.append({"vx": vx, "vy": vy, "omega": omega,
                               "speed": speed, "duration": duration})
        return {"ok": True}

    def stop(self):
        with self.lock:
            self.stops += 1
        return {"ok": True}


def test_missed_stop_ack_uses_deadman_without_ending_follow(monkeypatch):
    class MissedAck(FakeWheels):
        def stop(self):
            raise WheelsError("/stop: timed out")

    wheels = MissedAck()
    session = FollowSession.__new__(FollowSession)
    session._wheels_provider = lambda: wheels
    session._board = TrackBoard()
    session._was_moving = True
    session._wheel_hold_until = 0.0
    session._mount_yaw = 0.0
    now = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: now)

    session._apply_wheels(FollowCommand("arrived", stop=True))
    assert session._was_moving is False
    assert session._wheel_hold_until == pytest.approx(102.0)
    assert "acknowledgement missed" in session._board.snapshot()["events"][-1]["text"]

    session._apply_wheels(FollowCommand("following", vx=0.5))
    assert wheels.moves == []
    now = 102.1
    session._apply_wheels(FollowCommand("following", vx=0.5))
    assert len(wheels.moves) == 1

class FakeMotion:
    def __init__(self):
        self.lock = threading.Lock()
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.body_yaw = 0.0
        self.centered = 0
        self.looks: list[tuple[float, float]] = []

    def posture(self):
        with self.lock:
            return {"head_yaw": self.head_yaw, "head_pitch": self.head_pitch,
                    "body_yaw": self.body_yaw}

    def look(self, yaw, pitch):
        with self.lock:
            self.head_yaw, self.head_pitch = float(yaw), float(pitch)
            self.looks.append((self.head_yaw, self.head_pitch))
        return {"status": "ok"}

    def center(self):
        with self.lock:
            self.centered += 1
            self.head_yaw = self.head_pitch = self.body_yaw = 0.0
        return {"status": "ok"}


class ScriptedDetector:
    """Returns a fixed set of boxes every frame; counts calls."""

    name = "scripted"

    def __init__(self, boxes, vocabulary=("person", "dog", "cup")):
        self._boxes = boxes
        self._vocabulary = vocabulary
        self.calls = 0
        self.last_targets = None

    @property
    def vocabulary(self):
        return self._vocabulary

    def detect(self, frame, targets=()):
        self.calls += 1
        self.last_targets = tuple(targets)
        return list(self._boxes)

    def close(self):
        return None


def manager(detector, wheels=None, motion=None, **state):
    base = {"track_detector": "scripted", "track_rate_hz": 20.0,
            "track_hfov_deg": 70.0, "track_vfov_deg": 55.0, **state}
    fallback = FakeWheels()
    mgr = FollowManager(frame_provider=lambda: FRAME,
                        wheels_provider=lambda: wheels or fallback,
                        motion=motion, state_provider=lambda: base)
    # Stand in for the backend factory: same contract as the real one,
    # (detector, displaced-detector-to-close).
    mgr._detector = detector
    mgr._detector_key = ("scripted",)
    mgr._detector_for = lambda state: (detector, None)
    return mgr


def wait_for(predicate, timeout=4.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- target selection -----------------------------------------------------

def test_selection_prefers_the_big_central_object():
    info = FrameInfo(width=640, height=480)
    small_centre = box(320, w=40, h=60).with_track(1)
    big_centre = box(320, w=200, h=300).with_track(2)
    big_edge = box(30, w=200, h=300).with_track(3)
    assert select_target([small_centre, big_centre, big_edge], info) is big_centre


def test_track_id_zero_is_a_real_track_not_an_unconfirmed_one():
    """Roboflow trackers number tracks from ZERO; only -1 means unconfirmed.

    Filtering on ``> 0`` would make the robot ignore the very first object
    it ever sees — invisible under the fallback tracker, which starts at 1.
    """
    session = FollowSession(
        frame_provider=lambda: FRAME, wheels_provider=FakeWheels, motion=None,
        detector=ScriptedDetector([]), board=TrackBoard(),
        config=follow_config_from_state({}), target_phrase="person",
        labels={"person"})
    info = FrameInfo(width=640, height=480)
    picked = session._pick([box(320).with_track(0)], info, now=1.0)
    assert picked is not None and picked.track_id == 0


def test_unconfirmed_tracks_are_never_locked_onto():
    session = FollowSession(
        frame_provider=lambda: FRAME, wheels_provider=FakeWheels, motion=None,
        detector=ScriptedDetector([]), board=TrackBoard(),
        config=follow_config_from_state({}), target_phrase="person",
        labels={"person"})
    info = FrameInfo(width=640, height=480)
    assert session._pick([box(320).with_track(-1)], info, now=1.0) is None


def test_selection_of_nothing_is_none():
    assert select_target([], FrameInfo(width=640, height=480)) is None


# --- lifecycle ------------------------------------------------------------

def test_start_refuses_an_empty_target():
    mgr = manager(ScriptedDetector([box(320)]))
    assert mgr.start("")["status"] == "error"


def test_start_refuses_a_target_the_backend_cannot_detect():
    mgr = manager(ScriptedDetector([box(320)]))
    out = mgr.start("a unicorn")
    assert out["status"] == "error"
    assert "unicorn" in out["error"]
    assert "remote" in out["error"]        # tells the user the way forward


def test_open_vocabulary_backend_accepts_any_phrase():
    detector = ScriptedDetector([box(320, name="red mug")], vocabulary=())
    mgr = manager(detector)
    try:
        out = mgr.start("red mug")
        assert out["status"] == "ok"
        assert out["labels"] == ["red mug"]
    finally:
        mgr.stop()


def test_a_detector_that_cannot_start_reports_instead_of_driving():
    mgr = FollowManager(frame_provider=lambda: FRAME,
                        wheels_provider=FakeWheels,
                        state_provider=lambda: {"track_detector": "remote",
                                                "track_remote_url": ""})
    out = mgr.start("person")
    assert out["status"] == "error" and "track_remote_url" in out["error"]


def test_following_drives_the_chassis_and_tracks_with_the_head():
    wheels, motion = FakeWheels(), FakeMotion()
    # Small box off to the right: the robot should turn right and roll on.
    mgr = manager(ScriptedDetector([box(600, h=100)]), wheels, motion)
    try:
        assert mgr.start("person")["status"] == "ok"
        assert wait_for(lambda: len(wheels.moves) >= 2), "no chassis commands"
        assert motion.centered == 0, (
            "must NOT recentre by default — it throws a target that is "
            "already in view straight out of frame before the loop sees it")
        assert wait_for(lambda: motion.looks), "head never tracked"
        assert motion.looks[0][0] < 0, "head should turn right"
        assert wheels.moves[0]["omega"] < 0, "base should rotate right"
        assert all(m["duration"] and m["duration"] > 0 for m in wheels.moves), \
            "every command must arm the board deadman"
    finally:
        mgr.stop()


def test_centring_before_following_is_available_but_opt_in():
    motion = FakeMotion()
    mgr = manager(ScriptedDetector([box(320, h=100)]), motion=motion,
                  track_center_on_start=True)
    try:
        mgr.start("person")
        assert wait_for(lambda: motion.centered == 1)
    finally:
        mgr.stop()


def test_the_detector_is_told_what_to_look_for():
    detector = ScriptedDetector([box(320)])
    mgr = manager(detector)
    try:
        mgr.start("the doggy")
        assert wait_for(lambda: detector.last_targets is not None)
        assert detector.last_targets == ("dog",)
    finally:
        mgr.stop()


def test_stop_halts_the_chassis_and_ends_the_session():
    wheels = FakeWheels()
    mgr = manager(ScriptedDetector([box(600, h=100)]), wheels)
    mgr.start("person")
    assert wait_for(lambda: wheels.moves)
    out = mgr.stop()
    assert out["was_active"] is True
    assert wheels.stops >= 1
    assert mgr.status()["active"] is False
    before = len(wheels.moves)
    time.sleep(0.2)
    assert len(wheels.moves) == before, "session kept driving after stop"


def test_stop_when_idle_still_halts_the_base():
    wheels = FakeWheels()
    mgr = manager(ScriptedDetector([]), wheels)
    out = mgr.stop()
    assert out["was_active"] is False
    assert wheels.stops == 1


def test_starting_a_second_follow_replaces_the_first():
    wheels = FakeWheels()
    mgr = manager(ScriptedDetector([box(320), box(500, name="dog")]), wheels)
    try:
        mgr.start("person")
        assert wait_for(lambda: mgr.status().get("target") == "person")
        assert mgr.start("dog")["status"] == "ok"
        assert wait_for(lambda: mgr.status().get("target") == "dog")
        assert mgr.status()["active"] is True
    finally:
        mgr.stop()


def test_switching_targets_is_prompt():
    """Regression: `start` used to hold the lock a finishing session needs
    to deregister itself, so every target switch blocked for the full join
    timeout — three seconds of the robot still chasing the old target."""
    mgr = manager(ScriptedDetector([box(320), box(500, name="dog")]))
    try:
        mgr.start("person")
        assert wait_for(lambda: mgr.status().get("target") == "person")
        started = time.monotonic()
        mgr.start("dog")
        assert time.monotonic() - started < 1.5
    finally:
        mgr.stop()


def test_status_reports_the_backends_in_use():
    mgr = manager(ScriptedDetector([box(320)]))
    try:
        out = mgr.start("person")
        assert out["detector"] == "scripted"
        assert out["tracker"]
        assert wait_for(lambda: mgr.status().get("phase") in
                        ("acquiring", "following", "arrived"))
    finally:
        mgr.stop()


def test_a_never_seen_target_ends_the_session_on_its_own():
    wheels = FakeWheels()
    mgr = manager(ScriptedDetector([]), wheels, track_hold_seconds=0.1,
                  track_search_seconds=0.2, track_acquire_seconds=0.2)
    mgr.start("person")
    assert wait_for(lambda: mgr.status()["active"] is False, timeout=5.0)
    assert wheels.stops >= 1
    assert "stopped" in mgr.status()["detail"].lower() or \
        "never saw" in mgr.status()["detail"]


def test_shutdown_stops_everything():
    wheels = FakeWheels()
    mgr = manager(ScriptedDetector([box(320, h=100)]), wheels)
    mgr.start("person")
    assert wait_for(lambda: wheels.moves)
    mgr.shutdown()
    assert mgr.status()["active"] is False


# --- config plumbing ------------------------------------------------------

def test_config_overrides_come_from_track_prefixed_settings():
    config = follow_config_from_state({
        "track_target_size": 0.3, "track_drive_speed": 0.9,
        "track_max_session_seconds": 30.0,
        "default_speed": 0.8,             # not ours; must be ignored
        "track_nonsense": "junk",         # not a field; must be ignored
    })
    assert config.target_size == pytest.approx(0.3)
    assert config.drive_speed == pytest.approx(0.9)
    assert config.max_session_seconds == pytest.approx(30.0)


def test_unparseable_overrides_fall_back_to_the_default():
    assert follow_config_from_state({"track_target_size": "wide"}).target_size \
        == pytest.approx(0.55)


def test_distance_request_tightens_the_follow_size():
    detector = ScriptedDetector([box(320)])
    mgr = manager(detector)
    try:
        far = mgr.start("person", distance_cm=300)["follow_size"]
        mgr.stop()
        near = mgr.start("person", distance_cm=80)["follow_size"]
        assert near > far
    finally:
        mgr.stop()


# --- the status board -----------------------------------------------------

def test_board_collapses_repeated_events():
    board = TrackBoard()
    board.event("lost the target")
    board.event("lost the target")
    board.event("found it")
    assert [e["text"] for e in board.snapshot()["events"]] == \
        ["lost the target", "found it"]


def test_board_ignores_empty_events_and_previews():
    board = TrackBoard()
    board.event("   ")
    board.set_preview(b"")
    assert board.snapshot()["events"] == []
    assert board.preview() == b""


# --- regressions from the review pass ------------------------------------

def _session(detector, wheels=None, **kw):
    return FollowSession(
        frame_provider=lambda: FRAME, wheels_provider=lambda: wheels or FakeWheels(),
        motion=None, detector=detector, board=TrackBoard(),
        config=follow_config_from_state({}), target_phrase="person",
        labels={"person"}, **kw)


class IdentityTracker:
    backend = "test"

    def update(self, detections):
        return detections


def test_search_stops_on_sighting_then_approaches_after_confirmation():
    detector = ScriptedDetector([])
    session = _session(detector, tracker=IdentityTracker())
    searching = session._tick(10)
    assert searching.head_pitch is not None and not searching.moving
    detector._boxes = [box(320, h=100).with_track(7)]
    first = session._tick(10.2)
    assert first.stop and not first.moving and first.head_yaw is None
    second = session._tick(10.6)
    assert second.vx > 0 and second.phase == "following"
    detector._boxes = []
    missed = session._tick(11.8)
    assert not missed.moving and missed.phase == "holding"
    detector._boxes = [box(330, h=100).with_track(7)]
    assert session._tick(12).vx > 0


def test_reappearing_target_after_long_gap_requires_confirmation_again():
    detector = ScriptedDetector([box(320, h=100).with_track(7)])
    session = _session(detector, tracker=IdentityTracker())
    session._tick(1)
    assert session._tick(1.4).vx > 0
    assert session._tick(5).stop


def test_no_camera_never_commands_a_blind_search():
    session = _session(ScriptedDetector([]))
    session._frames = lambda: None
    cmd = session._tick(10)
    assert cmd.stop and not cmd.moving
    assert cmd.head_yaw is None and cmd.body_yaw_delta is None
    assert session._tick(300).done


def test_replacement_track_cannot_inherit_the_previous_targets_confirmation():
    detector = ScriptedDetector([box(320, h=100).with_track(7)])
    session = _session(detector, tracker=IdentityTracker(), relock_seconds=0.5)
    session._tick(1)
    assert session._tick(1.4).vx > 0
    detector._boxes = [box(320, h=100).with_track(9)]
    assert not session._tick(1.5).moving
    assert session._tick(2.1).stop
    assert session._tick(2.5).vx > 0


def test_one_blinked_frame_does_not_hand_the_robot_to_a_passer_by():
    """The guard used to be an `elif`, so the FIRST missed frame fell
    straight through to reselection — meaning a single motion-blur frame
    (routine at 8 Hz) let whoever else was in view steal the robot."""
    session = _session(ScriptedDetector([]), relock_seconds=1.5)
    info = FrameInfo(width=640, height=480)

    mine = box(320).with_track(7)
    assert session._pick([mine], info, now=1.0).track_id == 7

    # My track blinks; a stranger is in frame and is bigger and more central.
    stranger = box(320, w=300, h=400).with_track(9)
    assert session._pick([stranger], info, now=1.1) is None, \
        "relocked onto another track on the first missed frame"
    assert session._pick([stranger], info, now=1.5) is None

    # I come back — still mine.
    assert session._pick([mine, stranger], info, now=1.9).track_id == 7


def test_a_target_gone_for_good_does_eventually_release_the_lock():
    session = _session(ScriptedDetector([]), relock_seconds=0.5)
    info = FrameInfo(width=640, height=480)
    session._pick([box(320).with_track(7)], info, now=1.0)
    other = box(320).with_track(9)
    assert session._pick([other], info, now=1.1) is None
    assert session._pick([other], info, now=3.0).track_id == 9


def test_closing_an_actuator_discards_a_queued_command():
    """A drive submitted just before STOP must be dropped, not applied on
    the way out — that would be motion after the user said stop."""
    applied, gate = [], threading.Event()

    def slow_apply(item):
        applied.append(item)
        gate.wait(2.0)

    slot = _LatestSlot("test-slot", slow_apply)
    slot.submit("first")
    assert wait_for(lambda: applied == ["first"])
    slot.submit("second")          # queued behind the in-flight one
    gate.set()
    slot.close()
    assert applied == ["first"], f"stale command applied on close: {applied}"


def test_a_camera_that_never_yields_a_frame_ends_the_session():
    """The give-up ladder and the session time limit live in the
    controller, so a tick that returns before calling it would spin an
    'active' session forever on a dead camera."""
    wheels = FakeWheels()
    mgr = FollowManager(frame_provider=lambda: None,
                        wheels_provider=lambda: wheels,
                        state_provider=lambda: {
                            "track_detector": "scripted", "track_rate_hz": 20.0,
                            "track_hold_seconds": 0.1,
                            "track_search_seconds": 0.2,
                            "track_acquire_seconds": 0.2})
    mgr._detector_for = lambda state: (ScriptedDetector([]), None)
    mgr.start("person")
    assert wait_for(lambda: mgr.status()["active"] is False, timeout=5.0), \
        "session never gave up with no camera frames"


def test_a_detector_that_raises_every_frame_also_ends_the_session():
    class Exploding(ScriptedDetector):
        def detect(self, frame, targets=()):
            raise RuntimeError("model exploded")

    wheels = FakeWheels()
    mgr = manager(Exploding([]), wheels, track_hold_seconds=0.1,
                  track_search_seconds=0.2, track_acquire_seconds=0.2)
    mgr.start("person")
    assert wait_for(lambda: mgr.status()["active"] is False, timeout=5.0)


def test_a_failed_detector_swap_leaves_the_running_follow_alone():
    """Switching to `remote` with no URL used to close the live detector
    before discovering the new one could not be built."""
    detector = ScriptedDetector([box(320, h=100)])
    state = {"track_detector": "scripted", "track_rate_hz": 20.0}
    wheels = FakeWheels()
    mgr = FollowManager(frame_provider=lambda: FRAME,
                        wheels_provider=lambda: wheels,
                        state_provider=lambda: state)
    real_build = mgr._detector_for
    mgr._detector_for = lambda s: (detector, None)
    try:
        assert mgr.start("person")["status"] == "ok"
        assert wait_for(lambda: wheels.moves)

        # Now a settings change whose backend cannot be constructed.
        mgr._detector_for = real_build
        state.update({"track_detector": "remote", "track_remote_url": ""})
        assert mgr.start("person")["status"] == "error"

        # The original session is untouched and still driving.
        assert mgr.status()["active"] is True
        before = len(wheels.moves)
        assert wait_for(lambda: len(wheels.moves) > before), \
            "the running follow went blind after a failed detector swap"
    finally:
        mgr.stop()


def test_the_chassis_client_is_resolved_per_command():
    """WheelsLink swaps its client on every settings save; a session that
    captured the object would keep commanding the old chassis address."""
    first, second = FakeWheels(), FakeWheels()
    current = [first]
    mgr = FollowManager(frame_provider=lambda: FRAME,
                        wheels_provider=lambda: current[0],
                        state_provider=lambda: {"track_detector": "scripted",
                                                "track_rate_hz": 20.0})
    mgr._detector_for = lambda s: (ScriptedDetector([box(600, h=100)]), None)
    try:
        mgr.start("person")
        assert wait_for(lambda: first.moves)
        current[0] = second              # user edits the chassis host
        assert wait_for(lambda: second.moves), "kept using the old client"
    finally:
        mgr.stop()


def test_follow_distance_defaults_to_metres_not_a_frame_fraction():
    """A frame fraction means whatever the lens says. On the measured 42°
    vertical FOV, size 0.55 put the hold-station point four metres back and
    the robot reversed away from the user."""
    detector = ScriptedDetector([box(320)])
    mgr = manager(detector, track_vfov_deg=42.3, track_follow_distance_m=1.5)
    try:
        near = mgr.start("person")["follow_size"]
        mgr.stop()
        mgr2 = manager(ScriptedDetector([box(320)]),
                       track_vfov_deg=42.3, track_follow_distance_m=3.0)
        far = mgr2.start("person")["follow_size"]
        mgr2.stop()
        assert near > far, "closer follow distance must mean a bigger target"
        assert near > 0.55, "1.5 m should be nearer than the old default"
    finally:
        mgr.stop()


def test_an_explicit_request_still_beats_the_configured_distance():
    mgr = manager(ScriptedDetector([box(320)]),
                  track_vfov_deg=42.3, track_follow_distance_m=1.5)
    try:
        asked = mgr.start("person", distance_cm=400)["follow_size"]
        mgr.stop()
        default = mgr.start("person")["follow_size"]
        assert asked < default
    finally:
        mgr.stop()


def test_the_mounting_offset_reaches_the_chassis():
    """The robot is bolted 90° round on the chassis; a follow that drives
    'forward' must come out of the board as a strafe."""
    wheels = FakeWheels()
    mgr = manager(ScriptedDetector([box(320, h=60)]), wheels,
                  track_mount_yaw_deg=-90.0)
    try:
        mgr.start("person")
        assert wait_for(lambda: any(abs(m["vy"]) > 0.05 for m in wheels.moves)), \
            "no lateral component — the mounting offset was ignored"
        driving = [m for m in wheels.moves if abs(m["vy"]) > 0.05]
        assert all(abs(m["vx"]) < 1e-6 for m in driving), \
            "a 90-degree mount should put ALL translation on vy"
        assert driving[0]["vy"] < 0, "robot-forward must be a RIGHT strafe"
    finally:
        mgr.stop()
