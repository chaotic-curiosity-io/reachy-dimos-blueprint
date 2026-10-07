"""HTTP surface of the wheels app, wired onto a FastAPI instance.

Kept free of any ``reachy_mini`` import so the whole surface is testable
offline with a fake WheelsClient; ``main.py`` mounts it onto the SDK's
``settings_app`` (which also serves ``static/`` — the D-pad UI).
"""

from __future__ import annotations

import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from . import config as config_store
from . import mount as mount_mod
from . import wheel_lab
from .sensors import DEFAULT_DEPTH_URL, DEPTH_FRAME, RGB_FRAME, DepthSource, validate_url
from .tracking.detectors import BACKENDS
from .tracking.vocab import COCO_CLASSES
from .wheels_client import PRIMITIVES, WheelsClient, WheelsConfig, WheelsError

# Commands the UI may forward to the board, beyond the named primitives.
_EXTRA_COMMANDS = {"move", "stop", "brake"}


class WheelsLink:
    """Owns the client and rebuilds it when the host/token settings change."""

    # UI telemetry polls tolerate slightly stale data; drive commands must
    # not fight them for the board's single connection.
    STATE_CACHE_TTL = 1.0

    def __init__(self, state: dict | None = None):
        self._lock = threading.Lock()
        self.state = state if state is not None else config_store.load()
        self._state_cache: tuple[float, dict] | None = None
        self._rebuild()

    def _rebuild(self) -> None:
        self.client = WheelsClient(WheelsConfig(
            host=str(self.state["host"]),
            port=int(self.state["port"]),
            token=str(self.state.get("token", "")),
        ))

    # Settings that change where/how we talk to the board; everything else
    # is app-side and must not disturb a live client.
    TRANSPORT_KEYS = ("host", "port", "token")

    def reconfigure(self, **updates) -> dict:
        with self._lock:
            changed = {k: v for k, v in updates.items()
                       if k not in self.state or self.state[k] != v}
            if not changed:
                # A held wheel-lab mix re-posts the same settings several
                # times a second; re-saving them would hammer the flash for
                # nothing. It also keeps an injected test client alive.
                return dict(self.state)
            self.state.update(changed)
            if any(k in changed for k in self.TRANSPORT_KEYS):
                self._rebuild()
            config_store.save(self.state)
            return dict(self.state)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def board_state_cached(self) -> dict:
        """Board /state for telemetry consumers, at most once per TTL.

        The ESP32 serves one connection at a time; without this, every open
        browser tab's 2 s status poll competes with voice/drive commands and
        shows up as command latency. Only successes are cached — an
        unreachable board should keep being probed (the client's own retry
        pacing bounds that).
        """
        now = time.monotonic()
        with self._lock:
            cached = self._state_cache
        if cached is not None and now - cached[0] < self.STATE_CACHE_TTL:
            return cached[1]
        state = self.client.state()  # blocking HTTP — outside the lock
        with self._lock:
            self._state_cache = (time.monotonic(), state)
        return state


class CmdRequest(BaseModel):
    command: str
    speed: float | None = None
    duration: float | None = None
    vx: float | None = None
    vy: float | None = None
    omega: float | None = None


class WheelMixRequest(BaseModel):
    """One hand-built per-wheel mix, as driven from the wheel lab."""

    speeds: dict[str, float] = {}
    speed: float | None = None
    duration: float | None = None
    remember: bool = True       # keep it across a page reload


class WheelTuneRequest(BaseModel):
    wheel: str
    trim: float | None = None
    invert: bool | None = None


class ConfigPatch(BaseModel):
    host: str | None = None
    port: int | None = None
    token: str | None = None
    default_speed: float | None = None


class SensorConfigPatch(BaseModel):
    depth_url: str


class VoiceConfigPatch(BaseModel):
    gemini_api_key: str | None = None
    gemini_voice: str | None = None
    gemini_language: str | None = None
    gemini_model: str | None = None
    voice_enabled: bool | None = None
    enable_video: bool | None = None
    speak_responses: bool | None = None
    max_drive_seconds: float | None = None
    cal_cm_per_s: float | None = None
    cal_deg_per_s: float | None = None


class TrackConfigPatch(BaseModel):
    track_detector: str | None = None
    track_model_path: str | None = None
    track_remote_url: str | None = None
    track_imgsz: int | None = None
    track_min_score: float | None = None
    track_algorithm: str | None = None
    track_rate_hz: float | None = None
    track_hfov_deg: float | None = None
    track_vfov_deg: float | None = None
    track_mount_yaw_deg: float | None = None
    track_target_size: float | None = None
    track_follow_distance_m: float | None = None
    track_drive_speed: float | None = None
    track_rotate_speed: float | None = None
    track_max_session_seconds: float | None = None
    track_search_seconds: float | None = None
    track_acquire_seconds: float | None = None
    track_body_assist_deg: float | None = None
    track_search_waypoint_seconds: float | None = None
    track_search_waypoint_hold_seconds: float | None = None
    track_search_body_step_deg: float | None = None
    track_search_body_seconds: float | None = None
    track_search_wheel_step_deg: float | None = None
    track_search_full_turn_seconds: float | None = None
    track_search_wheels_after: float | None = None
    track_rotate_deg_per_s: float | None = None


class MountCalibrateRequest(BaseModel):
    state: str                      # "on" (bolted on) or "off"
    samples: int = 8


class MountConfigPatch(BaseModel):
    mount_enabled: bool | None = None
    mount_interval_s: float | None = None
    mount_rssi_on: float | None = None
    mount_rssi_off: float | None = None


class TrackStartRequest(BaseModel):
    target: str
    distance_cm: float | None = None


class LookRequest(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0


class TurnRequest(BaseModel):
    degrees: float


def wire_routes(app: FastAPI, link: WheelsLink, voice_board=None,
                motion=None, frame_provider=None, follow_manager=None,
                mount_monitor=None, posed_frame_provider=None) -> None:
    def depth_source():
        return DepthSource(link.snapshot().get("depth_url", DEFAULT_DEPTH_URL))

    @app.post("/api/sensors/config")
    def sensor_config(patch: SensorConfigPatch):
        try:
            url = validate_url(patch.depth_url)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        link.reconfigure(depth_url=url)
        return {"ok": True, "depth_url": url}

    @app.get("/api/sensors/status")
    def sensor_status():
        url = link.snapshot().get("depth_url", DEFAULT_DEPTH_URL)
        try:
            depth = {"connected": True, **depth_source().status()}
        except (OSError, ValueError) as exc:
            depth = {"connected": False, "error": str(exc)}
        return {
            "depth": {**depth, "url": url, "frame_id": DEPTH_FRAME},
            "rgb": {"endpoint": "/api/camera", "frame_id": RGB_FRAME,
                    "provider_available": frame_provider is not None},
            "navigation": {"actuation_enabled": False, "mode": "sensor_bringup",
                           "blockers": ["base localization", "sensor extrinsics",
                                        "metric wheel calibration", "motion arbitration"]},
            "synchronized_rgbd": False,
        }

    @app.get("/api/sensors/depth")
    def depth_frame():
        try:
            payload, meta = depth_source().frame()
        except (OSError, ValueError) as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
        return Response(content=payload, media_type="application/octet-stream", headers={
            "Cache-Control": "no-store", "X-Frame-Id": DEPTH_FRAME,
            "X-Sequence": str(meta["sequence"]),
            "X-Received-At": str(meta["received_at_unix"]),
            "X-Timestamp-Kind": meta["timestamp_kind"],
            "X-Age-Upper-Bound-Ms": str(meta["age_upper_bound_ms"]),
        })

    @app.get("/api/status")
    def status():
        snap = link.snapshot()
        base = {"host": snap["host"], "port": snap["port"],
                "default_speed": snap["default_speed"]}
        try:
            return {**base, "connected": True, "state": link.board_state_cached()}
        except WheelsError as exc:
            return {**base, "connected": False, "error": str(exc)}

    @app.post("/api/cmd")
    def cmd(req: CmdRequest):
        if req.command not in PRIMITIVES and req.command not in _EXTRA_COMMANDS:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": f"unknown command {req.command!r}"})
        extra = {}
        if req.command == "move":
            extra = {"vx": req.vx or 0.0, "vy": req.vy or 0.0,
                     "omega": req.omega or 0.0}
        try:
            reply = link.client.command(
                req.command, speed=req.speed, duration=req.duration, **extra)
            return {"ok": True, "state": reply.get("state")}
        except WheelsError as exc:
            return JSONResponse(status_code=502,
                                content={"ok": False, "error": str(exc)})

    @app.post("/api/stop")
    def stop():
        # A follow loop re-commands the base several times a second, so
        # halting the chassis without cancelling the loop would last about
        # 100 ms. STOP means stop everything.
        following = False
        if follow_manager is not None:
            following = bool(follow_manager.stop("stop requested").get("was_active"))
        try:
            return {"ok": True, "following_cancelled": following,
                    **link.client.stop()}
        except WheelsError as exc:
            return JSONResponse(status_code=502,
                                content={"ok": False, "error": str(exc),
                                         "following_cancelled": following})

    # --- wheel lab: drive the corners by hand ---------------------------
    # Rotate-in-place is where this chassis behaves worst, and `move()`
    # cannot express an asymmetric attempt at fixing it. These routes are
    # the manual escape hatch: name four numbers, send them, watch.

    @app.get("/api/wheels/lab")
    def wheels_lab():
        snap = link.snapshot()
        trim = dict(snap.get("wheel_trim") or {})
        invert = dict(snap.get("wheel_invert") or {})
        payload = {
            "wheels": list(wheel_lab.WHEEL_NAMES),
            "labels": wheel_lab.WHEEL_LABELS,
            "presets": [dict(p) for p in wheel_lab.PRESETS],
            "trim": trim,
            "invert": invert,
            "trim_range": [wheel_lab.TRIM_MIN, wheel_lab.TRIM_MAX],
            "mix": wheel_lab.clean_mix(snap.get("wheel_lab_mix")),
            "speed": float(snap.get("wheel_lab_speed", snap["default_speed"])),
            "duration": float(snap.get("wheel_lab_duration", 1.0)),
            "pins_snippet": wheel_lab.pins_snippet(trim, invert),
        }
        try:
            board = link.board_state_cached()
            # Present on boards new enough to report it; absent means the
            # chassis is running firmware from before the lab existed.
            payload["board_tuning"] = board.get("tuning")
            payload["connected"] = True
        except WheelsError as exc:
            payload["connected"] = False
            payload["error"] = str(exc)
        return payload

    @app.post("/api/wheels/mix")
    def wheels_mix(req: WheelMixRequest):
        unknown = wheel_lab.unknown_wheels(req.speeds)
        if unknown:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": f"unknown wheel {unknown[0]!r}"})
        snap = link.snapshot()
        mix = wheel_lab.clean_mix(req.speeds)
        speed = float(req.speed if req.speed is not None
                      else snap.get("wheel_lab_speed", snap["default_speed"]))
        speed = max(0.0, min(1.0, speed))
        duration = req.duration
        if duration is not None:
            # The board clamps to 30 s; a hand-driven experiment wants far
            # less than that between chances to hit stop.
            duration = max(0.05, min(10.0, float(duration)))
        try:
            reply = link.client.set_wheels(mix, speed=speed, duration=duration)
        except WheelsError as exc:
            return JSONResponse(status_code=502,
                                content={"ok": False, "error": str(exc)})
        if req.remember:
            link.reconfigure(wheel_lab_mix=mix, wheel_lab_speed=speed,
                             **({"wheel_lab_duration": duration}
                                if duration is not None else {}))
        return {"ok": True, "mix": mix, "speed": speed, "duration": duration,
                "summary": wheel_lab.describe(mix, speed),
                "state": reply.get("state")}

    @app.post("/api/wheels/tune")
    def wheels_tune(req: WheelTuneRequest):
        """Push one wheel's trim/polarity to the live board, and remember it.

        The board forgets both on reset, so the app keeps a copy: that is
        what makes a trim found at 11pm still there in the morning, and what
        `pins_snippet` prints for pasting into the chassis project.
        """
        if req.wheel not in wheel_lab.WHEEL_NAMES:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": f"unknown wheel {req.wheel!r}"})
        if req.trim is None and req.invert is None:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "nothing to tune"})
        trim = None if req.trim is None else wheel_lab.clamp_trim(req.trim)
        try:
            reply = link.client.tune(req.wheel, trim=trim, invert=req.invert)
        except WheelsError as exc:
            return JSONResponse(status_code=502,
                                content={"ok": False, "error": str(exc)})
        snap = link.snapshot()
        trims = dict(snap.get("wheel_trim") or {})
        inverts = dict(snap.get("wheel_invert") or {})
        if trim is not None:
            trims[req.wheel] = trim
        if req.invert is not None:
            inverts[req.wheel] = bool(req.invert)
        link.reconfigure(wheel_trim=trims, wheel_invert=inverts)
        return {"ok": True, "wheel": req.wheel, "trim": trims.get(req.wheel),
                "invert": inverts.get(req.wheel), "board": reply,
                "pins_snippet": wheel_lab.pins_snippet(trims, inverts)}

    @app.post("/api/wheels/tune/apply")
    def wheels_tune_apply():
        """Re-push every remembered trim — the board lost them on reset."""
        snap = link.snapshot()
        trims = dict(snap.get("wheel_trim") or {})
        inverts = dict(snap.get("wheel_invert") or {})
        applied, failed = [], []
        for name in wheel_lab.WHEEL_NAMES:
            if name not in trims and name not in inverts:
                continue
            try:
                link.client.tune(name, trim=trims.get(name),
                                 invert=inverts.get(name))
                applied.append(name)
            except WheelsError as exc:
                failed.append({"wheel": name, "error": str(exc)})
        return {"ok": not failed, "applied": applied, "failed": failed,
                "pins_snippet": wheel_lab.pins_snippet(trims, inverts)}

    @app.get("/api/config")
    def get_config():
        snap = link.snapshot()
        return {**snap, "token": bool(snap.get("token"))}  # never echo the secret

    @app.post("/api/config")
    def set_config(patch: ConfigPatch):
        updates = patch.model_dump(exclude_none=True)
        if "default_speed" in updates:
            updates["default_speed"] = max(0.1, min(1.0, updates["default_speed"]))
        if not updates:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "nothing to update"})
        snap = link.reconfigure(**updates)
        return {**snap, "token": bool(snap.get("token"))}

    @app.get("/api/voice/status")
    def voice_status():
        snap = link.snapshot()
        base = {
            "voice_enabled": bool(snap.get("voice_enabled", True)),
            "model": snap.get("gemini_model", ""),
            "gemini_voice": snap.get("gemini_voice", ""),
            "gemini_language": snap.get("gemini_language", ""),
            "key_present": bool(config_store.resolve_gemini_key(snap)),
        }
        if voice_board is None:
            return {**base, "connected": False, "detail": "voice not available",
                    "transcript": []}
        return {**base, **voice_board.snapshot()}

    @app.post("/api/voice/config")
    def set_voice_config(patch: VoiceConfigPatch):
        updates = patch.model_dump(exclude_none=True)
        if "max_drive_seconds" in updates:
            updates["max_drive_seconds"] = max(
                0.5, min(10.0, updates["max_drive_seconds"]))
        for cal in ("cal_cm_per_s", "cal_deg_per_s"):
            if cal in updates:
                updates[cal] = max(5.0, min(300.0, updates[cal]))
        if not updates:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "nothing to update"})
        snap = link.reconfigure(**updates)
        return {
            "ok": True,
            "key_present": bool(config_store.resolve_gemini_key(snap)),
            "gemini_voice": snap.get("gemini_voice", ""),
            "voice_enabled": bool(snap.get("voice_enabled", True)),
            "enable_video": bool(snap.get("enable_video", True)),
            "note": "voice session picks up changes on next app (re)start "
                    "unless it was still waiting for a key",
        }

    # --- external operator surface (head/body + camera) -----------------
    # Lets any LAN agent — a coding agent or script on the station, an XR
    # client — use the same motion tiers and eyes the voice agent has.

    def _no_motion():
        return JSONResponse(status_code=503, content={
            "ok": False, "error": "head/body motion not available"})

    @app.get("/api/motion")
    def motion_posture():
        if motion is None:
            return _no_motion()
        return {"status": "ok", **motion.posture(),
                "limits": {"head_yaw": motion.limits.head_yaw,
                           "head_pitch_up": motion.limits.head_pitch_up,
                           "head_pitch_down": motion.limits.head_pitch_down,
                           "body_yaw": motion.limits.body_yaw}}

    @app.post("/api/motion/look")
    def motion_look(req: LookRequest):
        if motion is None:
            return _no_motion()
        try:
            return motion.look(req.yaw, req.pitch)
        except Exception as exc:  # noqa: BLE001 — SDK faults become JSON
            return JSONResponse(status_code=500,
                                content={"status": "error", "error": repr(exc)})

    @app.post("/api/motion/turn")
    def motion_turn(req: TurnRequest):
        if motion is None:
            return _no_motion()
        try:
            return motion.turn_body(req.degrees)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=500,
                                content={"status": "error", "error": repr(exc)})

    @app.post("/api/motion/center")
    def motion_center():
        if motion is None:
            return _no_motion()
        try:
            return motion.center()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=500,
                                content={"status": "error", "error": repr(exc)})

    @app.get("/api/camera/posed-frame")
    def posed_frame(age_ms: float = 0):
        if posed_frame_provider is None:
            return JSONResponse(status_code=503, content={"error": "Posed camera unavailable"})
        try:
            return Response(content=posed_frame_provider(age_ms=age_ms), media_type="application/zip",
                            headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})

    @app.get("/api/camera")
    def camera():
        if frame_provider is None:
            return JSONResponse(status_code=503, content={
                "ok": False, "error": "camera not available"})
        jpeg = frame_provider()
        if not jpeg:
            return JSONResponse(status_code=503, content={
                "ok": False, "error": "no frame yet"})
        return Response(content=jpeg, media_type="image/jpeg", headers={
            "Cache-Control": "no-store", "X-Frame-Id": RGB_FRAME,
            "X-Received-At": str(time.time()),
            "X-Timestamp-Kind": "sdk_read_time_not_capture_time",
        })

    # --- visual following -----------------------------------------------

    def _no_follow():
        return JSONResponse(status_code=503, content={
            "status": "error",
            "error": "visual following is not available (no camera/chassis "
                     "wired in this process)"})

    @app.post("/api/track/start")
    def track_start(req: TrackStartRequest):
        if follow_manager is None:
            return _no_follow()
        result = follow_manager.start(req.target, distance_cm=req.distance_cm)
        if result.get("status") != "ok":
            return JSONResponse(status_code=400, content=result)
        return result

    @app.post("/api/track/stop")
    def track_stop():
        if follow_manager is None:
            return _no_follow()
        return follow_manager.stop("stopped from the UI")

    @app.get("/api/track/status")
    def track_status():
        snap = link.snapshot()
        base = {"detector": snap.get("track_detector", "onnx"),
                "remote_url": snap.get("track_remote_url", ""),
                "algorithm": snap.get("track_algorithm", "sort")}
        if follow_manager is None:
            return {**base, "available": False, "active": False,
                    "phase": "unavailable", "events": []}
        return {**base, "available": True, **follow_manager.status()}

    @app.get("/api/track/preview")
    def track_preview():
        if follow_manager is None:
            return _no_follow()
        jpeg = follow_manager.preview()
        if not jpeg:
            return JSONResponse(status_code=503, content={
                "status": "error", "error": "no annotated frame yet"})
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/track/probe")
    def track_probe(target: str = ""):
        """Diagnostics: what does the follow loop's own detector see now?"""
        if follow_manager is None:
            return _no_follow()
        return follow_manager.probe(target)

    @app.get("/api/track/vocabulary")
    def track_vocabulary():
        """What this backend can be asked to follow."""
        snap = link.snapshot()
        if str(snap.get("track_detector", "onnx")) == "remote":
            return {"open_vocabulary": True, "classes": []}
        return {"open_vocabulary": False, "classes": list(COCO_CLASSES)}

    @app.post("/api/track/config")
    def set_track_config(patch: TrackConfigPatch):
        updates = patch.model_dump(exclude_none=True)
        if "track_detector" in updates and updates["track_detector"] not in BACKENDS:
            return JSONResponse(status_code=400, content={
                "ok": False,
                "error": f"track_detector must be one of {', '.join(BACKENDS)}"})
        for key, lo, hi in (("track_min_score", 0.05, 0.95),
                            ("track_target_size", 0.05, 0.95),
                            ("track_follow_distance_m", 0.4, 6.0),
                            ("track_rate_hz", 1.0, 30.0),
                            ("track_drive_speed", 0.2, 1.0),
                            ("track_rotate_speed", 0.2, 1.0),
                            ("track_hfov_deg", 20.0, 180.0),
                            ("track_vfov_deg", 15.0, 180.0),
                            ("track_mount_yaw_deg", -180.0, 180.0),
                            ("track_max_session_seconds", 5.0, 1800.0),
                            ("track_search_seconds", 2.0, 300.0),
                            ("track_acquire_seconds", 2.0, 300.0),
                            ("track_body_assist_deg", 3.0, 60.0),
                            ("track_search_waypoint_seconds", 0.5, 10.0),
                            ("track_search_waypoint_hold_seconds", 0.0, 5.0),
                            ("track_search_body_step_deg", 5.0, 120.0),
                            ("track_search_body_seconds", 1.0, 30.0),
                            ("track_search_wheel_step_deg", 10.0, 90.0),
                            ("track_search_full_turn_seconds", 3.0, 120.0),
                            ("track_search_wheels_after", 0.0, 120.0),
                            ("track_rotate_deg_per_s", 10.0, 300.0)):
            if key in updates:
                updates[key] = max(lo, min(hi, float(updates[key])))
        if "track_imgsz" in updates:
            updates["track_imgsz"] = max(96, min(1280, int(updates["track_imgsz"])))
        if not updates:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "nothing to update"})
        snap = link.reconfigure(**updates)
        return {"ok": True, **{k: snap[k] for k in updates}}

    # --- am I on the wheels? --------------------------------------------

    @app.get("/api/mount/status")
    def mount_status():
        snap = link.snapshot()
        base = {"enabled": bool(snap.get("mount_enabled", True)),
                "ssid": snap.get("mount_ssid", "mecanum-beacon")}
        if mount_monitor is None:
            return {**base, "available": False, "state": "unavailable",
                    "detail": "mount detection is not running in this process"}
        return {**base, "available": True, **mount_monitor.snapshot()}

    @app.post("/api/mount/calibrate")
    def mount_calibrate(req: MountCalibrateRequest):
        """Record what the beacon sounds like in one of the two states.

        Put the robot in that state first — the whole point is that the two
        baselines are measured, not assumed.
        """
        if mount_monitor is None:
            return JSONResponse(status_code=503, content={
                "ok": False, "error": "mount detection is not running"})
        want = str(req.state).strip().lower()
        if want not in ("on", "off"):
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "state must be 'on' or 'off'"})

        samples = mount_monitor.collect(samples=max(1, min(30, req.samples)))
        if not samples:
            return JSONResponse(status_code=502, content={
                "ok": False,
                "error": "heard nothing from the beacon — is the chassis "
                         "powered and its ranging AP up?"})

        import statistics as _stats
        median = round(_stats.median(samples), 1)
        key = "mount_rssi_on" if want == "on" else "mount_rssi_off"
        snap = link.reconfigure(**{key: median})
        thresholds = mount_mod.thresholds_from_state(snap)
        return {"ok": True, "state": want, "median_dbm": median,
                "samples": len(samples),
                "spread_db": round(max(samples) - min(samples), 1),
                "on_dbm": thresholds.on_dbm, "off_dbm": thresholds.off_dbm,
                "calibrated": thresholds.calibrated,
                "threshold_dbm": thresholds.midpoint,
                "note": None if thresholds.calibrated else
                        ("record the other state too — and they must differ by "
                         f"at least {thresholds.min_separation:.0f} dB to be "
                         "told apart")}

    @app.post("/api/mount/config")
    def set_mount_config(patch: MountConfigPatch):
        updates = patch.model_dump(exclude_none=True)
        if "mount_interval_s" in updates:
            updates["mount_interval_s"] = max(0.5, min(30.0, updates["mount_interval_s"]))
        if not updates:
            return JSONResponse(status_code=400, content={
                "ok": False, "error": "nothing to update"})
        snap = link.reconfigure(**updates)
        return {"ok": True, **{k: snap[k] for k in updates},
                "note": "interval changes apply on the next app restart"}

    @app.get("/api/log")
    def board_log():
        try:
            return {"connected": True, "lines": link.client.log()}
        except WheelsError as exc:
            return {"connected": False, "lines": [], "error": str(exc)}
