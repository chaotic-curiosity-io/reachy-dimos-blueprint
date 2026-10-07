"""Reachy Mini → Wheels mecanum chassis bridge app, with voice.

Reachy Mini itself has no wheels; the Wheels chassis (separate ESP32
hardware project) is a piece of technology Reachy can tap into. This app:

  * serves a D-pad web UI at http://<robot>:8042 (manual driving), and
  * hosts a Gemini Live audio+video session
    where Gemini hears you through the robot mic, sees through the head
    camera, talks back through the speaker, and drives the chassis via
    drive/stop/wheels_state tool calls, and
  * runs a visual follow loop (``tracking/``): a detector plus Roboflow
    ``trackers`` lock onto a named person or object and hold the wheels and
    head on it continuously until stopped.

Threads: the run() thread idles (all work is event-driven); a worker thread
hosts the asyncio loop for the Gemini session + mic/camera/speaker pumps; a
follow session adds a perception thread plus two coalescing actuator threads
while it is active.

Safety: every chassis command arms a board-side deadman (2 s, or the sent
duration, capped at 30 s); voice drives are additionally clamped to
``max_drive_seconds`` (default 4 s). On shutdown we send a best-effort /stop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from reachy_mini import ReachyMini, ReachyMiniApp

from . import config as config_store
from .api import WheelsLink, wire_routes
from .mount import MountMonitor
from .tracking import FollowManager
from .voice.board import VoiceBoard
from .voice.motion import RobotMotion  # SDK-free at import time
from .wheels_client import WheelsError

log = logging.getLogger("reachy_wheels_app")


class ReachyWheelsApp(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        )
        log.info("=== reachy_wheels_app v0.9.1 — DimOS sensors + visual following "
                 "(head→shell→wheels) + WiFi mount detection ===")

        link = WheelsLink()
        board = VoiceBoard()
        if link.state.get("host"):
            log.info("wheels chassis expected at http://%s:%s",
                     link.state["host"], link.state["port"])
        else:
            log.warning("no wheels chassis host set — set WHEELS_HOST or enter "
                        "the ESP32's IP in the UI at %s", self.custom_app_url)

        # One motion adapter shared by the voice agent AND the HTTP surface,
        # so both see the same tracked head/body angles.
        motion = RobotMotion(reachy_mini)

        from .posed_camera import PosedCamera
        posed_camera = PosedCamera(reachy_mini)

        def raw_frame_provider():
            """Latest head-camera frame as a BGR ndarray (for tracking)."""
            try:
                return posed_camera.frame()
            except Exception:  # noqa: BLE001
                log.exception("get_frame failed")
                return None

        def frame_provider() -> bytes | None:
            """Latest head-camera frame as JPEG (for /api/camera)."""
            import cv2
            frame = raw_frame_provider()
            if frame is None:
                return None
            ok, jpeg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return jpeg.tobytes() if ok else None

        # Visual following shares the motion adapter and the chassis client
        # with the voice agent, so "stop" from any surface stops everything.
        follow = FollowManager(
            frame_provider=raw_frame_provider,
            wheels_provider=lambda: link.client,
            motion=motion,
            state_provider=link.snapshot,
        )

        # Is the robot actually bolted to the chassis? The board's ranging
        # beacon answers that; the monitor scans for it on its own thread
        # because `iw scan` blocks for a second or more.
        mount = None
        snap = link.snapshot()
        if snap.get("mount_enabled", True):
            mount = MountMonitor(
                link.snapshot,
                iface=str(snap.get("mount_iface", "wlan0")),
                ssid=str(snap.get("mount_ssid", "mecanum-beacon")),
                board_url=(f"http://{snap['host']}:{snap['port']}"
                           if snap.get("host") else ""),
                interval=float(snap.get("mount_interval_s", 2.5)),
                pause_when=lambda: bool(follow.status().get("active")),
            )
            mount.start()

        if self.settings_app is not None:
            wire_routes(self.settings_app, link, voice_board=board,
                        motion=motion, frame_provider=frame_provider,
                        follow_manager=follow, mount_monitor=mount,
                        posed_frame_provider=posed_camera.bundle)

        if link.client.ping():
            log.info("chassis reachable — UI at %s", self.custom_app_url)
        else:
            log.warning("chassis not reachable yet; UI stays up and will "
                        "retry on every command")

        voice = _VoiceWorker(reachy_mini, link, board, stop_event,
                             motion=motion, follow=follow)
        voice.start()

        try:
            # All the work happens in request handlers + the voice worker;
            # this thread only has to stay alive and responsive to stop.
            while not stop_event.is_set():
                time.sleep(0.2)
        finally:
            log.info("stop requested — tearing down")
            if mount is not None:
                mount.stop()
            follow.shutdown()
            posed_camera.close()
            voice.join(timeout=8.0)
            if voice.is_alive():
                log.warning("voice worker still alive after 8s (daemon thread)")
            try:
                link.client.stop()
                log.info("sent final stop to chassis")
            except WheelsError:
                pass  # deadman on the board covers us
            log.info("app stopped")


class _VoiceWorker(threading.Thread):
    """Hosts the asyncio loop for the Gemini Live session and media pumps.

    Waits for a usable Gemini key (pasted into the web UI's voice panel and
    persisted in the app's config) so the first-run flow is: start the app,
    open :8042, paste a key in the voice panel — the session self-starts.
    The D-pad works the whole time regardless.
    """

    def __init__(self, mini: ReachyMini, link: WheelsLink,
                 board: VoiceBoard, stop_event: threading.Event,
                 motion: RobotMotion | None = None, follow=None):
        super().__init__(name="reachy-wheels-voice", daemon=True)
        self._mini = mini
        self._link = link
        self._board = board
        self._stop_event = stop_event
        self._motion = motion
        self._follow = follow

    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:  # noqa: BLE001
            log.exception("voice worker crashed")
            self._board.set_status(False, "voice worker crashed (see app log)")

    def _wait_for_key(self) -> str | None:
        announced = False
        while not self._stop_event.is_set():
            state = self._link.snapshot()
            if not state.get("voice_enabled", True):
                self._board.set_status(False, "voice disabled in settings")
                time.sleep(2.0)
                continue
            key = config_store.resolve_gemini_key(state)
            if key:
                return key
            if not announced:
                log.info("no Gemini key yet — paste one at http://<robot>:8042")
                announced = True
            self._board.set_status(False, "waiting for Gemini API key")
            time.sleep(1.0)
        return None

    async def _main(self) -> None:
        key = await asyncio.get_running_loop().run_in_executor(None, self._wait_for_key)
        if key is None:
            return

        state = self._link.snapshot()

        # Imports deferred so the package (and tests) never require
        # google-genai/numpy unless voice actually starts.
        from .voice.gemini_live import GeminiConfig, GeminiDriver
        from .voice.io_harness import RobotIOHarness
        from .voice.tools import DriveLimits

        loop = asyncio.get_running_loop()

        try:
            self._mini.media.start_recording()
            self._mini.media.start_playing()
        except Exception:  # noqa: BLE001
            log.exception("could not acquire mic/speaker — voice disabled")
            self._board.set_status(False, "mic/speaker unavailable")
            return

        speak = bool(state.get("speak_responses", True))
        harness = RobotIOHarness(
            self._mini, loop=loop,
            enable_video=bool(state.get("enable_video", True)),
            speak_responses=speak,
        )
        driver = GeminiDriver(
            config=GeminiConfig(
                api_key=key,
                model=str(state.get("gemini_model") or "gemini-3.1-flash-live-preview"),
                voice=str(state.get("gemini_voice") or "") or None,
                language=str(state.get("gemini_language") or "") or None,
            ),
            wheels_client=self._link.client,
            motion=self._motion or RobotMotion(self._mini),
            follow=self._follow,
            board=self._board,
            audio_output_callback=harness.audio_output_callback if speak else None,
            limits=DriveLimits(
                max_duration=float(state.get("max_drive_seconds", 4.0)),
                default_speed=float(state.get("default_speed", 0.8)),
                cm_per_s=float(state.get("cal_cm_per_s", 30.0)),
                deg_per_s=float(state.get("cal_deg_per_s", 80.0)),
                mount_yaw_deg=float(state.get("track_mount_yaw_deg", 0.0)),
            ),
            mute_during_response=speak,
        )
        harness.set_provider(driver)

        try:
            await driver.start()
            self._board.set_status(True, "listening")
            log.info("gemini-live voice connected")
        except Exception as exc:  # noqa: BLE001
            log.exception("could not start Gemini session")
            self._board.set_status(False, f"gemini connect failed: {exc}")
            self._release_media()
            return

        harness.start_tasks()
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.25)
        finally:
            await harness.stop_tasks()
            try:
                await driver.stop()
            finally:
                self._board.set_status(False, "stopped")
                self._release_media()

    def _release_media(self) -> None:
        for fn, label in ((self._mini.media.stop_recording, "stop_recording"),
                          (self._mini.media.stop_playing, "stop_playing")):
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("%s raised during voice teardown", label)


if __name__ == "__main__":
    app = ReachyWheelsApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
