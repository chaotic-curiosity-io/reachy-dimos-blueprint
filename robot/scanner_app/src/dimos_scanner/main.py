"""DimosScanner — Reachy Mini app entry point.

The SDK's ``reachy-mini-app-assistant check`` does a substring scan of this
file for ``class DimosScanner(ReachyMiniApp)``, so the class definition must
live here directly (not be re-exported from a sibling module).

The heavy logic is split out:
  - ``config.BridgeConfig``             env-driven config dataclass
  - ``core.scan_state.ScanState``       head/body pose tracker
  - ``core.motion_loop.run_motion_loop``set_target pump (30 Hz background thread)
  - ``io.bridge_client.BridgeClient``   async WS to the station bridge
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
from reachy_mini import ReachyMini, ReachyMiniApp

from .config import BridgeConfig, load_config, persist_config
from .core.motion_loop import run_motion_loop
from .core.scan_state import (
    MAX_BODY_YAW_DEG,
    MAX_HEAD_PITCH_DEG,
    MAX_HEAD_YAW_DEG,
    ScanState,
)
from .io.bridge_client import BridgeClient

logger = logging.getLogger("dimos_scanner.main")


class DimosScanner(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None  # auto

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Persisted file > env > defaults, so the Mac bridge host (and the rest)
        # survives restarts/redeploys instead of resetting each time.
        self.config = load_config()
        # Shared pose state driven by both arrow keys (server window) and the
        # settings-page pose buttons. The 30 Hz motion loop pushes it to the
        # robot, so both input paths just mutate this one object.
        self._scan = ScanState()
        self._bridge_client = None
        # The framework reads ``custom_app_url`` at __init__ time to decide
        # whether to mount the settings FastAPI server. Honour the env override.
        if self.config.settings_url:
            type(self).custom_app_url = self.config.settings_url

    # ------------------------------------------------------------------ #
    # Main entry — called by ReachyMiniApp.wrapped_run                    #
    # ------------------------------------------------------------------ #

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        scan = self._scan
        if self.settings_app is not None:
            self._wire_settings_endpoints()

        # Wake the robot and start the motion pump immediately — BEFORE waiting
        # for a bridge host — so the settings-page pose buttons (and arrow keys)
        # can move the robot the moment the app is up, even with no bridge yet.
        try:
            reachy_mini.wake_up()
        except Exception as e:  # noqa: BLE001
            logger.warning("wake_up failed (continuing): %s", e)

        reachy_mini.set_target(head=scan.head_pose(), body_yaw=scan.body_yaw_rad)
        motion_thread = threading.Thread(
            target=run_motion_loop,
            args=(reachy_mini, scan, stop_event),
            daemon=True,
            name="dimos-scanner-motion",
        )
        motion_thread.start()

        try:
            self._run_bridge(reachy_mini, scan, stop_event)
        finally:
            stop_event.set()
            motion_thread.join(timeout=2.0)
            try:
                reachy_mini.goto_sleep()
            except Exception:  # noqa: BLE001
                pass

    def _run_bridge(
        self, reachy_mini: ReachyMini, scan: ScanState, stop_event: threading.Event
    ) -> None:
        # If no bridge host was configured, idle waiting for the operator to
        # set one via the settings page — don't fail-fast and lose the slot.
        # Pose controls already work here (the motion loop is running).
        if not self.config.host:
            logger.warning(
                "DIMOS_SCANNER_BRIDGE_HOST is empty — set it at %s before scanning.",
                self.config.settings_url,
            )
            while not stop_event.is_set() and not self.config.host:
                time.sleep(0.5)
            if stop_event.is_set():
                return

        def _producer() -> np.ndarray | None:
            try:
                return reachy_mini.media.get_frame()
            except Exception as e:  # noqa: BLE001
                logger.warning("get_frame failed: %s", e)
                return None

        # The IMU is wireless-only. Probe once at startup; if the attribute
        # doesn't exist (Lite variant, future SDK rename) we leave the producer
        # as None so the bridge_client skips spawning the send task entirely.
        imu_producer = None
        try:
            _ = reachy_mini.imu
            def _imu_producer() -> dict | None:
                try:
                    return reachy_mini.imu
                except Exception as e:  # noqa: BLE001
                    logger.warning("imu read failed: %s", e)
                    return None
            imu_producer = _imu_producer
            logger.info("IMU available — streaming at %.1f Hz", self.config.imu_hz)
        except Exception as e:  # noqa: BLE001
            logger.info("IMU not available on this Reachy Mini (%s) — skipping IMU stream", e)

        # Head-pose stream (forward kinematics, body/FLU convention — the Mac
        # composes the head->camera extrinsic). The Mac uses this to drive the
        # spatial pipeline from kinematics (--pose external) instead of
        # monocular VO, which smears the map on in-place rotation. Probe once;
        # if the SDK lacks the getter we leave it None.
        pose_producer = None
        pose_stream = False
        try:
            _ = reachy_mini.get_current_head_pose()
            def _pose_producer():
                try:
                    return reachy_mini.get_current_head_pose()
                except Exception as e:  # noqa: BLE001
                    logger.warning("head pose read failed: %s", e)
                    return None
            pose_producer = _pose_producer
            pose_stream = True
            logger.info(
                "head pose available — streaming at %.1f Hz for --pose external",
                self.config.pose_hz,
            )
        except Exception as e:  # noqa: BLE001
            logger.info("head pose not available (%s) — Mac will fall back to VO", e)

        hello_config: dict = {
            "depth_model": self.config.depth_model,
            "pose_stream": pose_stream,
        }
        # Advertise the real camera intrinsics so the Mac backprojects depth
        # with the true K instead of a calibration-file guess that may not
        # match the streamed resolution/crop.
        try:
            cam = reachy_mini.media.camera
            K = np.asarray(cam.K, dtype=float)
            res = getattr(cam, "resolution", None)
            hello_config["camera_K"] = [float(v) for v in K.flatten()]
            if res is not None:
                hello_config["camera_wh"] = [int(res[0]), int(res[1])]
            logger.info("advertising camera intrinsics (fx=%.1f)", K[0, 0])
        except Exception as e:  # noqa: BLE001
            logger.info("camera intrinsics unavailable (%s) — Mac will use its calibration file", e)

        client = BridgeClient(
            host=self.config.host,
            port=self.config.port,
            frame_producer=_producer,
            scan=scan,
            jpeg_quality=self.config.jpeg_quality,
            frame_hz=self.config.frame_hz,
            hello_config=hello_config,
            imu_producer=imu_producer,
            imu_hz=self.config.imu_hz,
            imu_enabled=self.config.imu_enabled,
            pose_producer=pose_producer,
            pose_hz=self.config.pose_hz,
        )
        # Stash the client so the settings-page POST handler can force a
        # reconnect when the operator picks a new depth model — that's how the
        # new preference reaches the Mac server.
        self._bridge_client = client

        # Cleanup (motion thread join + goto_sleep) is handled by run()'s finally.
        asyncio.run(client.run(stop_event))

    # ------------------------------------------------------------------ #
    # Settings page (FastAPI mounted at custom_app_url)                   #
    # ------------------------------------------------------------------ #

    def _wire_settings_endpoints(self) -> None:
        assert self.settings_app is not None

        from fastapi import Body

        @self.settings_app.get("/config")
        def get_config():  # noqa: ANN202
            # Expose the derived launcher URL too so the page doesn't have to
            # know the derivation rule.
            return {
                **self.config.__dict__,
                "derived_launcher_url": self.config.derived_launcher_url(),
            }

        # Annotate the body parameter with ``Body(...)`` so FastAPI doesn't
        # fall back to query-string inference (Pydantic-v1/v2 + older FastAPI
        # in the robot's apps_venv gave us ``loc=["query","patch"]`` + HTTP
        # 422). A plain ``dict`` body works in every FastAPI version.
        @self.settings_app.post("/config")
        def set_config(patch: dict = Body(...)):  # noqa: ANN202
            from .config import DEPTH_MODEL_CHOICES, POSE_CHOICES, DEVICE_CHOICES

            allowed = {
                "host", "port", "jpeg_quality", "frame_hz", "pose_hz", "depth_model",
                "imu_hz", "imu_enabled",
                # Mac-side pipeline knobs:
                "pose", "display_width", "max_fps", "device",
                "enable_clip_memory", "save_map", "no_detect", "launcher_url",
            }
            kw = {}
            for k in allowed:
                if k in patch and patch[k] is not None:
                    kw[k] = patch[k]
            # Light coercion so a form posting strings or a curl with mixed
            # types both work.
            if "port" in kw:
                kw["port"] = int(kw["port"])
            if "jpeg_quality" in kw:
                kw["jpeg_quality"] = int(kw["jpeg_quality"])
            if "frame_hz" in kw:
                kw["frame_hz"] = float(kw["frame_hz"])
            if "pose_hz" in kw:
                kw["pose_hz"] = float(kw["pose_hz"])
            if "imu_hz" in kw:
                kw["imu_hz"] = float(kw["imu_hz"])
            if "imu_enabled" in kw:
                kw["imu_enabled"] = bool(kw["imu_enabled"])
            if "display_width" in kw:
                kw["display_width"] = int(kw["display_width"])
            if "max_fps" in kw:
                kw["max_fps"] = float(kw["max_fps"])
            for bk in ("enable_clip_memory", "save_map", "no_detect"):
                if bk in kw:
                    kw[bk] = bool(kw[bk])
            if "depth_model" in kw and kw["depth_model"] not in DEPTH_MODEL_CHOICES:
                return {
                    "ok": False,
                    "error": f"depth_model must be one of {list(DEPTH_MODEL_CHOICES)}",
                }
            if "pose" in kw and kw["pose"] not in POSE_CHOICES:
                return {"ok": False, "error": f"pose must be one of {list(POSE_CHOICES)}"}
            if "device" in kw and kw["device"] not in DEVICE_CHOICES:
                return {"ok": False, "error": f"device must be one of {list(DEVICE_CHOICES)}"}
            old = self.config
            self.config = self.config.replace(**kw)
            persist_config(self.config)  # survive restarts/redeploys
            # If any field that the BridgeClient reads at connect time changed,
            # update the client and bounce the WS so the new hello goes out.
            connect_time_fields = ("host", "port", "jpeg_quality", "frame_hz", "pose_hz", "depth_model", "imu_hz")
            changed = any(getattr(old, k) != getattr(self.config, k) for k in connect_time_fields)
            client = getattr(self, "_bridge_client", None)
            if client is not None:
                # ``imu_enabled`` is mutable live — apply it without forcing a
                # reconnect so the operator can pause/resume the stream cheaply.
                client.imu_enabled = self.config.imu_enabled
            if changed and client is not None:
                client.host = self.config.host
                client.port = self.config.port
                client.jpeg_quality = self.config.jpeg_quality
                client.frame_hz = self.config.frame_hz
                client.pose_hz = self.config.pose_hz
                client.imu_hz = self.config.imu_hz
                # Mutate in place — replacing the dict would drop the
                # pose_stream/camera_K keys built at connect time.
                client.hello_config["depth_model"] = self.config.depth_model
                client.request_reconnect()
            return {
                "ok": True,
                "config": {
                    **self.config.__dict__,
                    "derived_launcher_url": self.config.derived_launcher_url(),
                },
            }

        @self.settings_app.get("/config/choices")
        def get_choices():  # noqa: ANN202
            from .config import DEPTH_MODEL_CHOICES, POSE_CHOICES, DEVICE_CHOICES

            return {
                "depth_model": list(DEPTH_MODEL_CHOICES),
                "pose": list(POSE_CHOICES),
                "device": list(DEVICE_CHOICES),
            }

        # --- robot pose controls ------------------------------------------
        # The settings-page sliders/nudge buttons mutate the shared ScanState;
        # the 30 Hz motion loop pushes it to the robot. Same model as the
        # arrow-key path — both just move this one object. Limits mirror
        # ScanState so curl/scripted callers can't command past the IK + motor.
        def _clamp(v: float, lim: float) -> float:
            return max(-lim, min(lim, v))

        @self.settings_app.post("/api/head_pose")
        def set_head_pose(patch: dict = Body(...)):  # noqa: ANN202
            yaw = _clamp(float(patch.get("yaw", self._scan.head_yaw_deg)), MAX_HEAD_YAW_DEG)
            pitch = _clamp(float(patch.get("pitch", self._scan.head_pitch_deg)), MAX_HEAD_PITCH_DEG)
            self._scan.head_yaw_deg = yaw
            self._scan.head_pitch_deg = pitch
            return {"yaw": yaw, "pitch": pitch}

        @self.settings_app.post("/api/body_yaw")
        def set_body_yaw(patch: dict = Body(...)):  # noqa: ANN202
            yaw = _clamp(float(patch.get("yaw", self._scan.body_yaw_deg)), MAX_BODY_YAW_DEG)
            self._scan.body_yaw_deg = yaw
            return {"yaw": yaw}

        @self.settings_app.post("/api/home")
        def go_home():  # noqa: ANN202
            self._scan.apply("reset")
            return {"yaw": 0.0, "pitch": 0.0, "body_yaw": 0.0}

        @self.settings_app.get("/api/pose_limits")
        def pose_limits():  # noqa: ANN202
            return {
                "head_yaw": MAX_HEAD_YAW_DEG,
                "head_pitch": MAX_HEAD_PITCH_DEG,
                "body_yaw": MAX_BODY_YAW_DEG,
            }

        @self.settings_app.get("/api/objects")
        def get_objects():  # noqa: ANN202
            # Live tracked-object list (name + distance-from-robot + detection
            # count), pushed by the Mac dimos pipeline over the bridge WS. The
            # web page polls this to render the list.
            client = getattr(self, "_bridge_client", None)
            if client is None:
                return {"objects": [], "ts": 0.0, "connected": False, "age_s": None}
            ts = client.latest_objects_ts
            age = (time.time() - ts) if ts else None
            return {
                "objects": client.latest_objects,
                "ts": ts,
                "connected": client.connected,
                "age_s": age,
            }

        @self.settings_app.post("/api/relocalize")
        def relocalize():  # noqa: ANN202
            # Ask the Mac to relocalize the (moved) robot against its saved
            # reference map. The Mac sweeps the head, captures a local scan, and
            # solves the alignment, streaming progress back as relocalize_status.
            client = getattr(self, "_bridge_client", None)
            if client is None or not client.connected:
                return {"ok": False, "error": "not connected to the Mac bridge"}
            sent = client.request_relocalize()
            return {"ok": bool(sent)}

        @self.settings_app.get("/api/relocalize/status")
        def relocalize_status():  # noqa: ANN202
            # Latest relocalization status pushed by the Mac pipeline (state /
            # fitness / translation). The web page polls this after pressing the
            # Relocalize button.
            client = getattr(self, "_bridge_client", None)
            if client is None:
                return {"status": {}, "ts": 0.0, "connected": False, "age_s": None}
            ts = client.latest_reloc_status_ts
            age = (time.time() - ts) if ts else None
            return {
                "status": client.latest_reloc_status,
                "ts": ts,
                "connected": client.connected,
                "age_s": age,
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = DimosScanner()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
