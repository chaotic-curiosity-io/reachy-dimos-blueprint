"""WebSocket client that streams camera frames + receives control messages.

Runs an asyncio loop. The send side reads frames from a callable supplied by
the caller (so this module stays decoupled from the Reachy Mini SDK). The
receive side applies control messages to a ``ScanState`` instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional

import numpy as np
import websockets


# Module-level logger so call sites in this file don't rely on the SDK app's
# own logger name.

from .frame_encoder import encode_jpeg
from .protocol import (
    encode_frame,
    encode_imu,
    encode_pose,
    hello,
    parse_text,
    relocalize_request,
)
from ..core.scan_state import ScanState

logger = logging.getLogger("dimos_scanner.bridge_client")


FrameProducer = Callable[[], Optional[np.ndarray]]
"""Synchronous callable that returns the current BGR frame (or None)."""

# The Reachy Mini SDK exposes ``mini.imu`` as a dict snapshot. The producer
# returns that dict (or None if the read failed); we tolerate missing keys.
ImuProducer = Callable[[], Optional[dict]]

# Returns the robot's current head pose as a 4x4 ndarray (forward kinematics,
# body/FLU convention: X-fwd Y-left Z-up, reachy_base -> head) or None. The Mac
# server composes the head->camera extrinsic (rotation to optical axes + lever
# arm) before back-projecting depth. Streamed continuously so the Mac can drive
# the spatial pipeline from kinematics instead of monocular VO.
PoseProducer = Callable[[], Optional[np.ndarray]]


class BridgeClient:
    """Long-lived WS client to the station bridge (``station/``).

    Use ``await run(...)`` to keep the connection alive (reconnecting on
    failure) until ``stop_event`` is set.
    """

    def __init__(
        self,
        host: str,
        port: int,
        frame_producer: FrameProducer,
        scan: ScanState,
        *,
        jpeg_quality: int = 80,
        frame_hz: float = 5.0,
        role: str = "robot",
        name: str = "reachy_mini",
        hello_config: dict | None = None,
        imu_producer: ImuProducer | None = None,
        imu_hz: float = 50.0,
        imu_enabled: bool = True,
        pose_producer: PoseProducer | None = None,
        pose_hz: float = 20.0,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_producer = frame_producer
        self.pose_producer = pose_producer
        self.scan = scan
        self.jpeg_quality = jpeg_quality
        self.frame_hz = frame_hz
        self.role = role
        self.name = name
        # Extra dict folded into the WS hello so the Mac server learns the
        # robot's preferences (depth_model etc.) immediately on connect.
        self.hello_config = hello_config or {}
        # IMU streaming. ``imu_enabled`` is the runtime toggle (mutable from
        # the settings page); ``imu_producer`` being None disables the stream
        # entirely regardless of the toggle.
        self.imu_producer = imu_producer
        self.imu_hz = imu_hz
        self.imu_enabled = imu_enabled
        # Dedicated pose stream rate. The Mac interpolates poses to each
        # frame's capture timestamp, so poses must bracket every frame:
        # stream them faster than frames, independently of the frame loop.
        # <= 0 falls back to the legacy one-pose-before-each-frame behaviour.
        self.pose_hz = pose_hz
        # Set while a WS is alive so request_reconnect() can close it.
        self._active_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Latest tracked-object list pushed by the Mac pipeline (name +
        # distance-from-robot + detection count). The settings page reads this
        # via GET /api/objects. Replaced wholesale on each "objects" message.
        self.latest_objects: list = []
        self.latest_objects_ts: float = 0.0
        # Latest relocalization status pushed by the Mac pipeline (state /
        # fitness / translation). The settings page reads it via
        # GET /api/relocalize/status and drives the Relocalize button's feedback.
        self.latest_reloc_status: dict = {}
        self.latest_reloc_status_ts: float = 0.0

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def connected(self) -> bool:
        """True while a WS to the bridge is live."""
        return self._active_ws is not None

    async def run(self, stop_event) -> None:  # type: ignore[no-untyped-def]
        self._loop = asyncio.get_running_loop()
        backoff = 1.0
        while not stop_event.is_set():
            try:
                logger.info("connecting to bridge %s ...", self.url)
                async with websockets.connect(self.url, max_size=8 * 1024 * 1024) as ws:
                    self._active_ws = ws
                    await ws.send(hello(self.role, name=self.name, config=self.hello_config))
                    logger.info("connected to bridge (hello: %s)", self.hello_config)
                    backoff = 1.0
                    try:
                        tasks = [
                            self._send_frames(ws, stop_event),
                            self._recv_control(ws, stop_event),
                        ]
                        if self.imu_producer is not None:
                            tasks.append(self._send_imu(ws, stop_event))
                        if self.pose_producer is not None and self.pose_hz > 0:
                            tasks.append(self._send_poses(ws, stop_event))
                        await asyncio.gather(*tasks)
                    finally:
                        self._active_ws = None
            except (OSError, websockets.exceptions.WebSocketException) as e:
                logger.warning(
                    "bridge connection lost (%s); reconnecting in %.1fs", e, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)

    def request_reconnect(self) -> None:
        """Close the active WS so ``run()``'s outer loop reconnects.

        Safe to call from any thread (e.g. the FastAPI settings handler) —
        we schedule the close on the bridge client's asyncio loop.
        """
        ws = self._active_ws
        loop = self._loop
        if ws is None or loop is None or loop.is_closed():
            return

        async def _close() -> None:
            try:
                await ws.close(code=1000, reason="config change")
            except Exception:  # noqa: BLE001
                pass

        try:
            asyncio.run_coroutine_threadsafe(_close(), loop)
        except RuntimeError:
            # Loop not running yet — next connect will already use the new
            # hello_config so no action needed.
            pass

    def request_relocalize(self) -> bool:
        """Ask the Mac to relocalize against its reference map.

        Sends a ``relocalize_request`` over the active WS; the Mac sweeps the
        head and runs the one-shot solve, streaming progress back as
        ``relocalize_status`` (see ``latest_reloc_status``). Safe to call from
        the FastAPI settings handler — scheduled on the client's asyncio loop.
        Returns False if no bridge connection is live.
        """
        ws = self._active_ws
        loop = self._loop
        if ws is None or loop is None or loop.is_closed():
            return False

        async def _send() -> None:
            try:
                await ws.send(relocalize_request())
            except Exception as e:  # noqa: BLE001
                logger.warning("relocalize request send failed: %s", e)

        try:
            asyncio.run_coroutine_threadsafe(_send(), loop)
            return True
        except RuntimeError:
            return False

    async def _send_frames(self, ws, stop_event) -> None:  # type: ignore[no-untyped-def]
        period = 1.0 / max(self.frame_hz, 0.1)
        while not stop_event.is_set():
            t0 = time.time()
            try:
                frame = self.frame_producer()
            except Exception as e:  # noqa: BLE001
                logger.warning("frame_producer error: %s", e)
                await asyncio.sleep(period)
                continue
            # Stamp at capture, not at send: JPEG encode + WS backpressure add
            # tens of ms, and the Mac pairs pose↔frame by this timestamp.
            t_cap_ns = time.time_ns()
            if frame is None:
                await asyncio.sleep(period)
                continue
            try:
                jpeg, w, h = encode_jpeg(frame, quality=self.jpeg_quality)
            except Exception as e:  # noqa: BLE001
                logger.warning("encode_jpeg failed: %s", e)
                await asyncio.sleep(period)
                continue
            # Legacy fallback when the dedicated pose stream is disabled
            # (pose_hz <= 0): send one pose immediately before its frame so the
            # Mac's "latest pose" roughly matches. Failures never block the frame.
            if self.pose_producer is not None and self.pose_hz <= 0:
                await self._send_one_pose(ws)
            await ws.send(encode_frame(jpeg, width=w, height=h, ts_ns=t_cap_ns))
            elapsed = time.time() - t0
            if elapsed < period:
                await asyncio.sleep(period - elapsed)

    async def _send_one_pose(self, ws) -> None:  # type: ignore[no-untyped-def]
        try:
            pose = self.pose_producer()
        except Exception as e:  # noqa: BLE001
            logger.warning("pose_producer error: %s", e)
            return
        if pose is None:
            return
        # Stamp at read time — the FK snapshot is current as of this call.
        ts_ns = time.time_ns()
        try:
            await ws.send(encode_pose(np.asarray(pose, dtype=float).flatten(), ts_ns=ts_ns))
        except Exception as e:  # noqa: BLE001
            logger.warning("pose send failed: %s", e)

    async def _send_poses(self, ws, stop_event) -> None:  # type: ignore[no-untyped-def]
        """Stream head poses at ``pose_hz``, independent of the frame loop.

        Poses must arrive densely enough to bracket every frame timestamp so
        the Mac can slerp a pose for the exact capture instant (the robot
        rotates fast enough that one pose per frame aliases badly).
        """
        assert self.pose_producer is not None
        period = 1.0 / max(self.pose_hz, 0.1)
        while not stop_event.is_set():
            t0 = time.time()
            await self._send_one_pose(ws)
            elapsed = time.time() - t0
            if elapsed < period:
                await asyncio.sleep(period - elapsed)

    async def _send_imu(self, ws, stop_event) -> None:  # type: ignore[no-untyped-def]
        """Poll ``imu_producer`` at ``imu_hz`` and stream IMU msgs to the bridge.

        Re-reads ``imu_enabled`` every tick so the settings page can pause /
        resume the stream without a WS reconnect.
        """
        assert self.imu_producer is not None
        warned_missing_field = False
        while not stop_event.is_set():
            period = 1.0 / max(self.imu_hz, 0.1)
            t0 = time.time()
            if not self.imu_enabled:
                await asyncio.sleep(period)
                continue
            try:
                sample = self.imu_producer()
            except Exception as e:  # noqa: BLE001
                logger.warning("imu_producer error: %s", e)
                await asyncio.sleep(period)
                continue
            if sample is None:
                await asyncio.sleep(period)
                continue
            try:
                accel = tuple(float(x) for x in sample.get("accelerometer", (0.0, 0.0, 0.0)))[:3]
                gyro = tuple(float(x) for x in sample.get("gyroscope", (0.0, 0.0, 0.0)))[:3]
                quat = tuple(float(x) for x in sample.get("quaternion", (1.0, 0.0, 0.0, 0.0)))[:4]
                temp = float(sample.get("temperature", float("nan")))
            except (TypeError, ValueError) as e:
                if not warned_missing_field:
                    logger.warning("imu sample shape unexpected (%s): %r", e, sample)
                    warned_missing_field = True
                await asyncio.sleep(period)
                continue
            if len(accel) != 3 or len(gyro) != 3 or len(quat) != 4:
                if not warned_missing_field:
                    logger.warning("imu sample missing fields: %r", sample)
                    warned_missing_field = True
                await asyncio.sleep(period)
                continue
            try:
                await ws.send(encode_imu(accel, gyro, quat, temp))
            except Exception as e:  # noqa: BLE001
                logger.warning("imu send failed: %s", e)
                return
            elapsed = time.time() - t0
            if elapsed < period:
                await asyncio.sleep(period - elapsed)

    async def _recv_control(self, ws, stop_event) -> None:  # type: ignore[no-untyped-def]
        async for msg in ws:
            if stop_event.is_set():
                break
            if isinstance(msg, bytes):
                continue
            try:
                obj = parse_text(msg)
            except Exception as e:  # noqa: BLE001
                logger.warning("bad text msg: %s", e)
                continue
            kind = obj.get("type")
            if kind == "control":
                action = obj.get("action", "")
                step = float(obj.get("step_deg", self.scan.step_deg))
                self.scan.apply(action, step)
            elif kind == "objects":
                objs = obj.get("objects")
                if isinstance(objs, list):
                    self.latest_objects = objs
                    self.latest_objects_ts = float(obj.get("ts", time.time()))
            elif kind == "relocalize_status":
                self.latest_reloc_status = {k: v for k, v in obj.items() if k != "type"}
                self.latest_reloc_status_ts = time.time()
            elif kind == "ping":
                await ws.send(json.dumps({"type": "pong", "t": time.time()}))
