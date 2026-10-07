"""Reachy Mini -> dimOS bridge (station side).

A WebSocket broker that:

  * accepts ONE robot connection (role="robot") streaming binary JPEG frames
  * accepts MANY controller connections (role="controller") sending JSON
    control messages, which are forwarded verbatim to the robot
  * exposes a thread-safe ``latest_frame()`` accessor so an in-process consumer
    (the dimos pipeline runner) can pull the most recent frame as a BGR ndarray.

The bridge is intentionally library-shaped: ``Bridge.start()`` spawns the WS
server in a background thread and returns. ``Bridge.latest_frame(timeout=...)``
blocks until a frame is available or the deadline elapses.

This way the dimos pipeline runner can do::

    bridge = Bridge()
    bridge.start()
    bgr = bridge.latest_frame(timeout=10.0)   # block until robot connects
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import websockets

from station.dimos_bridge.protocol import (
    DEFAULT_BRIDGE_HOST,
    DEFAULT_BRIDGE_PORT,
    FRAME_MSG_TYPE,
    IMU_MSG_TYPE,
    POSE_MSG_TYPE,
    ImuMsg,
    decode_frame,
    decode_imu,
    decode_pose,
    parse_text,
    peek_msg_type,
    relocalize_status,
)

logger = logging.getLogger("dimos_bridge.bridge")


class Bridge:
    """In-process broker for one robot + many controllers."""

    def __init__(
        self,
        host: str = DEFAULT_BRIDGE_HOST,
        port: int = DEFAULT_BRIDGE_PORT,
    ) -> None:
        self.host = host
        self.port = port

        # Latest frame, mutex-protected so the dimos thread can read while WS
        # writes. We also keep a monotonically-increasing sequence number so
        # consumers can detect "is there a new frame since I last read?"
        # without races.
        self._frame_lock = threading.Lock()
        self._frame_bgr: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._frame_seq: int = 0
        self._frame_event = threading.Event()
        self._frame_cond = threading.Condition(self._frame_lock)

        # IMU state — same shape as the frame state (mutex + cond + seq).
        self._imu_lock = threading.Lock()
        self._imu_latest: Optional[ImuMsg] = None
        self._imu_seq: int = 0
        self._imu_cond = threading.Condition(self._imu_lock)

        # Head poses (4x4 ndarray, body/FLU convention: reachy_base -> head)
        # streamed by the robot. Read by the pipeline's NetworkVideoSource to
        # drive --pose external. Besides the latest sample we keep a
        # timestamped ring buffer so ``pose_at(ts)`` can interpolate the pose
        # at a frame's capture instant — pairing a frame with whatever pose
        # arrived last misassigns rotation by up to the frame period during a
        # pan, which smears the accumulated map. Timestamps are the robot's
        # clock (same clock as frame timestamps, so no cross-machine sync).
        # 512 samples ≈ 25 s of history at the default 20 Hz pose stream.
        self._pose_lock = threading.Lock()
        self._pose_latest: Optional[np.ndarray] = None
        self._pose_track: deque[tuple[float, np.ndarray]] = deque(maxlen=512)
        # Optional subscriber callbacks invoked from the WS thread for each
        # IMU msg. Kept tiny — heavy work (Foxglove publishing) hops through
        # a queue inside the callback itself.
        self._imu_subscribers: list = []
        self._imu_subscribers_lock = threading.Lock()

        # The single robot websocket (None until the robot connects). Guarded
        # by ``_robot_lock`` because controllers forward through it.
        self._robot_ws = None
        self._robot_lock = threading.Lock()

        # Controllers — we forward each incoming control text message to the
        # robot. Controllers don't receive anything from us today, but we keep
        # the set so future features (status broadcasts) have a target.
        self._controllers: set = set()
        self._controllers_lock = threading.Lock()

        # Latest "config" payload the robot sent in its hello — currently
        # contains `depth_model` and anything else the on-robot app wants to
        # tell the Mac. Replaced atomically on each robot (re)connect.
        self._robot_config: dict = {}
        self._robot_config_event = threading.Event()

        # Set by the server to a zero-arg callable; invoked when the robot sends
        # a ``relocalize_request`` (the web button). Runs on the WS thread, so it
        # should return quickly — the server's handler just spawns the sweep.
        self.on_relocalize_request = None

        # asyncio loop running on the server thread (for cross-thread sends).
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Awaited inside ``serve()`` so we can shut the server down gracefully
        # from another thread (vs. abruptly stopping the loop).
        self._shutdown_future: Optional[asyncio.Future] = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._server_thread is not None:
            return
        ready = threading.Event()
        self._server_thread = threading.Thread(
            target=self._run_server, args=(ready,), daemon=True, name="dimos-bridge"
        )
        self._server_thread.start()
        ready.wait(timeout=5.0)
        logger.info("bridge listening on ws://%s:%d", self.host, self.port)

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None and self._shutdown_future is not None:
            fut = self._shutdown_future
            self._loop.call_soon_threadsafe(
                lambda: fut.done() or fut.set_result(None)
            )
        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)

    def latest_frame(self) -> Optional[np.ndarray]:
        """Most recent frame as a BGR ndarray, non-blocking. None if no frame yet."""
        with self._frame_lock:
            return None if self._frame_bgr is None else self._frame_bgr.copy()

    def wait_for_next_frame(
        self, last_seq: int = 0, timeout: float = 2.0
    ) -> tuple[Optional[np.ndarray], int]:
        """Block until a frame with seq > ``last_seq`` arrives.

        Returns ``(bgr_or_None, new_seq)``. Pass the returned seq back in on
        the next call so we never miss or double-deliver a frame.
        """
        deadline = time.monotonic() + timeout
        with self._frame_cond:
            while self._frame_seq <= last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, self._frame_seq
                self._frame_cond.wait(timeout=remaining)
            bgr = None if self._frame_bgr is None else self._frame_bgr.copy()
            return bgr, self._frame_seq

    def wait_for_first_frame(self, timeout: float = 30.0) -> bool:
        return self._frame_event.wait(timeout=timeout)

    def latest_imu(self) -> Optional[ImuMsg]:
        """Most recent IMU sample, non-blocking. None if no sample yet."""
        with self._imu_lock:
            return self._imu_latest

    def latest_pose(self) -> Optional[np.ndarray]:
        """Most recent head pose as a 4x4 ndarray, non-blocking. None if none yet."""
        with self._pose_lock:
            return None if self._pose_latest is None else self._pose_latest.copy()

    def pose_at(self, ts: float) -> Optional[np.ndarray]:
        """Head pose interpolated to robot-clock time ``ts`` (slerp + lerp).

        Queries outside the buffered range clamp to the nearest sample, so a
        frame that arrives before its bracketing pose still gets the closest
        available estimate. None until the first pose arrives.
        """
        with self._pose_lock:
            if not self._pose_track:
                return None if self._pose_latest is None else self._pose_latest.copy()
            track = list(self._pose_track)
        times = [t for t, _ in track]
        if ts <= times[0]:
            return track[0][1].copy()
        if ts >= times[-1]:
            return track[-1][1].copy()
        hi = bisect.bisect_right(times, ts)
        lo = hi - 1
        t0, m0 = track[lo]
        t1, m1 = track[hi]
        dt = t1 - t0
        if dt < 1e-9:
            return m0.copy()
        u = (ts - t0) / dt
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = _slerp_3x3(m0[:3, :3], m1[:3, :3], u)
        out[:3, 3] = (1.0 - u) * m0[:3, 3] + u * m1[:3, 3]
        return out

    def wait_for_next_frame_ts(
        self, last_seq: int = 0, timeout: float = 2.0
    ) -> tuple[Optional[np.ndarray], float, int]:
        """Like ``wait_for_next_frame`` but also returns the frame's capture
        timestamp (robot clock) so the consumer can pair it with ``pose_at``."""
        deadline = time.monotonic() + timeout
        with self._frame_cond:
            while self._frame_seq <= last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, self._frame_ts, self._frame_seq
                self._frame_cond.wait(timeout=remaining)
            bgr = None if self._frame_bgr is None else self._frame_bgr.copy()
            return bgr, self._frame_ts, self._frame_seq

    def wait_for_next_imu(
        self, last_seq: int = 0, timeout: float = 1.0
    ) -> tuple[Optional[ImuMsg], int]:
        """Block until an IMU sample with seq > ``last_seq`` arrives."""
        deadline = time.monotonic() + timeout
        with self._imu_cond:
            while self._imu_seq <= last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, self._imu_seq
                self._imu_cond.wait(timeout=remaining)
            return self._imu_latest, self._imu_seq

    def subscribe_imu(self, callback) -> None:  # type: ignore[no-untyped-def]
        """Register ``callback(imu_msg)`` invoked synchronously per IMU sample.

        The callback runs on the WS server thread, so keep it short — push to
        a queue if you need to do real work.
        """
        with self._imu_subscribers_lock:
            self._imu_subscribers.append(callback)

    def send_control(self, action: str, step_deg: float = 5.0) -> None:
        """Convenience for in-process callers (e.g. an embedded keyboard reader)."""
        import json

        payload = json.dumps(
            {"type": "control", "action": action, "step_deg": step_deg}
        )
        self._forward_to_robot(payload)

    def send_objects(self, objects: list, ts: float) -> None:
        """Forward the live tracked-object list to the robot.

        The dimos pipeline computes objects + their distance from the camera
        and hands them here (via ``mod.OBJECT_LIST_SINK``); the robot's
        dimos_scanner web app reads the latest set over the WS and renders it.
        Fire-and-forget — a missing robot connection just drops the message.
        """
        import json

        payload = json.dumps({"type": "objects", "ts": ts, "objects": objects})
        self._forward_to_robot(payload)

    def send_relocalize_status(self, status: dict) -> None:
        """Forward a relocalization status update to the robot's web app.

        Wired to the pipeline's ``RELOC_STATUS_SINK`` by the server, so each
        progress/result dict ({state, fitness, translation, ...}) reaches the
        robot, which surfaces it on the dimos_scanner settings page.
        """
        self._forward_to_robot(relocalize_status(status))

    # ------------------------------------------------------------------ #
    # WS server                                                           #
    # ------------------------------------------------------------------ #

    def _run_server(self, ready: threading.Event) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._shutdown_future = self._loop.create_future()

        async def serve() -> None:
            async with websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                max_size=8 * 1024 * 1024,
            ):
                ready.set()
                await self._shutdown_future  # resolved by stop()

        try:
            self._loop.run_until_complete(serve())
        except Exception as e:  # noqa: BLE001
            logger.exception("bridge server crashed: %s", e)
        finally:
            ready.set()
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _handle_client(self, ws) -> None:  # type: ignore[no-untyped-def]
        peer = getattr(ws, "remote_address", "?")
        role = await self._handshake(ws)
        if role is None:
            logger.warning("client %s: no/invalid hello, dropping", peer)
            return
        logger.info("client %s connected as %s", peer, role)
        try:
            if role == "robot":
                await self._handle_robot(ws)
            else:
                await self._handle_controller(ws)
        finally:
            logger.info("client %s (%s) disconnected", peer, role)

    async def _handshake(self, ws) -> Optional[str]:  # type: ignore[no-untyped-def]
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            return None
        if not isinstance(msg, str):
            return None
        try:
            obj = parse_text(msg)
        except Exception:  # noqa: BLE001
            return None
        if obj.get("type") != "hello":
            return None
        role = obj.get("role")
        # Capture any config the robot sent (depth_model etc.) so the dimos
        # server can read it before constructing the pipeline argv.
        if role == "robot":
            cfg = obj.get("config") or {}
            if isinstance(cfg, dict):
                self._robot_config = dict(cfg)
                self._robot_config_event.set()
                logger.info("robot hello config: %s", self._robot_config)
        return role

    def wait_for_robot_config(self, timeout: float = 30.0) -> dict:
        """Block until the robot has sent a hello (with optional config dict).

        Returns the config dict (may be empty if the robot didn't include one,
        or if the timeout elapsed before a robot connected).
        """
        self._robot_config_event.wait(timeout=timeout)
        return dict(self._robot_config)

    async def _handle_robot(self, ws) -> None:  # type: ignore[no-untyped-def]
        with self._robot_lock:
            self._robot_ws = ws
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    self._ingest_binary(msg)
                elif isinstance(msg, str):
                    self._ingest_robot_text(msg)
        finally:
            with self._robot_lock:
                if self._robot_ws is ws:
                    self._robot_ws = None

    def _ingest_robot_text(self, payload: str) -> None:
        """Handle JSON text from the robot (currently: relocalize_request)."""
        try:
            obj = parse_text(payload)
        except Exception:  # noqa: BLE001
            return
        if obj.get("type") == "relocalize_request":
            cb = self.on_relocalize_request
            if cb is None:
                logger.info("relocalize_request received but no handler is set "
                            "(this station build does not implement relocalization)")
                return
            try:
                cb()
            except Exception as e:  # noqa: BLE001
                logger.exception("on_relocalize_request failed: %s", e)

    async def _handle_controller(self, ws) -> None:  # type: ignore[no-untyped-def]
        with self._controllers_lock:
            self._controllers.add(ws)
        try:
            async for msg in ws:
                if not isinstance(msg, str):
                    continue
                # Forward verbatim to the robot.
                await self._forward_to_robot_async(msg)
        finally:
            with self._controllers_lock:
                self._controllers.discard(ws)

    # ------------------------------------------------------------------ #
    # Binary ingestion (frames + IMU)                                     #
    # ------------------------------------------------------------------ #

    def _ingest_binary(self, payload: bytes) -> None:
        kind = peek_msg_type(payload)
        if kind == FRAME_MSG_TYPE:
            self._ingest_frame(payload)
        elif kind == IMU_MSG_TYPE:
            self._ingest_imu(payload)
        elif kind == POSE_MSG_TYPE:
            self._ingest_pose(payload)
        else:
            logger.warning("unknown binary msg_type=0x%02x (len=%d)", kind, len(payload))

    def _ingest_frame(self, payload: bytes) -> None:
        try:
            frame_msg = decode_frame(payload)
        except Exception as e:  # noqa: BLE001
            logger.warning("bad binary frame: %s", e)
            return
        arr = np.frombuffer(frame_msg.jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("imdecode returned None")
            return
        with self._frame_cond:
            self._frame_bgr = bgr
            self._frame_ts = frame_msg.ts
            self._frame_seq += 1
            self._frame_cond.notify_all()
        self._frame_event.set()

    def _ingest_imu(self, payload: bytes) -> None:
        try:
            imu = decode_imu(payload)
        except Exception as e:  # noqa: BLE001
            logger.warning("bad imu payload: %s", e)
            return
        with self._imu_cond:
            self._imu_latest = imu
            self._imu_seq += 1
            self._imu_cond.notify_all()
        self._fan_out_imu(imu)

    def _ingest_pose(self, payload: bytes) -> None:
        try:
            pose = decode_pose(payload)
        except Exception as e:  # noqa: BLE001
            logger.warning("bad pose payload: %s", e)
            return
        m = np.asarray(pose.matrix, dtype=np.float64).reshape(4, 4)
        with self._pose_lock:
            self._pose_latest = m
            # The track must stay time-sorted for pose_at()'s bisect; a robot
            # reconnect (or clock step) can send an older ts — reset then.
            if self._pose_track and pose.ts < self._pose_track[-1][0]:
                self._pose_track.clear()
            self._pose_track.append((pose.ts, m))

    def _fan_out_imu(self, imu: ImuMsg) -> None:
        # Snapshot the subscriber list under the lock so a concurrent
        # subscribe() can't trip us mid-iteration.
        with self._imu_subscribers_lock:
            subs = list(self._imu_subscribers)
        for cb in subs:
            try:
                cb(imu)
            except Exception as e:  # noqa: BLE001
                logger.warning("imu subscriber raised: %s", e)

    # ------------------------------------------------------------------ #
    # Cross-thread forwarding                                             #
    # ------------------------------------------------------------------ #

    def _forward_to_robot(self, text: str) -> None:
        """Forward a control message from an in-process caller (e.g. embedded keyboard)."""
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(
            self._forward_to_robot_async(text), self._loop
        )

    async def _forward_to_robot_async(self, text: str) -> None:
        with self._robot_lock:
            ws = self._robot_ws
        if ws is None:
            logger.debug("dropping control message — no robot connected")
            return
        try:
            await ws.send(text)
        except Exception as e:  # noqa: BLE001
            logger.warning("forward to robot failed: %s", e)


# ---------------------------------------------------------------------- #
# Rotation interpolation helpers (for pose_at)                            #
# ---------------------------------------------------------------------- #
# Mirrored from dimos's reachy_replay_spatial_foxglove.py so the bridge has
# no dimos import. Quaternion slerp between rotation matrices, shortest path.

def _slerp_3x3(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    q0 = _rot_to_quat(R0)
    q1 = _rot_to_quat(R1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        q = q / np.linalg.norm(q)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
        s1 = np.sin(t * theta_0) / sin_theta_0
        q = s0 * q0 + s1 * q1
    return _quat_to_rot(q)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z) unit quaternion."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------- #
# CLI for running the bridge stand-alone (no embedded pipeline)           #
# ---------------------------------------------------------------------- #

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_BRIDGE_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_BRIDGE_PORT)
    parser.add_argument(
        "--save-frames",
        type=str,
        default=None,
        help="Optional directory to save incoming JPEGs (for debugging).",
    )
    parser.add_argument(
        "--print-rate", type=float, default=2.0, help="Print FPS every N seconds."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    bridge = Bridge(host=args.host, port=args.port)
    bridge.start()

    save_dir = None
    if args.save_frames:
        import pathlib

        save_dir = pathlib.Path(args.save_frames)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("saving frames to %s", save_dir)

    last_print = time.time()
    last_seq_printed = 0
    seen_seq = 0
    try:
        while True:
            bgr, seen_seq = bridge.wait_for_next_frame(last_seq=seen_seq, timeout=1.0)
            if bgr is None:
                continue
            if save_dir is not None:
                cv2.imwrite(str(save_dir / f"frame_{seen_seq:06d}.jpg"), bgr)
            now = time.time()
            if now - last_print >= args.print_rate:
                fps = (seen_seq - last_seq_printed) / (now - last_print)
                logger.info("rx fps=%.1f total=%d", fps, seen_seq)
                last_print = now
                last_seq_printed = seen_seq
    except KeyboardInterrupt:
        logger.info("stopping bridge")
        bridge.stop()


if __name__ == "__main__":
    main()
