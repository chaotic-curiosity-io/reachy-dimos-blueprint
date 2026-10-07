"""Thin Foxglove WebSocket publisher for Reachy Mini IMU samples.

Runs a ``FoxgloveServer`` on its own port in a background thread, exposing two
JSON-schema channels:

    /reachy/imu       — foxglove.Imu (accel + gyro + orientation)
    /reachy/imu/temp  — scalar temperature in °C

The dimos pipeline already binds the *primary* Foxglove server on port 8765,
so we deliberately use a separate port (default 8766). The operator adds a
second Foxglove Studio connection to see the IMU panels.

This server is optional — instantiate via ``ImuFoxgloveServer().start()`` and
subscribe to the bridge with ``bridge.subscribe_imu(server.publish)``. If the
foxglove-websocket SDK isn't installed, ``ImuFoxgloveServer.start()`` no-ops
with a warning so the rest of the pipeline keeps running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional

logger = logging.getLogger("dimos_bridge.imu_foxglove")


# Foxglove's well-known JSON message schemas. Studio recognizes channels by
# ``schemaName`` and picks the right panel automatically (3D arrow for Imu,
# line plot for scalar floats).
_IMU_SCHEMA = {
    "title": "foxglove.Imu",
    "description": "Inertial Measurement Unit data",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"},
            },
        },
        "frame_id": {"type": "string"},
        "orientation": {
            "type": "object",
            "title": "foxglove.Quaternion",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
        },
        "angular_velocity": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
        "linear_acceleration": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
        },
    },
}

_SCALAR_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"},
            },
        },
        "value": {"type": "number"},
    },
}


class ImuFoxgloveServer:
    """Background-thread Foxglove server with sync ``publish()`` method."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8766,
        frame_id: str = "reachy_imu",
    ) -> None:
        self.host = host
        self.port = port
        self.frame_id = frame_id
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._imu_chan = None
        self._temp_chan = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._available = True

    def start(self) -> bool:
        try:
            import foxglove_websocket  # noqa: F401
        except ImportError:
            logger.warning(
                "foxglove-websocket not installed — IMU Foxglove publisher disabled"
            )
            self._available = False
            return False

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="reachy-imu-foxglove",
        )
        self._thread.start()
        # Give the server up to 3 s to bind so the caller can wire subscribers
        # immediately after start() and not race the channel registration.
        self._ready.wait(timeout=3.0)
        return self._available

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def publish(self, imu_msg) -> None:  # type: ignore[no-untyped-def]
        """Sync entry point — safe to call from any thread (e.g. the bridge WS thread)."""
        if not self._available or self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._publish_async(imu_msg), self._loop)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except RuntimeError as e:
            # Expected during teardown: ``stop()`` calls ``loop.stop()`` which
            # raises "Event loop stopped before Future completed" before the
            # forever-wait in ``_serve()`` returns naturally. Not a real
            # crash — demoting to debug so it doesn't pollute the operator's
            # console at shutdown.
            if "Event loop stopped" in str(e):
                logger.debug("imu foxglove server shut down")
            else:
                logger.exception("imu foxglove server crashed: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.exception("imu foxglove server crashed: %s", e)
        finally:
            self._ready.set()  # unblock start() in case we never bound

    async def _serve(self) -> None:
        from foxglove_websocket.server import FoxgloveServer

        async with FoxgloveServer(
            self.host, self.port, "reachy-imu",
            capabilities=[], supported_encodings=["json"],
        ) as server:
            self._server = server
            self._imu_chan = await server.add_channel({
                "topic": "/reachy/imu",
                "encoding": "json",
                "schemaName": "foxglove.Imu",
                "schema": json.dumps(_IMU_SCHEMA),
            })
            self._temp_chan = await server.add_channel({
                "topic": "/reachy/imu/temp",
                "encoding": "json",
                "schemaName": "ImuTemperature",
                "schema": json.dumps(_SCALAR_SCHEMA),
            })
            logger.info(
                "IMU Foxglove server listening on ws://%s:%d (topics: /reachy/imu, /reachy/imu/temp)",
                self.host, self.port,
            )
            self._ready.set()
            # Idle forever — stop() drops us via loop.stop().
            await asyncio.Event().wait()

    async def _publish_async(self, imu_msg) -> None:  # type: ignore[no-untyped-def]
        if self._server is None or self._imu_chan is None:
            return
        ts_ns = int(imu_msg.ts_ns)
        sec = ts_ns // 1_000_000_000
        nsec = ts_ns % 1_000_000_000
        ax, ay, az = imu_msg.accel
        gx, gy, gz = imu_msg.gyro
        qw, qx, qy, qz = imu_msg.quat
        imu_payload = {
            "timestamp": {"sec": sec, "nsec": nsec},
            "frame_id": self.frame_id,
            # Note Foxglove orders quaternions xyzw, while the SDK gives wxyz.
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "angular_velocity": {"x": gx, "y": gy, "z": gz},
            "linear_acceleration": {"x": ax, "y": ay, "z": az},
        }
        try:
            await self._server.send_message(
                self._imu_chan, ts_ns, json.dumps(imu_payload).encode("utf-8"),
            )
            if self._temp_chan is not None:
                temp_payload = {
                    "timestamp": {"sec": sec, "nsec": nsec},
                    "value": float(imu_msg.temp_c),
                }
                await self._server.send_message(
                    self._temp_chan, ts_ns, json.dumps(temp_payload).encode("utf-8"),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("foxglove send_message failed: %s", e)
