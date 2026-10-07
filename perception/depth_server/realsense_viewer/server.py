#!/usr/bin/env python3
"""Capture a RealSense depth stream and expose it to a WebGL browser client."""

from __future__ import annotations

import argparse
import io
import zipfile
import uuid
import json
import math
import mimetypes
import signal
import struct
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


STATIC_DIR = Path(__file__).with_name("static")
FRAME_HEADER = struct.Struct("<4sIII")
FRAME_MAGIC = b"RSPC"
FRAME_VERSION = 1


class FrameStore:
    """Thread-safe latest-frame buffer and camera status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._status: Dict[str, Any] = {
            "state": "starting",
            "camera": None,
            "serial": None,
            "firmware": None,
            "resolution": None,
            "depth_scale_m": None,
            "point_count": 0,
            "sequence": 0,
            "capture_fps": 0.0,
            "published_fps": 0.0,
            "last_frame_age_ms": None,
            "error": None,
        }
        self._last_frame_at: Optional[float] = None
        self._calibration = None
        self._calibration_at = None

    def publish_calibration(self, color, depth, metadata):
        with self._lock:
            self._calibration = (color.copy(), depth.copy(), metadata)
            self._calibration_at = time.monotonic()

    def calibration_snapshot(self):
        with self._lock:
            if self._calibration_at is None or time.monotonic()-self._calibration_at > 2:
                return None
            color, depth, metadata = self._calibration
            return color, depth, dict(metadata, server_frame_age_ms=(time.monotonic()-self._calibration_at)*1000)

    def publish(self, payload: bytes, point_count: int, sequence: int) -> None:
        with self._lock:
            self._frame = payload
            self._last_frame_at = time.monotonic()
            self._status.update(
                state="streaming",
                point_count=point_count,
                sequence=sequence,
                error=None,
            )

    def update(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)

    def snapshot(self) -> Tuple[Optional[bytes], Dict[str, Any]]:
        with self._lock:
            status = dict(self._status)
            frame = self._frame
            last_frame_at = self._last_frame_at
        if last_frame_at is not None:
            status["last_frame_age_ms"] = round(
                (time.monotonic() - last_frame_at) * 1000.0, 1
            )
        return frame, status


def color_bmp(rgb):
    """Lossless RGB preview without another Pi imaging dependency."""
    height, width, _ = rgb.shape
    rows = rgb[::-1, :, ::-1].copy().reshape(height, width*3)
    padding = (-width*3) % 4
    if padding:
        rows = np.pad(rows, ((0,0),(0,padding)))
    pixels = rows.tobytes()
    return (struct.pack('<2sIHHI',b'BM',54+len(pixels),0,0,54)
            + struct.pack('<IiiHHIIiiII',40,width,height,1,24,0,len(pixels),0,0,0,0)
            + pixels)


def pack_points(points: np.ndarray, sequence: int) -> bytes:
    points = np.ascontiguousarray(points, dtype="<f4")
    return FRAME_HEADER.pack(
        FRAME_MAGIC, FRAME_VERSION, sequence & 0xFFFFFFFF, len(points)
    ) + points.tobytes()


class RealSenseCapture(threading.Thread):
    def __init__(
        self,
        store: FrameStore,
        stop_event: threading.Event,
        stride: int,
        max_fps: float,
        min_depth: float,
        max_depth: float,
        color: bool = False,
    ) -> None:
        super().__init__(name="realsense-capture", daemon=True)
        self.store = store
        self.stop_event = stop_event
        self.stride = stride
        self.publish_interval = 1.0 / max_fps
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.color = color

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._capture()
            except Exception as exc:
                self.store.update(state="reconnecting", error=str(exc))
                self.stop_event.wait(2.0)

    def _capture(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        # L515 firmware 1.5.2 on the Pi 5 advertises 320x240 but does not
        # deliver frames in that mode. The native 640x480 profile is stable.
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if self.color:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 6)
        profile = pipeline.start(config)

        try:
            device = profile.get_device()
            sensor = device.first_depth_sensor()
            depth_scale = float(sensor.get_depth_scale())
            stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = stream.get_intrinsics()
            session_id = str(uuid.uuid4())
            def intrinsics(p):
                i = p.get_intrinsics()
                return dict(width=i.width,height=i.height,fx=i.fx,fy=i.fy,ppx=i.ppx,ppy=i.ppy,
                            model=str(i.model),coeffs=list(i.coeffs))
            calibration = None
            if self.color:
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                extr = stream.get_extrinsics_to(color_profile)
                calibration = dict(depth_intrinsics=intrinsics(stream),
                    color_intrinsics=intrinsics(color_profile),depth_scale_m=depth_scale,
                    depth_to_color=dict(rotation_column_major=list(extr.rotation),translation_m=list(extr.translation)),
                    serial=device.get_info(rs.camera_info.serial_number),session_id=session_id)
            last_color_number = None

            rows = np.arange(0, intr.height, self.stride, dtype=np.float32)
            cols = np.arange(0, intr.width, self.stride, dtype=np.float32)
            uu, vv = np.meshgrid(cols, rows)
            x_norm = (uu - intr.ppx) / intr.fx
            y_norm = (vv - intr.ppy) / intr.fy

            def info(key: Any) -> Optional[str]:
                try:
                    return device.get_info(key)
                except Exception:
                    return None

            self.store.update(
                state="warming_up",
                camera=info(rs.camera_info.name),
                serial=info(rs.camera_info.serial_number),
                firmware=info(rs.camera_info.firmware_version),
                resolution=f"{intr.width}x{intr.height}@30",
                depth_scale_m=depth_scale,
                error=None,
            )

            sequence = 0
            captured = 0
            published = 0
            stats_started = time.monotonic()
            last_publish = 0.0

            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(5000)
                frame = frames.get_depth_frame()
                if not frame:
                    continue
                if self.color:
                    color_frame = frames.get_color_frame()
                    if color_frame and color_frame.get_frame_number() != last_color_number:
                        metadata = dict(calibration,
                            color_frame_number=color_frame.get_frame_number(),
                            depth_frame_number=frame.get_frame_number(),
                            color_timestamp_ms=color_frame.get_timestamp(),
                            depth_timestamp_ms=frame.get_timestamp(),
                            color_timestamp_domain=str(color_frame.get_frame_timestamp_domain()),
                            depth_timestamp_domain=str(frame.get_frame_timestamp_domain()),
                            received_at_unix=time.time(),
                            color_format='rgb8',depth_format='uint16_le',
                            synchronized_with_reachy=False)
                        self.store.publish_calibration(np.asanyarray(color_frame.get_data()),
                            np.asanyarray(frame.get_data()),metadata)
                        last_color_number = color_frame.get_frame_number()
                captured += 1
                now = time.monotonic()
                if now - last_publish < self.publish_interval:
                    continue

                depth = np.asanyarray(frame.get_data())[:: self.stride, :: self.stride]
                z = depth.astype(np.float32) * depth_scale
                valid = (z >= self.min_depth) & (z <= self.max_depth)
                points = np.empty((int(np.count_nonzero(valid)), 3), dtype=np.float32)
                points[:, 0] = (x_norm * z)[valid]
                points[:, 1] = (y_norm * z)[valid]
                points[:, 2] = z[valid]

                sequence += 1
                published += 1
                self.store.publish(pack_points(points, sequence), len(points), sequence)
                last_publish = now

                elapsed = now - stats_started
                if elapsed >= 2.0:
                    self.store.update(
                        capture_fps=round(captured / elapsed, 1),
                        published_fps=round(published / elapsed, 1),
                    )
                    captured = 0
                    published = 0
                    stats_started = now
        finally:
            pipeline.stop()


class SyntheticCapture(threading.Thread):
    """Development source that exercises the entire browser path without a camera."""

    def __init__(self, store: FrameStore, stop_event: threading.Event, max_fps: float):
        super().__init__(name="synthetic-capture", daemon=True)
        self.store = store
        self.stop_event = stop_event
        self.interval = 1.0 / max_fps

    def run(self) -> None:
        grid = np.linspace(-0.9, 0.9, 100, dtype=np.float32)
        xx, yy = np.meshgrid(grid, grid)
        sequence = 0
        started = time.monotonic()
        self.store.update(
            state="streaming",
            camera="Synthetic test source",
            serial="development",
            firmware="n/a",
            resolution="100x100",
            depth_scale_m=1.0,
        )
        while not self.stop_event.is_set():
            t = time.monotonic() - started
            zz = 1.55 + 0.18 * np.sin(xx * 5.0 + t) * np.cos(yy * 5.0 - t * 0.7)
            points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            sequence += 1
            self.store.publish(pack_points(points, sequence), len(points), sequence)
            self.store.update(capture_fps=10.0, published_fps=10.0)
            self.stop_event.wait(self.interval)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "RealSenseViewer/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/api/calibration-frame", "/api/mapping-frame", "/api/rgbd-frame", "/api/color.bmp"):
            snapshot = self.server.frame_store.calibration_snapshot()
            if snapshot is None:
                self._send_json({"error":"No fresh color/depth pair; enable --color"}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            color, depth, metadata = snapshot
            if path == "/api/mapping-frame":
                color = color[::4, ::4]
                metadata = dict(metadata, color_pixel_stride=4)
            bmp = color_bmp(color) if path != "/api/rgbd-frame" else None
            if path == "/api/color.bmp":
                content, mime = bmp, "image/bmp"
            else:
                buffer = io.BytesIO()
                # These ~212 KiB mapping bundles must not wait for Pi DEFLATE.
                # Lossless uncompressed transport keeps cross-camera pairing timely.
                with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr("metadata.json", json.dumps(metadata))
                    if bmp is not None:
                        archive.writestr("color.bmp", bmp)
                    archive.writestr("depth.u16", np.asarray(depth,dtype='<u2').tobytes())
                content, mime = buffer.getvalue(), "application/zip"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path == "/api/status":
            _, status = self.server.frame_store.snapshot()  # type: ignore[attr-defined]
            self._send_json(status)
            return
        if path == "/api/points":
            frame, _ = self.server.frame_store.snapshot()  # type: ignore[attr-defined]
            if frame is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No depth frame yet")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(frame)
            return
        if path == "/healthz":
            _, status = self.server.frame_store.snapshot()  # type: ignore[attr-defined]
            healthy = status["state"] == "streaming" and (
                status["last_frame_age_ms"] is not None
                and status["last_frame_age_ms"] < 3000
            )
            self._send_json(
                {"ok": healthy, "state": status["state"]},
                HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        relative = "index.html" if path in ("", "/") else path.lstrip("/")
        candidate = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stride", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--max-fps", type=float, default=8.0)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--color", action="store_true", help="Enable paired 640x480 RGB calibration capture at 6 Hz")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.max_fps <= 0:
        parser.error("--max-fps must be positive")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("depth range is invalid")
    return args


def main() -> None:
    args = parse_args()
    store = FrameStore()
    stop_event = threading.Event()
    if args.synthetic:
        capture: threading.Thread = SyntheticCapture(store, stop_event, args.max_fps)
    else:
        capture = RealSenseCapture(
            store,
            stop_event,
            args.stride,
            args.max_fps,
            args.min_depth,
            args.max_depth,
            args.color,
        )
    capture.start()

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.frame_store = store  # type: ignore[attr-defined]
    server.verbose = args.verbose  # type: ignore[attr-defined]

    def shutdown(_signum: int, _frame: Any) -> None:
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"RealSense viewer listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stop_event.set()
        server.server_close()
        capture.join(timeout=6.0)


if __name__ == "__main__":
    main()
