"""Live Reachy/L515 DimOS module. Sensor ingestion and dry-run control only.

Run from the repo root with PYTHONPATH=.:robot/wheels_app in the DimOS Python environment.
No wheel client is imported here; cmd_vel can never actuate hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
from pathlib import Path
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from reachy_wheels_app.sensors import (
    DEFAULT_DEPTH_URL, DEPTH_FRAME, RGB_FRAME, HEADER, DepthSource, validate_url,
)

log = logging.getLogger(__name__)


class ReachySensorConfig(ModuleConfig):
    reachy_url: str = os.environ.get("REACHY_URL", "http://reachy-mini.local:8042")
    depth_url: str = DEFAULT_DEPTH_URL
    depth_hz: float = 5.0
    rgb_hz: float = 2.0
    enable_rgb: bool = True


class ReachySensors(Module):
    """Independent native streams; deliberately no world pointcloud/odom output.

    Connect `lidar` to a calibrated localization frontend before a world mapper.
    `color_image` is the independent movable Reachy head camera, not aligned RGBD.
    """
    config: ReachySensorConfig
    lidar: Out[PointCloud2]
    color_image: Out[Image]
    cmd_vel: In[Twist]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        validate_url(self.config.reachy_url)
        self._depth = DepthSource(self.config.depth_url)
        if not all(np.isfinite(v) and 0 < v <= 30 for v in
                   (self.config.depth_hz, self.config.rgb_hz)):
            raise ValueError("sensor rates must be finite and in (0, 30] Hz")
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._threads = []
        self._last_depth_key = None
        self._latest_points = None
        self._stats = {name: {"frames": 0, "errors": 0, "last_error": None,
                              "last_monotonic": None} for name in ("depth", "rgb")}
        self._last_twist = None

    @rpc
    def start(self):
        super().start()
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self.preview_twist)))
        for name, hz, fn in (("depth", self.config.depth_hz, self.poll_depth),
                             ("rgb", self.config.rgb_hz, self.poll_rgb)):
            if name == "rgb" and not self.config.enable_rgb:
                continue
            thread = threading.Thread(target=self._poll_loop, args=(name, hz, fn),
                                      name=f"reachy-dimos-{name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def _poll_loop(self, name, hz, fn):
        while not self._halt.is_set():
            started = time.monotonic()
            try:
                fn()
            except Exception as exc:
                with self._lock:
                    self._stats[name]["errors"] += 1
                    self._stats[name]["last_error"] = str(exc)
                log.debug("%s source: %s", name, exc)
            self._halt.wait(max(0, 1 / hz - (time.monotonic() - started)))

    def _observed(self, name, metadata):
        with self._lock:
            self._stats[name].update(metadata, last_monotonic=time.monotonic(), last_error=None)
            self._stats[name]["frames"] += 1

    def poll_depth(self):
        payload, metadata = self._depth.frame()
        key = (metadata["sequence"], hashlib.blake2b(payload, digest_size=16).digest())
        if key == self._last_depth_key:
            return
        points = np.frombuffer(payload, dtype="<f4", offset=HEADER.size).reshape(-1, 3).copy()
        cloud = PointCloud2.from_numpy(points, frame_id=DEPTH_FRAME,
                                      timestamp=metadata["received_at_unix"])
        self.lidar.publish(cloud)
        self._last_depth_key = key
        with self._lock:
            self._latest_points = points
        self._observed("depth", metadata)

    def poll_rgb(self):
        with urlopen(self.config.reachy_url.rstrip("/") + "/api/camera", timeout=2) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("RGB image exceeds size limit")
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid Reachy JPEG")
        received = time.time()
        self.color_image.publish(Image.from_opencv(frame, frame_id=RGB_FRAME, ts=received))
        self._observed("rgb", {"received_at_unix": received,
                               "timestamp_kind": "receiver_time_not_capture_time",
                               "frame_id": RGB_FRAME,
                               "width": frame.shape[1], "height": frame.shape[0]})

    @rpc
    def preview_twist(self, twist: Twist):
        command = {"vx_m_s": float(twist.linear.x), "vy_m_s": float(twist.linear.y),
                   "omega_rad_s": float(twist.angular.z), "actuated": False}
        if not all(np.isfinite(command[k]) for k in ("vx_m_s", "vy_m_s", "omega_rad_s")):
            raise ValueError("cmd_vel must be finite")
        with self._lock:
            self._last_twist = command
        return command

    @rpc
    def snapshot(self):
        with self._lock:
            result = {key: dict(value) for key, value in self._stats.items()}
            result["last_cmd_vel"] = self._last_twist
        running = not self._halt.is_set() and any(t.is_alive() for t in self._threads)
        for key in ("depth", "rgb"):
            last = result[key].pop("last_monotonic")
            age = None if last is None else time.monotonic() - last
            result[key]["last_message_age_s"] = age
            result[key]["receiving"] = running and age is not None and age < 2 and not result[key]["last_error"]
        result.update(actuation_enabled=False, synchronized_rgbd=False,
                      world_mapping_ready=False, reported_at_unix=time.time(), running=running)
        return result

    def export_local_scan(self, path):
        """Voxelize ONLY the latest sensor-frame scan with the real DimOS mapper.

        This is metric local geometry, not SLAM or a traversability map. Optical
        axes are retained; no unmeasured transform is silently inserted.
        """
        with self._lock:
            points = None if self._latest_points is None else self._latest_points.copy()
            last = self._stats["depth"]["last_monotonic"]
        if points is None or last is None or time.monotonic() - last > 2:
            raise ValueError("no recent depth scan to export")
        from dimos.mapping.voxels import VoxelGrid
        import open3d as o3d
        grid = VoxelGrid(voxel_size=0.03, block_count=100_000, device="CPU:0",
                         carve_columns=False, frame_id=DEPTH_FRAME)
        try:
            grid.add_frame(PointCloud2.from_numpy(points, frame_id=DEPTH_FRAME))
            cloud = grid.get_global_pointcloud2()
            if not o3d.io.write_point_cloud(str(path), cloud.pointcloud):
                raise OSError(f"could not save {path}")
        finally:
            grid.dispose()

    @rpc
    def stop(self):
        self._halt.set()
        for thread in self._threads:
            thread.join(timeout=8)
        super().stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reachy-url", default=os.environ.get("REACHY_URL", "http://reachy-mini.local:8042"))
    parser.add_argument("--depth-url", default=DEFAULT_DEPTH_URL)
    parser.add_argument("--seconds", type=float, default=0, help="0 runs until Ctrl-C")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--local-scan", type=Path, help="export latest local voxel scan as PLY")
    parser.add_argument("--depth-only", action="store_true", help="RGB supplied by the RGB-D perception process")
    parser.add_argument("--record", type=Path, help="save native Rerun .rrd for inspection")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    from dimos.core.transport import LCMTransport
    module = ReachySensors(reachy_url=args.reachy_url, depth_url=args.depth_url, enable_rgb=not args.depth_only)
    module.lidar.transport = LCMTransport("/reachy/lidar", PointCloud2)
    module.color_image.transport = LCMTransport("/reachy/color_image", Image)
    module.cmd_vel.transport = LCMTransport("/reachy/cmd_vel_preview", Twist)
    unsubscribers = []
    if args.record:
        import rerun as rr
        rr.init("reachy-dimos-sensors", spawn=False)
        rr.save(str(args.record))
        # Separate roots: no false co-registration of head RGB and L515 depth.
        unsubscribers.append(module.lidar.subscribe(lambda msg: rr.log(
            "l515_optical/points", rr.Points3D(np.asarray(msg.pointcloud.points)))))
        unsubscribers.append(module.color_image.subscribe(lambda msg: rr.log(
            "reachy_head/rgb", rr.Image(msg.to_opencv()[:, :, ::-1]))))
    signal.signal(signal.SIGTERM, lambda *_: module._halt.set())
    started = time.monotonic()
    try:
        module.start()
        while not module._halt.is_set() and (not args.seconds or time.monotonic() - started < args.seconds):
            time.sleep(1)
            if args.report:
                temporary = args.report.with_suffix(args.report.suffix + ".tmp")
                temporary.write_text(json.dumps(module.snapshot(), indent=2) + "\n")
                temporary.replace(args.report)
        if args.local_scan:
            module.export_local_scan(args.local_scan)
    except KeyboardInterrupt:
        pass
    finally:
        module.stop()
        report = module.snapshot()
        for unsubscribe in unsubscribers:
            unsubscribe()
        if args.record:
            rr.disconnect()
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if not all(report[name]["frames"] > 0 for name in ("depth", "rgb")):
        raise SystemExit("Both sensor streams did not produce data")


if __name__ == "__main__":
    main()
