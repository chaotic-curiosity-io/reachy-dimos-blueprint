"""Fail-closed DimOS navigation for the Reachy mecanum chassis.

The continuous L515 mapper owns localization and the accumulated point map.
This process levels that optical map into a conventional x-forward, y-left,
z-up navigation frame, builds a DimOS occupancy grid, runs DimOS's replanning
A* planner, and (only with ``--execute``) translates its Twist output into
short ESP32 deadman pulses.

Navigation intentionally uses rotate-then-forward motion.  It does not strafe
or reverse, so every translation is observed by the forward-facing L515.

NOT QUALIFIED FOR AUTONOMOUS DRIVING. Without ``--execute`` this is a full
planner dry run that never commands the wheels. ``--execute`` only *requests*
bounded pulses; every pulse is still gated on calibrated L515-to-chassis
extrinsics (``extrinsics_calibrated`` in the mapper report, which no current
calibration provides), fresh localization, observed-free footprint and the
hardware interlocks below. No autonomous goal or obstacle test has passed.

Device addresses: ``--wheels-host`` / ``WHEELS_HOST`` (ESP32 base) and
``--reachy-url`` / ``REACHY_URL`` (wheels app, default
``http://reachy-mini.local:8042``).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import os
import threading
import time
from urllib.request import urlopen

import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation

from dimos.core.global_config import GlobalConfig
from dimos.core.transport import LCMTransport
from dimos.mapping.pointclouds.occupancy import general_occupancy
from dimos.mapping.occupancy.inflation import simple_inflate
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path as DimosPath
from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner
from reachy_wheels_app.wheels_client import WheelsClient, WheelsConfig, WheelsError

from station.l515.reachy_perception import atomic_json


# optical RDF (right/down/forward) -> navigation FLU (forward/left/up)
OPTICAL_TO_NAV = np.array(((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))


@dataclass(frozen=True)
class NavigationConfig:
    resolution_m: float = 0.05
    robot_radius_m: float = 0.23
    rotation_radius_m: float = 0.27
    sensor_forward_m: float = 0.151
    floor_clearance_m: float = 0.04
    obstacle_height_m: float = 1.20
    front_stop_m: float = 0.48
    max_goal_distance_m: float = 3.0
    pulse_speed: float = 0.15
    pulse_seconds: float = 0.06
    pulse_interval_s: float = 0.35
    localization_max_age_s: float = 2.0


def fit_navigation_frame(points_optical: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Return a fixed leveling rotation, floor z and supporting point count."""
    points = np.asarray(points_optical, dtype=float)
    nominal = points @ OPTICAL_TO_NAV.T
    candidates = nominal[np.isfinite(nominal).all(axis=1) & (nominal[:, 2] < 0.15)]
    if len(candidates) < 500:
        raise ValueError("not enough downward geometry to estimate the floor")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(candidates))
    best = None
    # A close wall can contain more points than the visible floor. Peel off
    # dominant planes until the largest gravity-compatible candidate emerges.
    for _ in range(8):
        if len(cloud.points) < 300:
            break
        model, indices = cloud.segment_plane(
            distance_threshold=0.025, ransac_n=3, num_iterations=1000
        )
        normal = np.asarray(model[:3], dtype=float)
        offset = float(model[3])
        if normal[2] < 0:
            normal, offset = -normal, -offset
        normal /= np.linalg.norm(normal)
        tilt = math.degrees(math.acos(float(np.clip(normal[2], -1, 1))))
        if tilt <= 25 and len(indices) >= 300 and (best is None or len(indices) > best[3]):
            best = normal, offset, tilt, len(indices)
        cloud = cloud.select_by_index(indices, invert=True)
    if best is None:
        raise ValueError("no gravity-compatible floor plane has enough support")
    normal, offset, tilt, support = best
    level = Rotation.align_vectors([[0.0, 0.0, 1.0]], [normal])[0].as_matrix()
    floor_z = -offset
    return level, floor_z, support


def transform_points(points_optical: np.ndarray, level: np.ndarray) -> np.ndarray:
    return np.asarray(points_optical) @ OPTICAL_TO_NAV.T @ np.asarray(level).T


def transform_pose(
    sensor_pose_optical: np.ndarray,
    level: np.ndarray,
    *,
    frame_id: str,
    sensor_forward_m: float = 0.151,
    ts: float = 0.0,
) -> PoseStamped:
    """Convert the mapped L515 pose to a planar base pose."""
    transform = np.asarray(sensor_pose_optical, dtype=float)
    basis = np.asarray(level) @ OPTICAL_TO_NAV
    rotation = basis @ transform[:3, :3] @ basis.T
    translation = basis @ transform[:3, 3]
    yaw = float(Rotation.from_matrix(rotation).as_euler("xyz")[2])
    base_xy = translation[:2] - sensor_forward_m * np.array([math.cos(yaw), math.sin(yaw)])
    return PoseStamped(
        ts=ts,
        frame_id=frame_id,
        position=[float(base_xy[0]), float(base_xy[1]), 0.0],
        orientation=Rotation.from_euler("z", yaw).as_quat().tolist(),
    )


def make_costmaps(
    points_nav: np.ndarray,
    pose: PoseStamped,
    floor_z: float,
    config: NavigationConfig,
    *,
    frame_id: str,
    ts: float,
    current_points_nav: np.ndarray | None = None,
    sensor_origin_nav: np.ndarray | None = None,
) -> tuple[OccupancyGrid, OccupancyGrid]:
    """Build display and conservative planning grids using DimOS messages."""
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

    # Only surface endpoints provide evidence of supported floor. A ray above
    # the floor does not rule out low obstacles, cables, or a drop underneath.
    # The append-only scene map retains people and displaced surfaces. It is
    # useful memory, not a current collision map. Never union that history into
    # a fresh observation: doing so makes observed floor permanently occupied.
    points_nav = np.asarray(current_points_nav if current_points_nav is not None else points_nav)
    points_nav = points_nav[np.isfinite(points_nav).all(axis=1)]
    points_nav = points_nav[points_nav[:, 2] >= floor_z - 0.025]
    cloud = PointCloud2.from_numpy(points_nav, frame_id=frame_id, timestamp=ts)
    display = general_occupancy(
        cloud,
        resolution=config.resolution_m,
        min_height=floor_z + config.floor_clearance_m,
        max_height=floor_z + config.obstacle_height_m,
        mark_free_radius=0.0,
        frame_id=frame_id,
    )
    # Inflate unknown as well as occupied space: the entire footprint, not
    # merely its center, must lie on observed floor. No invented start bubble.
    blocked = OccupancyGrid(
        grid=np.where(display.grid == CostValues.FREE, CostValues.FREE, CostValues.OCCUPIED),
        resolution=display.resolution, origin=display.origin, frame_id=frame_id, ts=ts,
    )
    inflated = simple_inflate(blocked, config.robot_radius_m)
    planning = OccupancyGrid(
        grid=np.where(inflated.grid == CostValues.UNKNOWN, CostValues.OCCUPIED, inflated.grid),
        resolution=inflated.resolution,
        origin=inflated.origin,
        frame_id=inflated.frame_id,
        ts=inflated.ts,
    )
    return display, planning


def footprint_clear(costmap: OccupancyGrid, pose: PoseStamped, radius_m: float) -> bool:
    center = costmap.world_to_grid(pose.position)
    radius = int(math.ceil(radius_m / costmap.resolution))
    x0, y0 = int(center.x), int(center.y)
    y1, y2 = max(0, y0 - radius), min(costmap.height, y0 + radius + 1)
    x1, x2 = max(0, x0 - radius), min(costmap.width, x0 + radius + 1)
    if x0 - radius < 0 or y0 - radius < 0 or x0 + radius >= costmap.width or y0 + radius >= costmap.height:
        return False
    yy, xx = np.indices((y2 - y1, x2 - x1))
    circle = (xx + x1 - x0) ** 2 + (yy + y1 - y0) ** 2 <= radius**2
    cells = costmap.grid[y1:y2, x1:x2][circle]
    return bool(len(cells) and np.all(cells == CostValues.FREE))


def choose_frontier(costmap: OccupancyGrid, pose: PoseStamped, radius_m: float,
                    planning_costmap: OccupancyGrid | None = None) -> tuple[float, float] | None:
    """Choose a footprint-safe free standoff near a mapped frontier."""
    free = costmap.grid == CostValues.FREE
    unknown = costmap.grid == CostValues.UNKNOWN
    safe = ndimage.distance_transform_edt(free) * costmap.resolution
    distance_to_unknown = ndimage.distance_transform_edt(~unknown) * costmap.resolution
    standoff = free & (safe >= radius_m) & (distance_to_unknown <= radius_m + 0.20)
    if planning_costmap is not None:
        center=planning_costmap.world_to_grid(pose.position)
        cx,cy=int(center.x),int(center.y)
        traversable=planning_costmap.grid<CostValues.OCCUPIED
        labels,_=ndimage.label(traversable,structure=np.ones((3,3),bool))
        if not (0<=cx<planning_costmap.width and 0<=cy<planning_costmap.height):return None
        component=labels[cy,cx]
        if component==0:return None
        standoff &= labels==component
    candidates = np.argwhere(standoff)
    if not len(candidates):
        return None
    wx = costmap.origin.position.x + candidates[:, 1] * costmap.resolution
    wy = costmap.origin.position.y + candidates[:, 0] * costmap.resolution
    distance = np.hypot(wx - pose.position.x, wy - pose.position.y)
    allowed = (distance >= 0.45) & (distance <= 2.5)
    if not np.any(allowed):
        return None
    candidates, wx, wy, distance = candidates[allowed], wx[allowed], wy[allowed], distance[allowed]
    # Prefer progress over tiny frontier oscillations, while staying within
    # the conservative local-map range.
    index = int(np.argmax(distance))
    return float(wx[index]), float(wy[index])


def remove_redundant_start_pose(path: DimosPath, current: PoseStamped | None) -> DimosPath:
    """Compatibility shim for DimOS local planner's coincident-start branch.

    The installed planner treats a path whose first point equals odometry as
    already at the final-rotation phase. A* normally includes that start cell,
    so remove only that redundant point before the planner consumes the path.
    """
    if current is not None and len(path.poses) > 1:
        if path.poses[0].position.distance(current.position) < 0.075:
            path.poses.pop(0)
    return path


class NavigationRuntime:
    def __init__(self, directory: Path, execute: bool, config: NavigationConfig,
                 wheels_host: str | None = None, reachy_url: str | None = None):
        self.directory = directory
        self.execute = execute
        self.config = config
        self.reachy_url = (reachy_url or os.environ.get("REACHY_URL", "http://reachy-mini.local:8042")).rstrip("/")
        self.level: np.ndarray | None = None
        self.floor_z: float | None = None
        self.segment: str | None = None
        self.pose: PoseStamped | None = None
        self.display_costmap: OccupancyGrid | None = None
        self.planning_costmap: OccupancyGrid | None = None
        self.mapping: dict = {}
        self.latest_twist = Twist()
        self.latest_path: list[list[float]] = []
        self.goal: dict | None = None
        self.exploring = False
        # A supervisor restart must never replay an old movement request.
        command_path = directory / "navigation-command.json"
        self.last_command_mtime = command_path.stat().st_mtime_ns if command_path.exists() else 0
        self.last_map_mtime = 0
        self.last_mapping_report = 0.0
        self.last_pulse = 0.0
        self.last_pulse_accepted = -1
        self.last_action = "idle"
        self.blocked_reason = "waiting for map"
        self.motion_sent = False
        self.halt = threading.Event()
        # Host from --wheels-host or WHEELS_HOST (the wheels client's own default).
        board_cfg = WheelsConfig(timeout=2, retries=1)
        if wheels_host:
            board_cfg.host = wheels_host
        self.board = WheelsClient(board_cfg)
        g = GlobalConfig(
            # Obstacles are already inflated by the physical footprint before
            # unknown cells are blocked. Keep DimOS's second inflation to one
            # grid cell rather than applying the footprint twice.
            robot_width=config.resolution_m,
            robot_rotation_diameter=config.resolution_m,
            nerf_speed=0.25,
            viewer="none",
        )
        self.planner = GlobalPlanner(g)
        # The stock threshold assumes a much faster Go2.  Reachy deliberately
        # advances in millimetre-scale stop-and-observe pulses.
        from dimos.navigation.replanning_a_star.position_tracker import PositionTracker

        self.planner._position_tracker = PositionTracker(8.0, 0.025)
        self.planner.path.subscribe(self._on_path)
        self.planner.cmd_vel.subscribe(self._on_twist)
        self.planner.goal_reached.subscribe(self._on_goal_reached)
        self.transports = {
            "odom": LCMTransport("/reachy/navigation/odom", PoseStamped),
            "costmap": LCMTransport("/reachy/navigation/global_costmap", OccupancyGrid),
            "path": LCMTransport("/reachy/navigation/path", DimosPath),
            "cmd": LCMTransport("/reachy/navigation/cmd_vel", Twist),
        }

    def _on_path(self, path: DimosPath) -> None:
        remove_redundant_start_pose(path, self.pose)
        self.latest_path = [[float(p.position.x), float(p.position.y)] for p in path.poses]
        self.transports["path"].publish(path)

    def _on_twist(self, twist: Twist) -> None:
        self.latest_twist = twist
        self.transports["cmd"].publish(twist)

    def _on_goal_reached(self, message) -> None:
        if bool(message.data):
            self.last_action = "goal reached"
            self.goal = None
            if self.exploring:
                self._start_frontier_goal()
        else:
            self.last_action = "goal cancelled: no safe connected path"
            self.goal = None
            self.exploring = False

    def _frame_path(self) -> Path:
        return self.directory / "navigation-frame.json"

    def _load_or_fit_frame(self, segment: str, points: np.ndarray) -> None:
        path = self._frame_path()
        try:
            saved = json.loads(path.read_text())
            if saved["segment"] == segment:
                self.level = np.asarray(saved["level_rotation"], dtype=float)
                self.floor_z = float(saved["floor_z_m"])
                return
        except (OSError, ValueError, KeyError):
            pass
        level, floor_z, support = fit_navigation_frame(points)
        self.level, self.floor_z = level, floor_z
        atomic_json(path, {
            "segment": segment,
            "level_rotation": level.tolist(),
            "floor_z_m": floor_z,
            "floor_support_points": support,
            "axis_convention": "x forward, y left, z up; origin at first L515 pose",
            "sensor_forward_m": self.config.sensor_forward_m,
            "estimated": True,
        })

    def update_map(self) -> None:
        mapping_path = self.directory / "continuous-mapping.json"
        active_path = self.directory / "continuous-active.json"
        mapping = json.loads(mapping_path.read_text())
        segment = json.loads(active_path.read_text())["segment"]
        saved = self.directory / "colored-map-segments" / f"{segment}.npz"
        mtime = saved.stat().st_mtime_ns
        self.mapping = mapping
        with np.load(saved, allow_pickle=False) as data:
            points = data["points"].copy()
        if self.segment != segment:
            if self.segment is not None:
                self.stop("map segment changed; a new goal is required")
            self.level = self.floor_z = None
        self.segment = segment
        self._load_or_fit_frame(segment, points)
        assert self.level is not None and self.floor_z is not None
        self.update_pose(mapping,segment)
        assert self.pose is not None
        frame_id = self.pose.frame_id
        stamp = self.pose.ts
        points_nav = transform_points(points, self.level)
        current_points_nav=None;sensor_origin_nav=None
        try:
            from station.l515.persistent_rgbd import depth_points, transform_points as register_points
            with np.load(self.directory / "mapping-frame.npz", allow_pickle=False) as data:
                handoff=json.loads(str(data["handoff"]));depth=data["depth"]
            if (handoff.get("segment")==segment and handoff.get("map_pose") is not None
                    and 0 <= time.monotonic()-handoff["depth_time"] < self.config.localization_max_age_s):
                accepted_pose=np.asarray(handoff["map_pose"],dtype=float)
                current_optical=register_points(depth_points(depth,handoff["meta"]),accepted_pose)
                current_points_nav=transform_points(current_optical,self.level)
                sensor_origin_nav=(self.level @ OPTICAL_TO_NAV) @ accepted_pose[:3,3]
        except (OSError,ValueError,KeyError):
            pass
        if current_points_nav is None:
            raise ValueError("fresh accepted LiDAR scan is required for the live costmap")
        self.display_costmap, self.planning_costmap = make_costmaps(
            points_nav, self.pose, self.floor_z, self.config, frame_id=frame_id, ts=stamp,
            current_points_nav=current_points_nav,sensor_origin_nav=sensor_origin_nav,
        )
        self.planner.handle_global_costmap(self.planning_costmap)
        self.transports["costmap"].publish(self.display_costmap)
        self.last_map_mtime = mtime
        self._save_costmap_image()

    def update_pose(self,mapping:dict,segment:str) -> None:
        if self.level is None:return
        self.mapping=mapping
        stamp=float(mapping.get("source_received_at_unix",time.time()))
        self.pose=transform_pose(np.asarray(mapping["sensor_pose_matrix"]),self.level,
            frame_id="reachy_nav_"+segment,sensor_forward_m=self.config.sensor_forward_m,ts=stamp)
        self.planner.handle_odom(self.pose)
        self.transports["odom"].publish(self.pose)
        self.last_mapping_report=float(mapping.get("reported_at_unix",0))

    def _save_costmap_image(self) -> None:
        assert self.display_costmap is not None
        grid = self.display_costmap.grid
        rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
        rgb[grid == CostValues.UNKNOWN] = (35, 42, 52)
        rgb[grid == CostValues.FREE] = (224, 232, 238)
        rgb[grid == CostValues.OCCUPIED] = (220, 65, 65)
        if self.pose is not None:
            cell = self.display_costmap.world_to_grid(self.pose.position)
            x, y = int(cell.x), int(cell.y)
            if 0 <= x < self.display_costmap.width and 0 <= y < self.display_costmap.height:
                rgb[max(0, y-2):y+3, max(0, x-2):x+3] = (35, 190, 120)
        for xw, yw in self.latest_path:
            cell = self.display_costmap.world_to_grid((xw, yw, 0.0))
            x, y = int(cell.x), int(cell.y)
            if 0 <= x < self.display_costmap.width and 0 <= y < self.display_costmap.height:
                rgb[y, x] = (45, 125, 255)
        temp = self.directory / "navigation-costmap.tmp.png"
        Image.fromarray(np.flipud(rgb)).resize((rgb.shape[1] * 3, rgb.shape[0] * 3)).save(temp)
        temp.replace(self.directory / "navigation-costmap.png")

    def _goal_is_safe(self, x: float, y: float) -> tuple[bool, str | None]:
        if self.display_costmap is None or self.pose is None:
            return False, "costmap is not ready"
        if not all(map(math.isfinite, (x, y))):
            return False, "goal must be finite"
        if math.hypot(x - self.pose.position.x, y - self.pose.position.y) > self.config.max_goal_distance_m:
            return False, f"goal exceeds {self.config.max_goal_distance_m:.1f} m local limit"
        cell = self.display_costmap.world_to_grid((x, y, 0.0))
        gx, gy = int(cell.x), int(cell.y)
        if not (0 <= gx < self.display_costmap.width and 0 <= gy < self.display_costmap.height):
            return False, "goal is outside the mapped area"
        if self.display_costmap.grid[gy, gx] != CostValues.FREE:
            return False, "goal is not on observed free floor"
        clearance = ndimage.distance_transform_edt(
            self.display_costmap.grid == CostValues.FREE
        )[gy, gx] * self.display_costmap.resolution
        if clearance < self.config.rotation_radius_m:
            return False, "goal lacks a fully observed rotation footprint"
        return True, None

    def set_goal(self, x: float, y: float, yaw_deg: float = 0.0, *, source: str = "operator") -> None:
        ok, reason = self._goal_is_safe(x, y)
        if not ok or not math.isfinite(yaw_deg):
            raise ValueError(reason or "goal yaw must be finite")
        assert self.pose is not None
        goal = PoseStamped(
            frame_id=self.pose.frame_id,
            position=[x, y, 0.0],
            orientation=Rotation.from_euler("z", math.radians(yaw_deg)).as_quat().tolist(),
        )
        self.goal = {"x": x, "y": y, "yaw_deg": yaw_deg, "source": source}
        self.last_action = f"planning {source} goal"
        self.planner.handle_goal_request(goal)

    def _start_frontier_goal(self) -> None:
        if not self.exploring or self.display_costmap is None or self.pose is None:
            return
        target = choose_frontier(self.display_costmap, self.pose, self.config.rotation_radius_m,
                                 self.planning_costmap)
        if target is None:
            self.exploring = False
            self.goal = None
            self.last_action = "exploration complete: no safe local frontier"
            return
        yaw = math.degrees(math.atan2(target[1]-self.pose.position.y, target[0]-self.pose.position.x))
        self.set_goal(*target, yaw, source="frontier")

    def stop(self, reason: str = "stop requested") -> None:
        self.exploring = False
        self.goal = None
        self.latest_twist = Twist()
        self.planner.cancel_goal()
        self.last_action = reason
        if self.execute:
            try:
                self.board.stop()
            except WheelsError as exc:
                self.blocked_reason = f"STOP delivery failed: {exc}"

    def handle_command(self) -> None:
        path = self.directory / "navigation-command.json"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return
        if mtime == self.last_command_mtime:
            return
        self.last_command_mtime = mtime
        command = json.loads(path.read_text())
        action = command.get("action")
        if action != "stop" and not 0 <= time.time() - float(command.get("requested_at_unix", 0)) <= 5:
            raise ValueError("expired navigation command; submit a new goal")
        if action == "stop":
            self.stop("operator stop")
        elif action == "goal":
            self.exploring = False
            self.set_goal(float(command["x"]), float(command["y"]), float(command.get("yaw_deg", 0)))
        elif action == "explore":
            self.exploring = True
            self._start_frontier_goal()
        else:
            raise ValueError(f"unknown navigation action {action!r}")

    @staticmethod
    def _http_json(url: str, timeout: float = 2.0) -> dict:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())

    def safety_reason(self, twist: Twist) -> str | None:
        if not all(math.isfinite(v) for v in (twist.linear.x, twist.linear.y, twist.linear.z,
                                              twist.angular.x, twist.angular.y, twist.angular.z)):
            return "non-finite velocity command"
        # Mapping currently explicitly reports this false. Do not interpret
        # an estimated sensor offset / commanded body angle as calibration.
        if self.execute and not self.mapping.get("extrinsics_calibrated", False):
            return "physical L515-to-chassis calibration is not validated"
        if self.pose is None or self.display_costmap is None:
            return "navigation map is unavailable"
        now = time.time()
        reported = float(self.mapping.get("reported_at_unix", 0))
        source = float(self.mapping.get("source_received_at_unix", 0))
        if not self.mapping.get("running") or self.mapping.get("state") != "tracking":
            return "L515 localization is not tracking"
        if not (0 <= now - reported <= self.config.localization_max_age_s and
                0 <= now - source <= self.config.localization_max_age_s):
            return "L515 localization is stale"
        if self.mapping.get("segment") != self.segment:
            return "map segment changed"
        if not 0 <= now - self.display_costmap.ts <= self.config.localization_max_age_s:
            return "live costmap is stale"
        if twist.linear.x < -1e-6 or abs(twist.linear.y) > 1e-6:
            return "reverse and lateral autonomous commands are disabled"
        if twist.linear.x > 0 and (self.mapping.get("front_clearance_m") is None or
                                  float(self.mapping["front_clearance_m"]) < self.config.front_stop_m):
            return "forward obstacle inside stopping margin"
        if twist.linear.x > 0:
            yaw = float(self.pose.orientation.euler[2])
            # Check the full short forward sweep, including the current pose.
            # Unknown floor must stop translation as well as rotation.
            for distance in np.linspace(0, 0.05, 4):
                probe = PoseStamped(frame_id=self.pose.frame_id,
                    position=[self.pose.position.x + distance * math.cos(yaw),
                              self.pose.position.y + distance * math.sin(yaw), 0])
                if not footprint_clear(self.display_costmap, probe, self.config.robot_radius_m):
                    return "forward swept footprint is not fully observed free"
        if abs(twist.angular.z) > 1e-6 and not footprint_clear(
            self.display_costmap, self.pose, self.config.rotation_radius_m
        ):
            return "rotation footprint is not fully clear"
        return None

    def _hardware_interlocks(self) -> str | None:
        base = self.reachy_url
        try:
            following = self._http_json(base + "/api/track/status")
            voice = self._http_json(base + "/api/voice/status")
            posture = self._http_json(base + "/api/motion")
            state = self.board.state()
        except (OSError, ValueError, WheelsError) as exc:
            return f"hardware interlock unavailable: {exc}"
        if following.get("active"):
            return "visual following owns motion"
        if voice.get("voice_enabled"):
            return "voice control must be disabled"
        if abs(float(posture.get("body_yaw", 999))) > 3.0:
            return "Reachy body must be centered for the L515/base transform"
        expected = {"front_left": False, "rear_left": False,
                    "front_right": True, "rear_right": True}
        tuning = state.get("tuning") or {}
        if any(bool((tuning.get(name) or {}).get("invert")) != invert for name, invert in expected.items()):
            return "wheel direction tuning does not match the verified configuration"
        if state.get("moving"):
            return "chassis reports motion outside this controller"
        return None

    def actuate(self) -> None:
        twist = self.latest_twist
        reason = self.safety_reason(twist)
        if reason:
            self.blocked_reason = reason
            if self.motion_sent:
                self.stop("safety stop: " + reason)
            return
        if twist.is_zero() or self.goal is None:
            self.blocked_reason = None
            return
        accepted = int(self.mapping.get("accepted", -1))
        if accepted <= self.last_pulse_accepted or time.monotonic() - self.last_pulse < self.config.pulse_interval_s:
            return
        if not self.execute:
            self.blocked_reason = "dry-run: restart with --execute after supervised preflight"
            self.last_action = "would pulse " + ("rotate" if abs(twist.angular.z) > 0.15 else "forward")
            self.last_pulse_accepted = accepted
            self.last_pulse = time.monotonic()
            return
        interlock = self._hardware_interlocks()
        if interlock:
            self.blocked_reason = interlock
            return
        # Network interlocks can take seconds; refresh the localization gate
        # immediately before any command, never reuse the preflight snapshot.
        self.mapping = json.loads((self.directory / "continuous-mapping.json").read_text())
        reason = self.safety_reason(twist)
        if reason:
            self.stop("safety stop: " + reason)
            self.blocked_reason = reason
            return
        try:
            if abs(twist.angular.z) > 0.15:
                self.board.move(omega=1.0 if twist.angular.z > 0 else -1.0,
                                speed=self.config.pulse_speed, duration=self.config.pulse_seconds)
                self.last_action = "rotate pulse"
            elif twist.linear.x > 0:
                self.board.move(vx=1.0, speed=self.config.pulse_speed,
                                duration=self.config.pulse_seconds)
                self.last_action = "forward pulse"
            else:
                return
        except WheelsError as exc:
            self.stop("wheel command failed; new goal required")
            self.blocked_reason = f"wheel command failed: {exc}"
            return
        self.motion_sent = True
        self.blocked_reason = None
        self.last_pulse_accepted = accepted
        self.last_pulse = time.monotonic()

    def report(self, error: str | None = None) -> None:
        pose = None if self.pose is None else {
            "x": self.pose.position.x, "y": self.pose.position.y,
            "yaw_deg": math.degrees(self.pose.orientation.euler[2]),
            "frame_id": self.pose.frame_id,
        }
        grid = self.display_costmap.grid if self.display_costmap is not None else np.array([])
        atomic_json(self.directory / "navigation-status.json", {
            "running": True,
            "execution_enabled": self.execute,
            "hardware_validation": {
                "extrinsics_calibrated": bool(self.mapping.get("extrinsics_calibrated", False)),
                "autonomous_goal_test_passed": False,
                "obstacle_avoidance_test_passed": False,
                "note": "Experimental planner; bounded wheel probes are not autonomous validation.",
            },
            "state": self.planner.get_state().value,
            "goal": self.goal,
            "exploring": self.exploring,
            "pose": pose,
            "path": self.latest_path,
            "last_action": self.last_action,
            "blocked_reason": error or self.blocked_reason,
            "front_clearance_m": self.mapping.get("front_clearance_m"),
            "costmap": {
                "source": "current accepted L515 scan; persistent history is not collision evidence",
                "resolution_m": self.display_costmap.resolution if self.display_costmap else None,
                "free_cells": int(np.sum(grid == CostValues.FREE)),
                "occupied_cells": int(np.sum(grid == CostValues.OCCUPIED)),
                "unknown_cells": int(np.sum(grid == CostValues.UNKNOWN)),
                "unknown_is_blocked_for_planning": True,
            },
            "dimos": {"planner": "GlobalPlanner/Replanning A*", "transport": "LCM"},
            "reported_at_unix": time.time(),
        })

    def run(self) -> None:
        self.planner.start()
        last_report = 0.0
        try:
            while not self.halt.wait(0.05):
                error = None
                try:
                    mapping=json.loads((self.directory / "continuous-mapping.json").read_text())
                    active = json.loads((self.directory / "continuous-active.json").read_text())["segment"]
                    saved = self.directory / "colored-map-segments" / f"{active}.npz"
                    if saved.stat().st_mtime_ns != self.last_map_mtime:
                        self.update_map()
                    elif float(mapping.get("reported_at_unix",0))!=self.last_mapping_report:
                        self.update_map()
                    self.handle_command()
                    self.actuate()
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    error = str(exc)
                    if self.motion_sent:
                        self.stop("navigation error")
                if time.monotonic() - last_report > 0.25:
                    self.report(error)
                    last_report = time.monotonic()
        finally:
            self.stop("navigation process stopped")
            self.planner.stop()
            for transport in self.transports.values():
                transport.stop()
            status_path = self.directory / "navigation-status.json"
            try:
                status = json.loads(status_path.read_text())
            except (OSError, ValueError):
                status = {}
            atomic_json(status_path, {**status, "running": False, "state": "stopped",
                                     "reported_at_unix": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="request bounded wheel pulses (still gated on calibrated "
                             "extrinsics and interlocks; not a qualified autonomous mode). "
                             "Omitted is a full planner dry-run.")
    parser.add_argument("--wheels-host", default=os.environ.get("WHEELS_HOST", ""),
                        help="ESP32 wheel base host (env WHEELS_HOST), e.g. <wheels-ip>")
    parser.add_argument("--reachy-url", default=os.environ.get("REACHY_URL", "http://reachy-mini.local:8042"),
                        help="wheels app base URL on the robot (env REACHY_URL)")
    args = parser.parse_args()
    runtime = NavigationRuntime(args.directory.resolve(), args.execute, NavigationConfig(),
                                wheels_host=args.wheels_host or None, reachy_url=args.reachy_url)
    signal.signal(signal.SIGTERM, lambda *_: runtime.halt.set())
    signal.signal(signal.SIGINT, lambda *_: runtime.halt.set())
    runtime.run()


if __name__ == "__main__":
    main()
