"""Needs the full dimOS environment (see station/README.md); skipped otherwise."""
import pytest
pytest.importorskip('open3d')
pytest.importorskip('scipy')
pytest.importorskip('dimos')
import math
import threading

import numpy as np
from scipy.spatial.transform import Rotation

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.core.global_config import GlobalConfig
from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner
from station.l515.reachy_navigation import (
    NavigationConfig,
    choose_frontier,
    fit_navigation_frame,
    footprint_clear,
    make_costmaps,
    remove_redundant_start_pose,
    transform_points,
    transform_pose,
)


def optical_floor() -> np.ndarray:
    # Optical x=right, y=down, z=forward.  A floor 0.28 m below the camera.
    right, forward = np.meshgrid(np.linspace(-1, 1, 45), np.linspace(0.25, 2.0, 45))
    floor = np.column_stack((right.ravel(), np.full(right.size, 0.28), forward.ravel()))
    wall_y, wall_z = np.meshgrid(np.linspace(-0.2, 0.8, 25), np.linspace(0.4, 1.8, 25))
    wall = np.column_stack((np.full(wall_y.size, 0.65), wall_y.ravel(), wall_z.ravel()))
    return np.vstack((floor, wall))


def test_floor_level_and_axis_conversion():
    points = optical_floor()
    level, floor_z, support = fit_navigation_frame(points)
    nav = transform_points(points, level)
    assert support >= 1500
    assert np.allclose(level, np.eye(3), atol=0.02)
    assert math.isclose(floor_z, -0.28, abs_tol=0.02)
    # Optical forward is navigation +x; optical right is navigation -y.
    assert np.allclose(transform_points([[0.2, 0.28, 1.0]], level)[0], [1, -0.2, -0.28], atol=0.02)
    assert len(nav) == len(points)


def test_pose_conversion_removes_sensor_forward_offset():
    transform = np.eye(4)
    pose = transform_pose(transform, np.eye(3), frame_id="nav", sensor_forward_m=0.15)
    assert np.allclose([pose.position.x, pose.position.y], [-0.15, 0.0])
    transform[:3, :3] = Rotation.from_euler("y", -90, degrees=True).as_matrix()
    pose = transform_pose(transform, np.eye(3), frame_id="nav", sensor_forward_m=0.15)
    assert math.isclose(pose.orientation.euler[2], math.pi / 2, abs_tol=1e-6)
    assert np.allclose([pose.position.x, pose.position.y], [0.0, -0.15], atol=1e-6)


def test_costmap_blocks_unknown_for_planning():
    points = optical_floor()
    level, floor_z, _ = fit_navigation_frame(points)
    pose = PoseStamped(frame_id="nav", position=[0, 0, 0], orientation=[0, 0, 0, 1])
    display, planning = make_costmaps(
        transform_points(points, level), pose, floor_z, NavigationConfig(), frame_id="nav", ts=1
    )
    assert display.free_cells > 0
    assert display.occupied_cells > 0
    assert np.any(display.grid == CostValues.UNKNOWN)
    assert not np.any(planning.grid == CostValues.UNKNOWN)
    assert not footprint_clear(display, pose, 0.15)


def test_overhead_rays_do_not_mark_unobserved_floor_free():
    points = optical_floor()
    level, floor_z, _ = fit_navigation_frame(points)
    pose = PoseStamped(frame_id="nav", position=[0, 0, 0], orientation=[0, 0, 0, 1])
    current = np.array([[1.5, y, floor_z + 1.0] for y in np.linspace(-0.5, 0.5, 100)])
    # Deliberately remove all surface returns around the ray midpoint.
    nav = transform_points(points, level)
    nav = nav[(nav[:, 0] < 0.4) | (nav[:, 0] > 1.0)]
    display, _ = make_costmaps(
        nav, pose, floor_z, NavigationConfig(), frame_id="nav", ts=1,
        current_points_nav=current, sensor_origin_nav=np.array([0.15, 0, 0]),
    )
    midpoint = display.world_to_grid((0.7, 0, 0))
    assert display.grid[int(midpoint.y), int(midpoint.x)] == CostValues.UNKNOWN


def test_live_costmap_uses_current_floor_not_historical_ghosts():
    points = optical_floor()
    level, floor_z, _ = fit_navigation_frame(points)
    accumulated = transform_points(points, level)
    accumulated = np.vstack((accumulated, [0.7, 0.0, floor_z + 0.5]))
    pose = PoseStamped(frame_id="nav", position=[0, 0, 0], orientation=[0, 0, 0, 1])
    current = np.vstack((
        transform_points(points, level),
        np.repeat([[1.5, 0.0, floor_z]], 24, axis=0),
        np.repeat([[1.0, 0.2, floor_z + 0.5]], 24, axis=0),
    ))
    display, _ = make_costmaps(
        accumulated, pose, floor_z, NavigationConfig(), frame_id="nav", ts=1,
        current_points_nav=current, sensor_origin_nav=np.array([0.15, 0, 0]),
    )
    stale = display.world_to_grid((0.7, 0, 0))
    obstacle = display.world_to_grid((1.0, 0.2, 0))
    assert display.grid[int(stale.y), int(stale.x)] == CostValues.FREE
    assert display.grid[int(obstacle.y), int(obstacle.x)] == CostValues.OCCUPIED


def test_current_costmap_does_not_certify_floor_seen_only_in_history():
    level, floor_z, _ = fit_navigation_frame(optical_floor())
    history = transform_points(optical_floor(), level)
    current = history[history[:, 0] > 1.2]
    pose = PoseStamped(frame_id='nav', position=[0, 0, 0])
    display, planning = make_costmaps(history, pose, floor_z, NavigationConfig(),
        frame_id='nav', ts=1, current_points_nav=current)
    cell = display.world_to_grid((0.7, 0, 0))
    assert display.grid[int(cell.y), int(cell.x)] == CostValues.UNKNOWN
    assert planning.grid[int(cell.y), int(cell.x)] == CostValues.OCCUPIED


def test_footprint_rejects_unknown_and_out_of_bounds():
    grid = np.zeros((20, 20), np.int8)
    costmap = OccupancyGrid(grid=grid, resolution=0.1,
        origin=Pose(position=[-1, -1, 0]), frame_id="nav")
    pose = PoseStamped(frame_id="nav", position=[0, 0, 0])
    assert footprint_clear(costmap, pose, 0.2)
    costmap.grid[10, 11] = CostValues.UNKNOWN
    assert not footprint_clear(costmap, pose, 0.2)
    costmap.grid[10, 11] = CostValues.FREE
    pose.position.x = -0.95
    assert not footprint_clear(costmap, pose, 0.2)


def test_execution_requires_validated_extrinsics():
    from station.l515.reachy_navigation import NavigationRuntime
    from dimos.msgs.geometry_msgs.Twist import Twist
    runtime = object.__new__(NavigationRuntime)
    runtime.execute = True
    runtime.mapping = {"extrinsics_calibrated": False}
    assert "calibration" in runtime.safety_reason(Twist())
    twist = Twist()
    twist.linear.x = float("nan")
    assert "non-finite" in runtime.safety_reason(twist)


def test_safety_rejects_future_stale_and_untracked_poses():
    import time
    from station.l515.reachy_navigation import NavigationRuntime
    from dimos.msgs.geometry_msgs.Twist import Twist
    runtime = object.__new__(NavigationRuntime)
    runtime.execute = False
    runtime.config = NavigationConfig()
    runtime.segment = "s"
    runtime.pose = PoseStamped(position=[0, 0, 0])
    runtime.display_costmap = OccupancyGrid(grid=np.zeros((40, 40), np.int8),
        resolution=0.05, origin=Pose(position=[-1, -1, 0]), ts=time.time())
    runtime.mapping = dict(running=True, state="tracking", segment="s",
        reported_at_unix=time.time(), source_received_at_unix=time.time(), front_clearance_m=1.0)
    assert runtime.safety_reason(Twist()) is None
    runtime.mapping['source_received_at_unix'] = time.time() + 10
    assert "stale" in runtime.safety_reason(Twist())
    runtime.mapping['source_received_at_unix'] = time.time() - 10
    assert "stale" in runtime.safety_reason(Twist())
    runtime.mapping['state'] = 'anchored'
    assert "not tracking" in runtime.safety_reason(Twist())


def test_expired_commands_cannot_start_motion(tmp_path):
    import json
    import pytest
    from station.l515.reachy_navigation import NavigationRuntime
    runtime = object.__new__(NavigationRuntime)
    runtime.directory = tmp_path
    runtime.last_command_mtime = 0
    (tmp_path/'navigation-command.json').write_text(json.dumps(
        dict(action='explore', requested_at_unix=1)))
    with pytest.raises(ValueError, match='expired'):
        runtime.handle_command()


def test_frontier_goal_stays_on_known_free_space():
    grid = np.full((50, 50), CostValues.UNKNOWN, np.int8)
    grid[15:35, 15:35] = CostValues.FREE
    grid[15:35, 35] = CostValues.OCCUPIED
    origin = Pose(position=[-1.25, -1.25, 0], orientation=[0, 0, 0, 1])
    costmap = OccupancyGrid(grid=grid, resolution=0.05, origin=origin, frame_id="nav")
    pose = PoseStamped(frame_id="nav", position=[0, 0, 0], orientation=[0, 0, 0, 1])
    goal = choose_frontier(costmap, pose, 0.1)
    assert goal is not None
    cell = costmap.world_to_grid((*goal, 0))
    assert costmap.grid[int(cell.y), int(cell.x)] == CostValues.FREE


def test_native_dimos_planner_routes_around_obstacle():
    grid = np.zeros((80, 80), np.int8)
    grid[10:70, 40] = CostValues.OCCUPIED
    grid[37:44, 40] = CostValues.FREE
    origin = Pose(position=[-2, -2, 0], orientation=[0, 0, 0, 1])
    costmap = OccupancyGrid(grid=grid, resolution=0.05, origin=origin, frame_id="nav")
    start = PoseStamped(frame_id="nav", position=[-1, 0, 0], orientation=[0, 0, 0, 1])
    goal = PoseStamped(frame_id="nav", position=[1, 0, 0], orientation=[0, 0, 0, 1])
    planner = GlobalPlanner(GlobalConfig(robot_width=0.2, robot_rotation_diameter=0.25,
                                         viewer="none"))
    ready = threading.Event()
    command_ready = threading.Event()
    paths = []
    def receive_path(path):
        remove_redundant_start_pose(path,start)
        if path.poses:
            paths.append(path)
            ready.set()
    planner.path.subscribe(receive_path)
    planner.cmd_vel.subscribe(lambda twist: command_ready.set() if not twist.is_zero() else None)
    planner.start()
    try:
        planner.handle_global_costmap(costmap)
        planner.handle_odom(start)
        planner.handle_goal_request(goal)
        assert ready.wait(3)
        assert command_ready.wait(2)
        assert len(paths[-1].poses) > 2
        assert all(costmap.cell_value(p.position) != CostValues.OCCUPIED for p in paths[-1].poses)
    finally:
        planner.stop()
