# Reachy + L515 on dimOS: sensors, mapping and (dry-run) navigation

> **Not qualified for autonomous driving.** Native sensor ingestion and
> experimental L515 mapping are implemented and have been exercised on
> hardware. `reachy_navigation.py` is a Reachy-specific dimOS
> costmap/planning/actuation service, and it defaults to a planner-only dry
> run. No autonomous goal or obstacle test has passed. Execution also requires
> calibrated L515-to-chassis extrinsics. The mapper reports these as
> unavailable, because that calibration doesn't exist yet (see
> `perception/calibration/README.md`). Operator permission or the `--execute`
> flag can't stand in for that calibration.

## Hardware

| Component | Address (configure via env) | Role |
| --- | --- | --- |
| Reachy Mini | `pollen@reachy-mini.local`, wheels app HTTP `:8042` (`REACHY_URL`), daemon `:8000` (`REACHY_DAEMON_URL`) | head RGB camera; manual/voice/follow app; single camera owner |
| Raspberry Pi + RealSense L515 | `http://<pi-ip>:8765` (`DEPTH_SERVER_URL`) | 320x240 depth, organized RGB-D bundles, ~7.5 Hz (`perception/depth_server`) |
| ESP32 mecanum base | `<wheels-ip>` (`WHEELS_HOST`) | normalized wheel commands, on-board deadman; no measured odometry (`firmware/wheels`) |
| Station (Mac or DGX Spark) | the dimOS Python environment | mapping, perception, planning, Rerun |

## Run the stack

From the repository root, with the wheels app running on the robot and the Pi
depth streamer up:

```sh
export DEPTH_SERVER_URL=http://<pi-ip>:8765
export WHEELS_HOST=<wheels-ip>
export REACHY_URL=http://reachy-mini.local:8042
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -u -m station.l515.l515_stack --directory ./l515-output
```

`l515_stack.py` sets `PYTHONPATH` for its children to the repo root plus
`robot/wheels_app`. `--assets` defaults to
`perception/depth_server/realsense_viewer/static`, the WebGL viewer that the
depth streamer also serves. `--native` also opens a native Rerun window. It
supervises five processes and restarts any that exit, with bounded backoff:

| Child | Module | Role |
| --- | --- | --- |
| continuous | `continuous_mapping` | L515 scan matching, accumulated coloured map ([PERSISTENT_RGBD.md](PERSISTENT_RGBD.md)) |
| perception | `reachy_rgbd` | YOLO11n-seg on head RGB + articulated depth association ([REACHY_PERCEPTION.md](REACHY_PERCEPTION.md)) |
| rerun | `l515_rerun` | Rerun web viewer (web `:8778`, gRPC `:9878`) |
| viewer | `l515_viewer` | local HTTP UI at `http://127.0.0.1:8777/` (localhost only) |
| navigation | `reachy_navigation` | dimOS costmap + A*; dry run unless `--execute-navigation` |

`stack-status.json` reports children, health and restart counts. Stop the
supervisor with SIGTERM to stop all children. Killing only a child just makes it
restart. The previous map is archived before a mapper restart sets a new
origin.

`--execute-navigation` asks for motor execution, but it doesn't bypass the
calibration and safety gates, and it isn't a qualified operating mode.

## Navigation service

```
accumulated optical map + accepted L515 pose
  -> fixed floor fit and FLU navigation frame
  -> dimOS OccupancyGrid (unknown space blocked)
  -> dimOS GlobalPlanner / replanning A*
  -> guarded rotate-then-forward stop-and-observe pulses
  -> ESP32 board deadman
```

Published topics:

| Topic | Type |
| --- | --- |
| `/reachy/navigation/odom` | `PoseStamped` |
| `/reachy/navigation/global_costmap` | `OccupancyGrid` |
| `/reachy/navigation/path` | `Path` |
| `/reachy/navigation/cmd_vel` | `Twist` |

`http://127.0.0.1:8777/navigation` shows the costmap. From there you can enter
an x/y goal in observed-free space, start conservative local-frontier
exploration, or press STOP. Every pulse requires fresh accepted localization,
forward clearance, a clear rotation footprint, a centred Reachy body, voice and
follow control disabled, stopped wheels, the verified wheel polarity, and
calibrated extrinsics. Any failed gate suppresses motion. Map unknowns are
lethal to the planner, not merely expensive. Navigation only rotates and drives
forward, never strafing or reversing, so the forward-facing L515 observes every
translation.

The costmap never projects elevated rays into floor cells, never erases
historical obstacles, and never invents clearance around the robot. Unknown
space blocks the whole footprint, including rotation and map boundaries. This
can leave the starting location unplannable, because the forward camera can't
see its near, side or rear footprint. That missing evidence needs validated
sensing, not relaxed gates. Old movement requests aren't replayed after a
restart, stale commands expire, and a segment switch or command failure cancels
the goal.

The fixed frame is saved as `navigation-frame.json`, status as
`navigation-status.json`, and the rendered costmap as `navigation-costmap.png`.
The nominal sensor-forward offset and footprint are estimates and must be
physically measured before anyone calls this navigation-grade. No software can
make a forward-only L515 see an obstacle or drop-off behind the chassis. Full
unsupervised coverage needs side/rear ranging or a validated 360-degree pre-scan
procedure.

`navigation_acceptance.py` drives the real HTTP command path against a
**disarmed** service (it refuses an armed one), records whether exploration
produces a path, then stops and checks that the wheels report zero output:

```sh
python -m station.l515.navigation_acceptance --directory ./l515-output \
  --pi http://<pi-ip>:8765 --wheels-host <wheels-ip>
```

## Native sensor module

`reachy_sensors.py` is a standalone dimOS `Module` that publishes the raw
streams over LCM. It takes `Twist` only as a dry-run preview and has no motor
client:

```sh
PYTHONPATH=.:robot/wheels_app python -m station.l515.reachy_sensors \
  --reachy-url http://reachy-mini.local:8042 --depth-url http://<pi-ip>:8765 \
  --seconds 20 --report ./out/report.json --local-scan ./out/local-scan.ply \
  --record ./out/sensors.rrd
```

| Topic | Type | Frame / meaning |
| --- | --- | --- |
| `/reachy/lidar` | `PointCloud2` | `realsense_depth_optical`, metres, right/down/forward |
| `/reachy/color_image` | `Image` | `reachy_head_camera_optical`, BGR |
| `/reachy/cmd_vel_preview` | `Twist` input | SI velocity intent, recorded only, never actuated |

On purpose, there is no `odom` or world-frame `pointcloud` output, so an
unregistered cloud can't be fed into a world mapper by accident. The optional
PLY export passes **one** scan through dimOS `VoxelGrid` (3 cm voxels, column
carving off, optical frame kept). It's a local geometry sample. It isn't SLAM.
The supervisor doesn't run this module, since `continuous_mapping` reads the Pi
directly.

## Experimental L515 localization

```
Pi L515 depth -> robust point-to-plane ICP against a held keyframe
  -> accepted L515 pose in the FIRST optical frame
  -> transformed clouds -> dimOS VoxelGrid -> provisional metric map
Reachy RGB -> independent Image stream
```

`l515_mapping.L515Odometry` is the Open3D registration frontend. Voxel
accumulation, message schemas and transports are dimOS's. Nothing here is
fabricated: no guessed base extrinsic, gravity vector, wheel encoder or
synchronized RGB-D measurement.

A held keyframe stops stationary frame-to-frame noise from building up as a
random walk. It updates after 8 cm or 8 degrees of accepted motion. The matcher
uses three correspondence radii, robust Tukey residuals, minimum point
count/overlap, residual limits, a six-dimensional geometry-condition check, and
low-speed pose-jump limits. It rejects ambiguous single-plane geometry and large
jumps. After a timestamp gap it retries its retained keyframe with a bounded
pose-jump allowance. Ten consecutive rejected scans start a new map segment. The
previous segment is archived and never merged without alignment. There is no
loop closure and no long-range relocalization.

Topics: `/reachy/experimental/sensor_odom` (`PoseStamped` in
`l515_start_optical`), `/reachy/experimental/registered_cloud`, and
`/reachy/experimental/global_map`. The map axes are those of the initial depth
camera (x right, y down, z forward). The pose describes the L515: don't rename
it to `base_link` or treat map z as floor height before the missing transforms
and gravity are estimated.

## Rerun

The page at `http://127.0.0.1:8777/` embeds the Rerun web GUI. Direct URL:
`http://127.0.0.1:8778/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9878%2Fproxy`.
Native viewer on the same stream:

```sh
rerun rerun+http://127.0.0.1:9878/proxy --memory-limit 512MiB
```

The tabs are **RGB**, **Points**, **Voxels** (4 cm `Boxes3D`), **3D segmentation
+ boxes** (plane removal + DBSCAN clusters), and **Status**. Geometric clusters
aren't semantic detections. `geometry cluster 3` describes spatial
connectivity, not a recognized object, and cluster IDs don't persist. The server
keeps a bounded 256 MiB rolling history.

## Supervised motion probes (operator present, clear floor)

`l515_motion_probe.py` sends one low-speed pulse. It requires `--execute`,
rejects speed above 0.25 and duration above 0.25 s, needs a fresh tracking
report, checks that voice and follow are idle and the chassis is stopped, never
retries an uncertain motion, and writes before/after evidence:

```sh
PYTHONPATH=.:robot/wheels_app python -m station.l515.l515_motion_probe \
  --execute --command strafe_right --speed 0.2 --duration 0.15 \
  --mapping-status ./l515-output/continuous-mapping.json \
  --output ./l515-output/probe.json
```

Example scan-matched estimates from 0.20 x 0.15 s pulses: `strafe_right`
13.4 mm, `strafe_left` 8.7 mm, `rotate_ccw` 2.7 degrees. A 30 s stationary
baseline drifted ~0.5 mm. These are scan-matching estimates, **not externally
measured ground truth**. The unequal strafe distances show that no calibrated
velocity model can be inferred from tiny pulses.

`continuous_trial.py` (supervised forward / obstacle-stop qualification) and
`supervised_rgb_follow.py` (person following with a frontal depth stop) are
also operator-supervised, `--execute`-gated routines. Neither is a route
planner. Both read `WHEELS_HOST`, `REACHY_URL` and (for follow)
`DEPTH_SERVER_URL`.

## Why autonomous navigation is not enabled

dimOS's navigation stack provides voxel mapping, costmaps, A* replanning and
exploration. Its Go2 integration supplies world-aligned clouds and robot pose,
and raw L515 optical clouds can't substitute for those. See the
[upstream navigation guide](https://github.com/dimensionalOS/dimos/blob/main/docs/capabilities/navigation/deep_dive.md).

Remaining gates:

1. Measure `base_link -> L515 optical`, plus the Reachy mount and head-camera
   transforms. The known **-90 degree Reachy-to-chassis yaw** is not an L515
   extrinsic.
2. Supply a metric base pose with confidence and lost-tracking detection, then
   transform accepted clouds into `world` before dimOS VoxelGridMapper /
   CostMapper. Don't manufacture odometry by integrating normalized motor
   commands.
3. Measure the chassis's SI velocity response, footprint, braking distance and
   slip. The ESP32 mix normalizes wheel ratios and remaps a minimum duty, so a
   Twist in m/s is not a raw `move(vx, vy, omega)` request.
4. Add a single motion owner, shared with manual, voice and follow control and
   revocable on STOP, with freshness, localization, clearance and
   command-expiry gates. Keep the board deadman.
5. Tune traversability for this small wheeled chassis. Don't inherit Go2's
   climb/underpass defaults. The forward-view L515 can't validate unseen rear
   and side space, stairs, drop-offs, or transparent obstacles.
6. Validate in simulation, then in supervised low-speed physical trials. Enable
   goals and exploration only after tracking-loss and STOP behaviour are shown
   on hardware.

## Deployment notes

The wheels app is the only owner of the Reachy camera on this path. The older
`dimos_scanner` app (mono path, `robot/scanner_app`) must not run alongside it:
both use port 8042 and the SDK. Use `scripts/deploy.sh robot robot/wheels_app`
to install it. Never call the daemon's `/cache/reset-apps`.

Tests (pure tests run anywhere; the dimOS ones skip without the full env):

```sh
python -m pytest station/l515/tests -q
```
