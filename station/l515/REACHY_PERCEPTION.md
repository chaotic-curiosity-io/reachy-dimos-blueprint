# Reachy recognized objects and the dimOS integration path

For the current articulation transform and its limits, see
[ARTICULATED_RGBD.md](ARTICULATED_RGBD.md).

## Components

The `l515_stack.py` supervisor runs `reachy_rgbd.py`. YOLO11n-seg recognizes
objects in the Reachy head-camera image and associates their instance masks
with calibrated L515 depth. The module publishes native dimOS RGB images, 2D
detections and experimental camera-optical 3D surface bounds. Rerun displays the
RGB, labelled objects, depth/RGB alignment, coloured 3D objects, points, voxels
and geometric clusters. `reachy_perception.py` is a standalone 2D-only fallback
(dimOS `Yolo2DDetector`) that the supervisor doesn't run. It also provides the
observation journal helpers that the other modules import.

## Observation memory and read-only agent interface

`observations.sqlite3` stores one observation every five seconds, including
empty frames. It keeps the latest 10,000 (roughly 14 hours of continuous
running). This service persists no images. Each observation has a UUID, source,
time, model and class/box/score list. Repeated detections are repeated
sightings, not distinct persistent physical objects. Records carry optional
camera-optical 3D positions, and 2D-only observations have
`position_3d: null`.

- `GET http://127.0.0.1:8777/api/perception`: latest observation/status
  (including `rgbd_state` / `rgbd_reason`).
- `GET http://127.0.0.1:8777/api/observations?name=cup&limit=20`: recent
  sightings of an exact, case-insensitive class. Omit `name` for all classes.
  The limit is capped at 100.
- `GET http://127.0.0.1:8777/api/localization`: mapping status.

These are usable read-only inputs for an agent. This journal is not dimOS's
SpatialMemory module, and it can't answer where a cup is in the map or supply a
navigation goal. An old sighting doesn't mean the object is still there.

## Running

Model weights are **not** committed. Download them once into the stack output
directory:

```sh
mkdir -p ./l515-output/models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt \
  -o ./l515-output/models/yolo11n-seg.pt
```

Then start the stack as described in [REACHY_NAVIGATION.md](REACHY_NAVIGATION.md).
To run inference standalone, use the same dimOS environment from the repo root:

```sh
DEPTH_SERVER_URL=http://<pi-ip>:8765 \
python -m station.l515.reachy_rgbd --directory ./l515-output
```

The service runs `./l515-output/models/yolo11n-seg.pt` on CPU. It needs the
paired calibration report (`--calibration`, else `$CALIBRATION_REPORT`, else
`<directory>/calibration/paired-extrinsics-report.json`, else
`perception/calibration/paired-extrinsics-report.json`) and the official MJCF
(`--mjcf` / `REACHY_MJCF`, default `station/assets/official_reachy/reachy_mini.xml`).
`--reachy`, `--daemon` and `--pi` (env `REACHY_URL`, `REACHY_DAEMON_URL`,
`DEPTH_SERVER_URL`) override the HTTP addresses. No inference image leaves the
station.

## Live calibrated preview

Native topics:

| Topic | Type | Frame |
| --- | --- | --- |
| `/reachy/color_image` | `Image` | `reachy_head_camera_optical` |
| `/reachy/perception/annotated_image` | `Image` | annotated RGB |
| `/reachy/perception/detections2d` | `Detection2DArray` | image |
| `/reachy/experimental/rgbd_cloud` | `PointCloud2` | Reachy head optical coordinates |
| `/reachy/experimental/detections3d` | `Detection3DArray` | Reachy head optical coordinates |

Empty detections and clouds clear invalidated observations. Consumers must
still expire messages, because a stopped process can't keep sending
invalidations.

Rerun adds **RGB + depth alignment**: depth dots projected through the fitted
camera transform onto the original RGB pixels, with the fitted lens distortion.
It also adds **RGB-coloured 3D objects**: only the depth points inside Reachy's
image, coloured from RGB, with named visible-surface bounds wherever a
segmentation mask has enough consistent depth. This is a current optical view,
not a coloured global map.

Projected depth uses a nearest-sample z-buffer. Masks are eroded by 7 pixels.
Association needs at least 30 depth points and rejects a broad mixed-depth
range. Bounds use the 5th-95th percentiles of the supported visible points:
they estimate a partial surface, not complete object dimensions. Objects without
depth support stay named 2D detections with null 3D positions. Observation
memory records the optional `surface_3d`, frame ID and calibration SHA-256, and
notes that physical scale is estimated (see `perception/calibration/`).

On source, pose or timing failures the native live 3D topics clear, and
recognition continues while RGB is available. Rerun keeps
`rgbd-last-accepted.npz` for inspection, with a status panel showing capture
receipt age, the current rejection reason and an explicit FROZEN label. An old
snapshot is never labelled current and must never drive motion. The Pi's
`/api/rgbd-frame` returns organized depth and metadata without the unused L515
colour image, cutting each live transfer from about 1.08 MB to 156 KB. The full
`/api/calibration-frame` bundles are for calibration captures.

The perception process resolves device hostnames outside the acquisition
window and caches addresses for 60 seconds. Repeated mDNS lookups had made
Reachy JPEG fetches take ~2.4 s, against ~0.35 s at the resolved address.

## Calibration status and remaining alignment work

RGB intrinsics and fixed-head paired extrinsics are fitted
(`perception/calibration/`) and drive this preview. The board scale is still
estimated. Remaining work:

1. Fix and measure the L515 mounting and define the chassis `base_link` axes.
   Measure `base_link -> l515_depth_optical`, both translation and rotation. The
   Reachy-to-chassis yaw alone isn't enough.
2. For the moving head, calibrate the head/camera kinematic chain against
   timestamped actual joint poses (the MJCF lever arm is a model value, not a
   measurement).
3. Measure clock offset and transport latency, use real capture timestamps, and
   pair frames within a measured, motion-dependent tolerance.
4. Check reprojection on held-out target poses, image edges and near/far
   surfaces, and set application tolerances before using this for anything
   beyond visualization.
5. Never assign an arbitrary geometric cluster to a recognized class.

## Route toward the broader dimOS stack

| Capability | Present | Next integration and acceptance gate |
| --- | --- | --- |
| Perception | Native RGB/depth, YOLO labels, Rerun | Open-vocabulary detection after benchmarking; calibrate before publishing semantic `Detection3DArray` as world data |
| Mapping/localization | Experimental L515 ICP + dimOS voxel map, explicit segments | Calibrate gravity/base frame, add loop closure; validate trajectories and recovery without merging unrelated origins |
| Spatial memory | Searchable camera observation journal | Wire dimOS SpatialMemory with timestamped world/base pose and verified object positions |
| Navigation | Dry-run planner | Feed validated registered clouds and base pose into dimOS mapper/costmap/A*; measure footprint and stopping margins |
| Agentic control | Read-only observation, recall and localization APIs | Wrap these as dimOS skills first; add goal tools only through a single revocable motion owner |
| Physical execution | Manual ESP32 control and bounded probes | Calibrate SI velocity and braking, arbitrate manual/voice/follow/agent, expire commands, stop on stale pose/depth |

Candidate upstream integration points: `dimos/perception/spatial_memory_spec.py`
(`query_by_text`, `tag_location`), `dimos/navigation/replanning_a_star/`,
`dimos/navigation/basic_path_follower/` and `dimos/agents/`. These are
candidates, not a claim that every robot blueprint works unchanged. The current
optical sensor origin isn't a navigation-ready world/base pose, and this
perception service has no wheel client and publishes no motion command.
