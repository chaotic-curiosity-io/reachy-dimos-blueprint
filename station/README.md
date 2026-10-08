# Station: the Mac / DGX side that feeds the robot into dimOS

The *station* is the companion computer: a Mac (Apple silicon, `mps`) or an
NVIDIA machine such as a DGX Spark (`cuda`). It turns what the robot streams
into dimOS inputs. It has two independent paths:

| Path | Code | Input | What dimOS gets |
| --- | --- | --- | --- |
| **Mono RGB** (the original Reachy, no extra hardware) | `station/dimos_bridge/` | head-camera JPEGs + head-pose FK over WebSocket from `robot/scanner_app` | estimated metric depth (DepthPro) + kinematic pose -> ObjectDB, SpatialMemory, voxel map, Rerun |
| **L515 RGB-D** (depth add-on + wheels) | `station/l515/` | measured depth from the Pi (`perception/depth_server`) + head RGB from `robot/wheels_app` | calibrated RGB-D clouds, L515 odometry, dimOS VoxelGrid, OccupancyGrid + A* (dry run) |

> **Built on [dimOS](https://github.com/dimensionalOS/dimos)** by
> Dimensional, the agentive operating system for physical space. Everything
> downstream of the bridge (ObjectDB, SpatialMemory, voxel mapping, LCM
> transports, message types, costmaps, the replanning A* planner) is dimOS's;
> the depth-model wrappers and the ObjectDB extensions come from our dimOS
> fork (section 1). This directory is the glue that gets a Reachy Mini's
> sensors into it.

## 1. Which dimOS: the fork, pinned

The mono path runs the pipeline script from our dimOS fork:

- **Repo:** <https://github.com/TheWiselyBearded/dimos>
- **Pinned commit:** `0d39b5ad4` (`0d39b5ad47fd0204b38faa5d2be72b64f947ab6b`)
- **`xr_nav`:** vendored here in [`vendor/`](vendor/README.md) — you do
  **not** need the fork's `xr-nav` submodule.

The fork carries three fork-only additions (written in the fork, not in
upstream `dimensionalOS/dimos`):

1. `mac_iphone_spatial_foxglove.py`, the end-to-end spatial pipeline script
   (depth -> pose -> ObjectDB -> SpatialMemory -> voxel map). `server.py` loads
   it as a module and swaps in its own `VideoSource`.
2. `viz_backend.py`, the Rerun visualization backend (`--viz rerun|foxglove|both`).
3. `dimos/perception/depth/estimator.py`, the DepthPro and Depth-Anything-3
   estimator wrappers.

```sh
git clone https://github.com/TheWiselyBearded/dimos.git
cd dimos && git checkout 0d39b5ad4
export DIMOS_DIR=$PWD
```

Clone **without** `--recurse-submodules`: the fork's `xr-nav` submodule points
at a private repository, and you don't need it. The pipeline script imports
the `xr_nav` package unconditionally (its argument groups, voxel map, map I/O,
keyframes, relocalization, and the ICP registration `server.py` enables by
default), so the eleven modules it actually reaches are vendored in
[`vendor/xr_nav/`](vendor/README.md), copied verbatim from the pinned
submodule commit. `server.py` puts that copy first on `sys.path` after loading
the fork script, so it's used whether the submodule directory is empty or
populated.

This was verified the way a newcomer would hit it: the pinned fork exported
with an **empty** `xr-nav/` directory, a recorded session replayed through
`replay_to_bridge.py`, and the full mono pipeline ran — DepthPro depth, ICP
registration, a growing voxel map, and object detections — with `xr_nav`
resolved from `station/vendor/`.

The one thing the vendored copy doesn't include is the Depth-Anything-3 source
the submodule also carried. `--depth depthpro` (the default) doesn't need it.
For the `da3*` depth models, install Depth-Anything-3 from
[upstream](https://github.com/ByteDance-Seed/Depth-Anything-3) into the same
environment. The L515 path (`station/l515`) never touches `xr_nav`.

## 2. Python environment

One environment runs both the dimOS fork and this station code. Python 3.10-3.12
(dimOS requires `<3.13`). Start from the fork's own declarations:

- **Fork `pyproject.toml`:** `pip install -e "$DIMOS_DIR[perception,visualization]"`.
  Core dependencies already include `numpy`, `scipy`, `opencv-contrib-python`,
  `open3d`, `numba`, `rerun-sdk` and `dimos-lcm`. The `perception` extra adds
  `ultralytics` (YOLOE detection), `transformers[torch]` and `chromadb`
  (SpatialMemory's vector store).
- **Vendored `xr_nav`:** needs nothing beyond the above — `numpy`, `numba`,
  `scipy`, `open3d`, `opencv`, all already dimOS core dependencies. No install
  step; `server.py` finds it on its own.

On top of those, this directory needs:

| Package | Used by |
| --- | --- |
| `numpy`, `opencv-python`/`opencv-contrib-python` | everything |
| `websockets` | `dimos_bridge/bridge.py`, `replay_to_bridge.py` (not a dimOS dependency) |
| `torch` (MPS or CUDA build) | depth models |
| `depth_pro` (`pip install git+https://github.com/apple/ml-depth-pro.git`) | `--depth depthpro`; put the checkpoint at `$DIMOS_DIR/checkpoints/depth_pro.pt` (the server `chdir`s into `$DIMOS_DIR`) |
| `open3d`, `scipy` | pipeline, L515 mapping/registration |
| `rerun-sdk` | `--viz rerun`, `l515/l515_rerun.py` |
| `dimos_lcm` / LCM | dimOS transports (LCMTransport), via dimOS |
| `ultralytics` | L515 path: `reachy_rgbd.py` (YOLO11n-seg, weights downloaded manually, see `l515/REACHY_PERCEPTION.md`); the mono path's YOLOE detector |
| `foxglove-websocket` | *optional*: IMU side channel (`--imu-foxglove-port`; skipped if missing) |

Run the station code from the repository root (so `station.*` imports
resolve). The L515 modules also need `robot/wheels_app` on the path for the
wheels client and depth decoder:

```sh
export PYTHONPATH=$PWD:$PWD/robot/wheels_app
```

## 3. Three ways to run it

### a) Live scan (mono RGB, robot required)

Deploy and start `dimos_scanner` on the robot (`./scripts/deploy.sh robot
robot/scanner_app`), then:

```sh
DIMOS_DIR=/path/to/dimos PYTHON=python ./scripts/scan.sh            # Mac (mps)
DIMOS_DIR=/path/to/dimos ./scripts/scan.sh --device cuda            # DGX Spark
```

`scan.sh` frees bridge port 9879, autodetects this machine's LAN IP, POSTs it
to the robot app's `/config` (cold-starting the app through the daemon if
needed), and execs:

```sh
python -m station.dimos_bridge.server --dimos-dir "$DIMOS_DIR" \
  --depth depthpro --pose external --ws-port 9879 --device mps \
  --extra --viz rerun --save-map ~/.dimos/sessions/reachy_<ts>.pkl ...
```

Arrow keys / WASD in that terminal pan and tilt the head. Rerun opens by
itself. The bridge uses **9879**, not the protocol default 9876, because
Rerun's gRPC server also defaults to 9876, and any open viewer would grab the
port.

Server knobs worth knowing: `--dimos-dir` / `DIMOS_DIR` (required),
`--ws-port` / `DIMOS_BRIDGE_PORT`, `--device mps|cuda|cpu` / `DIMOS_DEVICE`,
and `--extra ...` (everything after it goes to the dimOS script, so it must
come last). For A/B debugging there are escape hatches:
`REACHY_POSE_RAW=1` (no head->camera extrinsic), `REACHY_POSE_NO_LEVER=1`
(rotation only), `REACHY_POSE_LATEST=1` (latest pose instead of capture-time
interpolation) and `REACHY_POSE_TIME_OFFSET` (seconds).

### b) Offline replay (no robot)

`dimos_bridge/replay_to_bridge.py` plays a recording to the server as a fake
robot, with the original timestamps:

```sh
# terminal 1
python -m station.dimos_bridge.server --dimos-dir "$DIMOS_DIR" \
  --depth depthpro --pose external --extra --viz rerun
# terminal 2
python -m station.dimos_bridge.replay_to_bridge /path/to/recording --host 127.0.0.1
```

Recording layout: `camera.mp4`, `camera_timestamps.jsonl`
(`{"ts": <unix s>, "value": ...}` per frame) and `head_pose.jsonl`
(`{"ts": <unix s>, "value": [16 floats]}`, a row-major 4x4 body->head pose),
plus an optional `metadata.json`. Without `head_pose.jsonl` the server falls
back to visual odometry. The recorder that produced ours isn't part of this
repo. Any tool that writes this layout works.

### c) L515 stack (depth add-on + wheels)

```sh
export DEPTH_SERVER_URL=http://<pi-ip>:8765 WHEELS_HOST=<wheels-ip>
python -m station.l515.l515_stack --directory ./l515-output
```

This supervises continuous mapping, RGB-D perception, Rerun, a local web UI
(`http://127.0.0.1:8777/`) and dimOS navigation **in dry-run mode**. It needs
the calibration report from `perception/calibration/`. Start with
[l515/REACHY_NAVIGATION.md](l515/REACHY_NAVIGATION.md), then
[l515/PERSISTENT_RGBD.md](l515/PERSISTENT_RGBD.md),
[l515/REACHY_PERCEPTION.md](l515/REACHY_PERCEPTION.md) and
[l515/ARTICULATED_RGBD.md](l515/ARTICULATED_RGBD.md). Navigation is **not
qualified for autonomous driving**, and every wheel-moving script requires
`--execute`.

## Layout

```
station/
  dimos_bridge/
    server.py            CLI: bridge + keyboard + dimOS fork pipeline with a network VideoSource
    bridge.py            WebSocket broker (one robot, many controllers), capture-time pose interpolation
    protocol.py          wire format (JSON control + binary frame/IMU/pose), shared with robot/scanner_app
    frames.py            OPT_TO_BODY, T_HEAD_CAM (head->camera extrinsic for the mono path)
    replay_to_bridge.py  fake robot for offline runs
    imu_foxglove.py      optional IMU -> Foxglove side channel
  l515/                  RGB-D mapping, perception, dry-run navigation, supervised probes (+ docs)
  assets/official_reachy/  Pollen Robotics' official Reachy Mini MJCF (Apache-2.0)
  conftest.py            test path setup (repo root + robot/wheels_app)
```

## Tests

```sh
python -m pytest station -q
```

The bridge/protocol tests and most L515 tests are numpy-only. The tests that
need the full dimOS environment (`test_l515_mapping`, `test_l515_rerun`,
`test_reachy_sensors`, `test_reachy_navigation`, `test_continuous_mapping`)
skip themselves via `pytest.importorskip` when `dimos`/`open3d`/`rerun` are
missing.

## Credits

- [dimensionalOS/dimos](https://github.com/dimensionalOS/dimos), the upstream
  this all runs on. The fork above carries our spatial pipeline script, its
  helpers and the ObjectDB extensions, pinned at a known-good commit.
- [Apple ml-depth-pro](https://github.com/apple/ml-depth-pro) for monocular
  metric depth.
- [Pollen Robotics](https://github.com/pollen-robotics/reachy_mini) for the
  Reachy Mini, its SDK and the official MJCF model.
- [Rerun](https://rerun.io) for visualization.
