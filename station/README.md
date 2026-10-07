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
> downstream of the bridge (depth-model wrappers, ObjectDB, SpatialMemory,
> voxel mapping, LCM transports, message types, costmaps, the replanning A*
> planner) is dimOS's. This directory is the glue that gets a Reachy Mini's
> sensors into it.

## 1. Which dimOS: the fork, pinned

The mono path runs the pipeline script from our dimOS fork:

- **Repo:** <https://github.com/TheWiselyBearded/dimos>
- **Pinned commit:** `0d39b5ad4` (`0d39b5ad47fd0204b38faa5d2be72b64f947ab6b`)
- **Submodule:** `xr-nav` @ `b96c95d8` (`b96c95d83371c042cd23763bec0f1c8cf3716dc6`)

The fork carries four things that upstream `dimensionalOS/dimos` has since
refactored away:

1. `mac_iphone_spatial_foxglove.py`, the end-to-end spatial pipeline script
   (depth -> pose -> ObjectDB -> SpatialMemory -> voxel map). `server.py` loads
   it as a module and swaps in its own `VideoSource`.
2. `viz_backend.py`, the Rerun visualization backend (`--viz rerun|foxglove|both`).
3. `dimos/perception/depth/estimator.py`, the DepthPro and Depth-Anything-3
   estimator wrappers.
4. The `xr-nav` submodule: voxel map, map I/O, keyframes, ICP registration,
   CLI arg groups, and the vendored Depth-Anything-3 source.

```sh
git clone --recurse-submodules https://github.com/TheWiselyBearded/dimos.git
cd dimos && git checkout 0d39b5ad4 && git submodule update --init --recursive
export DIMOS_DIR=$PWD
```

> **Caveat: the `xr-nav` submodule is private at the time of writing.**
> Until it's made public, `--recurse-submodules` (and `git submodule update`)
> fails for anyone outside the project. The top-level fork clones fine, but
> the pipeline won't run without `xr-nav`, as explained below.

**Does the slim mono path (`--depth depthpro`) need `xr-nav` at runtime? Yes.**
We checked the fork's script imports at `0d39b5ad4`:

- `mac_iphone_spatial_foxglove.py` puts `xr-nav/src` on `sys.path` at import
  time. Inside `main()` it then imports `xr_nav.cli_args` **unconditionally**
  while building its argument parser (`add_map_io_args`, `add_keyframe_args`,
  `add_reloc_args`). Right after parsing, again unconditionally, it imports
  `xr_nav.voxel_map`, `xr_nav.map_io`, `xr_nav.keyframe_recorder`,
  `xr_nav.reference_map` and `xr_nav.relocalize_live`. None of this depends on
  `--depth`.
- `server.py` turns on `--registration icp` by default, which adds
  `xr_nav.icp` and `xr_nav.keyframe` (`--no-registration` drops these two).
- Only conditional: `xr_nav.scale_align` (relative DA3 only) and
  `xr_nav.mv_window` (`--mv-window`).
- What DepthPro does **not** need is the vendored Depth-Anything-3 source
  inside `xr-nav` (`awesome-depth-anything-3` on macOS, `Depth-Anything-3`
  elsewhere). The `da3` depth path loads that, and so do the `da3*` robot
  presets.

So the minimal runtime closure for `--depth depthpro` is the pure-Python
`xr_nav` package (numpy / numba / scipy / open3d / opencv), not the DA3 weights
or source. A third party without `xr-nav` access can't run the mono path
today, even with DepthPro. The L515 path (`station/l515`) imports only `dimos.*`
and never touches `xr_nav`.

## 2. Python environment

One environment runs both the dimOS fork and this station code. Python 3.10-3.12
(dimOS requires `<3.13`). Start from the fork's own declarations:

- **Fork `pyproject.toml`:** `pip install -e "$DIMOS_DIR[perception,visualization]"`.
  Core dependencies already include `numpy`, `scipy`, `opencv-contrib-python`,
  `open3d`, `numba`, `rerun-sdk` and `dimos-lcm`. The `perception` extra adds
  `ultralytics` (YOLOE detection), `transformers[torch]` and `chromadb`
  (SpatialMemory's vector store).
- **`$DIMOS_DIR/xr-nav/environment.yml`** (conda env, Python 3.12, with
  `pytorch`, `numba`, `open3d`, `opencv-python`) plus `pip install -e
  "$DIMOS_DIR/xr-nav"`. Private for now, see above.

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
  this all runs on. The fork above only exists to pin a known-good snapshot of
  the spatial pipeline script and its helpers.
- [Apple ml-depth-pro](https://github.com/apple/ml-depth-pro) for monocular
  metric depth.
- [Pollen Robotics](https://github.com/pollen-robotics/reachy_mini) for the
  Reachy Mini, its SDK and the official MJCF model.
- [Rerun](https://rerun.io) for visualization.
