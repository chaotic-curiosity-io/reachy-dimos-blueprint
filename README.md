# Reachy × dimOS Blueprint

**Build a mobile robot with spatial memory from off-the-shelf parts, powered by
[dimOS](https://github.com/dimensionalOS/dimos).**

This repository is the complete, reproducible record of how we took a
[Reachy Mini](https://github.com/pollen-robotics/reachy_mini) — a desktop robot
with a single mono RGB camera and no mobility — and turned it into a robot that
drives itself around a home, remembers where objects are, and takes natural-
language commands. Every part is commercial off-the-shelf; every line of glue
code is in this repo.

## The whole story in 40 seconds

[![Annotated frame of the robot: Raspberry Pi + camera for vision and tracking, LiDAR depth sensor for point clouds, ESP32 for wireless motor control, motor drivers and mecanum wheels for body rotation and translation](docs/media/reachy-hardware-callouts.jpg)](docs/media/reachy-r3-stack.mp4)

**[▶ Watch the walkthrough](docs/media/reachy-r3-stack.mp4)** — startup scan,
the hardware stack, and the closed *sense → track → move* loop running live.
(GitHub serves repo-hosted video as a download; the clip opens in a new tab.)

## What we built, in order

1. **Bought and assembled a Reachy Mini.** Out of the box it can look around —
   that's it. One mono RGB camera in the head, no depth, no wheels.
2. **Gave the mono camera 3D perception.** A companion machine (a Mac, or an
   NVIDIA DGX Spark — we call it the *station*) runs a depth-estimation and
   pose pipeline on the live head-camera stream: monocular metric depth
   (DepthPro) plus visual odometry, descended from our workbench experiments
   with MASt3R and Depth-Anything-3.
3. **Fed it into dimOS.** The station pipes RGB-D + pose into dimensionalOS's
   own pipeline — ObjectDB, SpatialMemory — unmodified. dimOS gives the robot
   a persistent spatial map with remembered object surfaces; Rerun is the
   viewer. We don't reimplement any of it.
4. **Built open-source wheels.** An ESP32, hobby motor drivers, four mecanum
   wheels and a dedicated battery become a wireless omnidirectional base with
   a tiny HTTP API and a hardware deadman.
5. **Mounted the Reachy on the base** and taught it to drive: person
   following (detector → tracker → control law → head gaze + wheel motion)
   and voice-driven navigation layered on the same loop.
6. **Added true depth.** An Intel RealSense LiDAR depth camera on the Reachy,
   streaming through a Raspberry Pi, replaces estimated depth with measured
   point clouds.
7. **Calibrated camera ↔ depth sensor** with classical computer vision, so
   LiDAR depth projects into the head camera's frame of reference before it
   enters the dimOS pipeline.

The result: a robot assembled from COTS primitives — an RGB camera, a depth
sensor, a set of wheels — that dimOS elevates into spatial memory, perception,
and agentic control.

## Build it

Want one? Start here:

- **[hardware/BOM.md](hardware/BOM.md)** — the full bill of materials. Every
  part is commercial off-the-shelf; the priciest non-robot item is a
  discontinued LiDAR camera bought used.
- **[hardware/ASSEMBLY.md](hardware/ASSEMBLY.md)** — the build guide, in the
  order that works: bench the wheel base alone, mount the Reachy, add the
  depth sensor, calibrate, then drive and scan. Wiring diagram and safety
  notes included.

## dimOS building the map, live

The same Rerun layout at three points in one drive around a room:

| Early in the scan | Clusters forming | Room mostly covered |
| --- | --- | --- |
| ![Rerun with four panels: detector boxes on the head camera, a sparse active map segment, first labelled geometry clusters, RGB-D points projected over the camera image](docs/media/dimos-rerun-early.jpg) | ![The same layout later: ten labelled geometry clusters inside the bounding volume and a denser room point cloud](docs/media/dimos-rerun-mid.jpg) | ![End of scan: a person detected in the camera panel, remembered chair and microwave surfaces persisting in the map, a filled-out room point cloud](docs/media/dimos-rerun-late.jpg) |

Note the labels that persist across all three frames — `chair (remembered
surface)`, `microwave (remembered surface)`. Those are dimOS SpatialMemory
entries outliving the current view: the robot remembers where things are even
when it's no longer looking at them. Full clip:
[reachy-3d-perception.mp4](docs/media/reachy-3d-perception.mp4).

## It moves

| | |
| --- | --- |
| [**Person following**](docs/media/reachy-person-following.mp4) — ONNX detector → tracker → pure control law → head gaze + continuous mecanum motion | [**Voice-driven navigation**](docs/media/reachy-nav-demo.mp4) — a live conversation steering the three-tier *look / turn / drive* motion hierarchy against a live point cloud |

## Architecture

![Dataflow diagram. On the chassis: the Reachy Mini's head camera and robot apps on port 8042, the Raspberry Pi + L515 depth server on port 8765, and the ESP32 mecanum base on port 80 with a 2-second deadman. On the station: the dimOS bridge (WebSocket 9876, 9879 via scan.sh) runs DepthPro, visual odometry, ObjectDB and SpatialMemory into the Rerun viewer, while the L515 stack (web UI 8777, Rerun web 8778, gRPC 9878) builds calibrated RGB-D maps and sends dry-run-gated drive commands back to the base](docs/diagrams/system-architecture.svg)

Three wire protocols hold it together: the robot streams **JPEG frames + head
pose over WebSocket** to the station, the Pi serves **binary point clouds over
pull-based HTTP**, and everything that moves wheels is a **JSON `POST /cmd`**
to the ESP32 — which stops itself two seconds after the last command, no
matter what crashed upstream.

## Quick start

Three run modes, in increasing order of hardware. Environment setup (the dimOS
fork checkout, `DIMOS_DIR`, Python deps) is in [station/README.md](station/README.md).

**1. Zero hardware — synthetic depth server.** Prove the depth add-on's whole
HTTP + browser path with nothing but numpy:

```bash
pip install numpy
cd perception/depth_server
python -m realsense_viewer.server --synthetic --host 127.0.0.1
# open http://127.0.0.1:8765/ — an animated point cloud in WebGL
```

**2. Offline replay — the dimOS pipeline with no robot.** Feed a recording
(`camera.mp4` + `camera_timestamps.jsonl` + `head_pose.jsonl`) to the station
server as a fake robot:

```bash
python -m station.dimos_bridge.server --dimos-dir "$DIMOS_DIR" \
  --depth depthpro --pose external --extra --viz rerun          # terminal 1
python -m station.dimos_bridge.replay_to_bridge /path/to/recording --host 127.0.0.1
```

**3a. Live mono scan — Reachy + station.** Deploy the streamer app, then one
command runs bridge, pipeline and Rerun, and points the robot at your machine:

```bash
./scripts/deploy.sh robot robot/scanner_app reachy-mini.local
DIMOS_DIR=/path/to/dimos ./scripts/scan.sh        # Mac; add --device cuda on NVIDIA
# arrow keys / WASD pan-tilt the head; quit saves a map under ~/.dimos/sessions/
```

> **Caveat:** modes 2 and 3a run the dimOS fork's pipeline, which imports its
> `xr-nav` submodule unconditionally — and that repo is **private at the time
> of writing**, so these paths don't work for third parties yet. Details in
> [station/README.md](station/README.md#1-which-dimos-the-fork-pinned). The
> RGB-D path below needs no `xr-nav`.

**3b. RGB-D mapping + dry-run navigation — full build.** With the depth
add-on attached and calibrated, and the wheels app deployed
(`./scripts/deploy.sh robot robot/wheels_app`):

```bash
export DEPTH_SERVER_URL=http://<pi-ip>:8765 WHEELS_HOST=<wheels-ip>
python -m station.l515.l515_stack --directory ./l515-output
# web UI at http://127.0.0.1:8777/ (embeds the Rerun web viewer, :8778)
```

Navigation plans paths but stays a dry run — see the honest list of what is
and isn't calibrated in
[perception/calibration/README.md](perception/calibration/README.md).

## Repository layout

```
robot/
  scanner_app/     Reachy Mini app: stream head-camera frames + pose to the station
  wheels_app/      Reachy Mini app: drive the base — D-pad, person following, voice
station/
  dimos_bridge/    WebSocket bridge + the dimOS fork pipeline (mono path)
  l515/            RGB-D mapping, perception, dry-run navigation (depth path)
  assets/          Pollen Robotics' official Reachy Mini MJCF model
perception/
  depth_server/    Raspberry Pi service: L515 → point clouds over HTTP
  calibration/     ChArUco head-camera intrinsics + camera↔depth extrinsics
firmware/
  wheels/          ESP32 MicroPython firmware for the mecanum base
hardware/          bill of materials + assembly guide
docs/              diagrams and demo media
scripts/           deploy.sh (apps → robot/sim) · scan.sh (one-command mono scan)
```

## Credits

- [dimOS](https://github.com/dimensionalOS/dimos) — the agentive operating
  system for physical space. This repo exists to show how little you need on
  top of it. The mono pipeline runs via
  [our pinned fork](https://github.com/TheWiselyBearded/dimos), whose `xr-nav`
  submodule is private at the time of writing — see
  [station/README.md](station/README.md#1-which-dimos-the-fork-pinned) for
  what that currently blocks (the L515 path is unaffected).
- [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) by Pollen
  Robotics.
- [Rerun](https://rerun.io) for visualization.
