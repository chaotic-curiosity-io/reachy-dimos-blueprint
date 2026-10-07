# Head/body-aware RGB-D

Status: experimental moving preview. The L515 sits on Reachy's rotating body.
The RGB camera is on the head, which moves independently. The transform below
reuses the fixed-pose stereo calibration from `perception/calibration/` and
needs no extra board captures.

## Transform

`articulated_rgbd.py` takes the measured stereo reference (camera <- depth at one
recorded head/body pose) and carries it over to the current actual head/body
poses. The Reachy SDK's analytical kinematics already folds body yaw into the
head pose it reports, and it subtracts a constant head-height offset. So body
yaw must not be applied to that head pose a second time. The constant vertical
origin shift cancels in the relative transform.

The official MJCF `head` and `camera_optical` sites
(`station/assets/official_reachy/reachy_mini.xml`) put the camera about 39.5 mm
forward and 52.5 mm up from the head origin, plus the optical-axis rotation.
This model offset is not a new measured hand-eye calibration. At runtime each
observation records the model SHA-256, the stereo calibration SHA-256, the RGB
frame/session and the transform.

### Two head->camera extrinsics, and which path uses which

The repo has two head->camera lever arms, and they disagree by a few
millimetres:

| Path | Source | Lever arm (forward / up) |
| --- | --- | --- |
| Mono RGB -> dimOS (`station/dimos_bridge/server.py`) | `station/dimos_bridge/frames.py::T_HEAD_CAM`, which mirrors the SDK's `ReachyMini.T_head_cam` | 43.7 mm / 51.2 mm |
| L515 RGB-D (`station/l515/reachy_rgbd.py`) | `articulated_rgbd.model_head_T_camera()` on the official MJCF sites | ~39.5 mm / ~52.5 mm |

The rotation part (optical RDF -> body FLU) is the same in both. Only the
translation differs, by about 4 mm forward and 1 mm up. The mono path uses
`T_HEAD_CAM` to compose the streamed head pose into a camera pose for depth
back-projection. That's where the lever-arm fix removed centimetre-scale smear.
The L515 path uses the MJCF value only *relative to the calibrated reference
pose*: there, the measured stereo extrinsic absorbs any constant error, and the
lever arm only matters for how the camera swings between poses. Neither number
is a measured hand-eye calibration. Pick one, if you ever unify them, by
measuring it, for example with a ChArUco hand-eye solve over several head poses.

## Capture and pairing

The wheels app's `PosedCamera` is the single reader of the robot's IPC camera.
Media, voice and tracking reads all use its recent-image cache. It buffers 20
frames along with actual SDK head/body telemetry. Duplicate cached telemetry
messages don't refresh pose history. The endpoint
`/api/camera/posed-frame?age_ms=...` returns the JPEG, frame/session identity,
the pose interpolated at image time, recent body-pose history and timing
provenance. Pose brackets over 120 ms are rejected.

The station fetches Pi depth first, then asks for a recent RGB frame near that
time. HTTP round trips and each server's monotonic frame age bound the clock
relation. Body yaw is interpolated at the depth time. Head pose is tied to the
RGB time. Pairs are rejected if the RTT is over 400 ms, or if skew plus timing
uncertainty is over 200 ms. These thresholds are preview tolerances, not
navigation accuracy guarantees. When the native 3D output is invalid, it
clears. Rerun keeps a clearly labelled historical snapshot. Recognition can
continue when depth is unavailable.

## Measured limitation

The robot's GStreamer unixfd camera source first reports PTS=0. Later it
reports timestamps with an incompatible clock origin and no reference-timestamp
metadata. So the reader uses **frame arrival time** and sets
`source_capture_timestamp_available=false`. SDK pose messages also lack hardware
timestamps. HTTP uncertainty can't bound unknown upstream camera/telemetry
delay. The result is good enough for approximate moving visualization. It is not
synchronized RGB-D. Neither inter-frame chassis movement nor independently
moving objects are compensated, and fast movement can visibly misalign colours
or labels. Fix source capture timestamps before relying on this for precision.
The board scale also remains an estimate (see `perception/calibration/`).

## Validation

- Focused projection, pose, API and viewer tests:
  `station/l515/tests/test_articulated_rgbd.py`, `test_rgbd_projection.py`,
  `test_rgbd_acquisition.py`.
- An earlier MuJoCo regression compared computed camera<-depth transforms with
  the actual simulated camera sites after head-only, combined body/head, and
  chassis motion, to a matrix tolerance of 2e-5. That simulation harness isn't
  part of this repo.
- Live: pose bundles carried ~20 ms pose brackets. Articulated projection and
  depth-supported surface boxes were observed. Some pairs fail the Wi-Fi timing
  bounds, and the viewer labels those intervals as frozen.
- Physical head/body motion validation is still pending. Don't claim the
  simulation regression covers it.

An A/B check that disabled the wheels app's Wi-Fi mount-detection scans showed
no meaningful change in accepted pairs (31/46 with scanning, 19/30 without).
Timing jitter is a measured limitation, and scanning alone doesn't explain it.
