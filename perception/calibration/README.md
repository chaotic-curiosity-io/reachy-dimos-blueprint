# Camera <-> depth calibration (Reachy head RGB <-> RealSense L515)

Classical OpenCV calibration that maps L515 depth into the Reachy head camera's
optical frame, so measured depth can be projected onto the head-camera image
before it enters dimOS. Two stages:

1. **Intrinsics** of the Reachy head camera (1280x720, Brown-Conrady).
2. **Extrinsics** L515 -> Reachy optical, from stationary paired views of one
   ChArUco board, fitted with the intrinsics held fixed.

Our results are in [CALIBRATION_RESULTS.md](CALIBRATION_RESULTS.md). Captured
images, ZIPs and generated reports are git-ignored here (see `.gitignore`).
[`paired-extrinsics-report.example.json`](paired-extrinsics-report.example.json)
shows the output schema with invented values.

## What you need

- The Reachy running the wheels app (`robot/wheels_app`). It serves the head
  camera at `/api/camera`, and the stock daemon reports head/body pose at
  `:8000`.
- The Pi depth streamer (`perception/depth_server`) running with `--color`,
  which enables `/api/calibration-frame` (L515 colour + depth + factory
  intrinsics/extrinsics from one frameset).
- A Python env with `opencv-contrib-python` (for `cv2.aruco`, OpenCV >= 4.7
  API), `numpy`, and `scipy` (only used by the downstream consumers).

```sh
export REACHY_URL=http://reachy-mini.local:8042
export REACHY_DAEMON_URL=http://reachy-mini.local:8000
export DEPTH_SERVER_URL=http://<pi-ip>:8765
# optional: keep data outside the repo
export CALIBRATION_DIR=$HOME/.config/dimos-blueprint/calibration
```

All scripts read and write `$CALIBRATION_DIR` if set, else this folder.

## The board

```python
cv2.aruco.CharucoBoard((5, 7), 0.0235, 0.01175,
                       cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
```

5x7 squares in OpenCV's native orientation, 23.5 mm squares, 11.75 mm markers,
`DICT_5X5_100`. We displayed it full-screen on an iPad, turned sideways. Any
flat, rigid display or print works, as long as you know the square size.

**How the square size was measured: it wasn't, precisely.** 23.5 mm was
*estimated* from a photo of the board on the iPad screen, not measured with
calipers. Intrinsics don't depend on board scale. The extrinsic **translation**
does, linearly: a 2% error in square size gives a 2% error in the ~100 mm
camera separation. Every report carries `scale_basis` to say so. If you print
the board, measure a run of 5 squares with calipers, divide by 5, and update
the constant in all four capture/fit scripts (`capture_rgb_view.py`,
`capture_calibration_pair.py`, `fit_rgb.py`, `fit_pair.py`) and
`square_length_m` in the reports.

## Step 1: intrinsics views (head camera only)

Hold the board still in front of the head camera and capture one view at a
time. Between views, move the board around the whole image (centre, the four
corners, the edges), change distance and tilt it. We used **13 views**.

```sh
python perception/calibration/capture_rgb_view.py --label centre
python perception/calibration/capture_rgb_view.py --label top_left
# ... ~13 views
```

Each view is saved as `rgb-ipad/<stamp>.jpg` plus `.json` (detected corner IDs
and pixel positions), and a view needs at least 12 corners. The tool only reads
the camera and never moves the robot.

## Step 2: fit intrinsics

```sh
python perception/calibration/fit_rgb.py
python perception/calibration/make_intrinsics_candidate.py
```

`fit_rgb.py` fits two Brown-Conrady variants with
`cv2.calibrateCameraExtended`: a full 5-coefficient model (`brown_5`) and one
with `k3` fixed at 0 (`brown_4_fixed_k3`). For each, it reports **leave-one-view-out**
validation: it refits without each view, solves that view's board pose with
PnP, and measures the held-out reprojection RMS. It writes `rgb-fit-report.json`,
`corner-coverage.jpg` (check that corners cover the image) and
`rgb-correction-comparison.jpg`. `make_intrinsics_candidate.py` keeps the model
with the lowest mean held-out error and writes
`reachy-rgb-intrinsics-candidate.json`, the file step 4 reads. Our fit came out
at 0.120 px RMS and 0.126 px mean held-out, with `k3` fixed.

Held-out PnP isn't independent 3D ground truth. It catches overfitting and bad
views, but it doesn't measure absolute accuracy.

## Step 3: paired captures (both cameras, stationary)

**Don't move either sensor mount or the Reachy head during this session.** The
result is valid only for this exact head pose and mount.

Put the board where both the head camera and the L515 colour camera see it,
then run:

```sh
python perception/calibration/capture_calibration_pair.py \
  --directory "${CALIBRATION_DIR:-perception/calibration}/paired"
```

Each run reads the head/body pose from the daemon, then grabs **three**
bracketed samples, each pairing a head-camera JPEG with an L515 calibration
bundle. It accepts the middle sample only if:

- both cameras detect **at least 12 ChArUco corners in common**;
- no corner moves **more than 1.5 px** across the three samples, in either
  camera (the board and cameras were stationary);
- the Pi didn't restart and returned three distinct colour frames;
- the **head pose didn't change** during the capture (<2 mm position,
  <0.02 rad angles, <0.02 rad body yaw between the before/after daemon reads).

This is stationary bracketing, not hardware sync. The Reachy SDK doesn't expose
capture timestamps, so the scene has to hold still. Move the board between runs
to cover different positions and angles. We used **6 pairs**, and one pair is
never enough. Output per pair: `paired/<stamp>/reachy.jpg`, `l515.zip`,
`observation.json`.

## Step 4: fit extrinsics

```sh
python perception/calibration/fit_pair.py
```

`fit_pair.py` runs `cv2.stereoCalibrate` with `CALIB_FIX_INTRINSIC`. The L515
colour intrinsics come from its factory calibration (inside each bundle), and
the Reachy intrinsics from step 2. It reports:

- the joint stereo RMS;
- **leave-one-pair-out** validation: refit without each pair, fit the held-out
  board pose in one camera, project it into the other, and report both
  directions' RMS plus how much the transform moved (mm / degrees);
- independent per-pair PnP consistency.

It writes `paired-extrinsics-report.json`. Convention:
`p_reachy_optical = R @ p_l515_optical + t`, in metres, with x right, y down and
z forward. `l515_depth_to_reachy_optical` composes the fitted colour transform
with the L515's factory depth->colour extrinsics. The report also records the
L515 serial (consumers refuse a different unit), both intrinsics, and the head
poses of every pair.

Our numbers: 0.196 px joint RMS. Held-out prediction averaged 0.36 px into the
Reachy image (worst 0.43 px) and 0.40 px into the L515 (worst 0.53 px).
Dropping any one pair moved translation by at most 0.34 mm and rotation by at
most 0.07 degrees. These measure internal consistency, not absolute accuracy.

## Where the report is used

Both consumers look for the report in this order: `--calibration`,
`$CALIBRATION_REPORT`, `<run directory>/calibration/paired-extrinsics-report.json`,
then this folder.

- `station/l515/continuous_mapping.py`: checks the L515 serial before it
  restores a saved map, and colours points with the factory colour stream.
- `station/l515/rgbd_projection.py` and `station/l515/reachy_rgbd.py`: project
  depth into the head image with `l515_depth_to_reachy_optical` and
  `reachy_intrinsics` (K, D). `articulated_rgbd.py` transfers that reference to
  the current head/body pose, using the first pair with a recorded head pose
  as the reference.

## What does not exist (yet)

- **No wheel-base calibration.** No `base_link`, and no transform from the L515
  mount to the chassis, has been measured. The extrinsics here relate two
  *cameras* at one fixed head pose, nothing more.
- Every path that would command the wheels from mapped geometry is gated on
  `extrinsics_calibrated` in the mapper status
  (`station/l515/reachy_navigation.py`). With no base calibration that flag is
  false, so navigation stays a dry run.
- No hand-eye calibration of the moving head. The head->camera lever arm comes
  from the official MJCF model (see `station/l515/ARTICULATED_RGBD.md`).
- Board scale is estimated, not measured (see above).
