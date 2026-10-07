# Continuous accumulated mapping

`continuous_mapping.py` builds the active map, independent of Reachy
head-camera timing and YOLO inference. It acquires the exact L515 depth/RGB
pair and colours points with the L515's factory depth-to-colour intrinsics and
extrinsics. It scan-matches every fresh depth acquisition. The points, voxels,
persistent coloured objects and the PLY download all come from this same
accumulated map. The independent head camera adds semantic observations when
the articulated timing/projection gates pass (see
[ARTICULATED_RGBD.md](ARTICULATED_RGBD.md)). A head-RGB warning never stops
depth mapping.

Run it with the depth streamer address and (for the semantic hand-off) the
paired calibration report:

```sh
DEPTH_SERVER_URL=http://<pi-ip>:8765 \
python -m station.l515.continuous_mapping --directory ./l515-output
# --calibration path/to/paired-extrinsics-report.json  (default lookup:
#   $CALIBRATION_REPORT, ./l515-output/calibration/, perception/calibration/)
```

## Acquisition and persistence

The Pi's `/api/mapping-frame` endpoint (`perception/depth_server`) returns full
320x240 depth and a 160x120 RGB sample in one compressed bundle.
`color_pixel_stride=4` keeps correspondence with the original 640x480 colour
intrinsics. This cuts transfer size from about 1 MB to about 152 KB per frame.
Live mapping has run at around 2.5 Hz over home Wi-Fi. That rate is not
guaranteed.

Every two seconds, and on graceful exit, the mapper saves a 4 cm coloured voxel
map, provisional object records and a tracking keyframe. On restart it loads the
retained coordinate frame and requires a successful scan match before adding
anything. Losing tracking freezes integration but keeps stored geometry. A
bounded +/-60 degree optical-yaw hypothesis search helps recover from turns
outside the local ICP basin, under the same overlap, residual, observability and
motion limits. This is local scan matching. It is neither global loop closure
nor robust recovery from arbitrary relocation.

Output layout under `--directory`:

| Path | Contents |
| --- | --- |
| `colored-map-segments/*.npz` | atomic voxel maps per segment |
| `colored-map-segments/*.tracking` | resumable tracking checkpoints |
| `continuous-active.json` | which segment resumes on restart |
| `continuous-mapping.json` | live status (state, accepted/rejected, clearance, pose) |
| `live-map.ply` | current map as a coloured point cloud |
| `mapping-frame.npz` | depth frame + its exact accepted pose, handed to perception |

Older experimental segments stay archived and are listed by
`GET /api/map-segments`. The main Rerun view shows only the active map.
Recognized surfaces enter memory only when the pose, the head-camera transform
and the active segment all agree.

Object identities are provisional: same class within 30 cm, one association per
observation. Records stay when their object leaves the view. Nearby same-class
objects and moving objects can cause errors. There's no appearance-based
re-identification and no cleanup of dynamic scenes. Geometry is capped at
200,000 voxels without evicting history, and objects at 2,000 records. The
estimated calibration scale and approximate head-camera timing still limit
semantic positioning.

## Wheel polarity and the scan sweep

Live tests found a wrong motor-inversion configuration: with every wheel
inverted, `rotate_cw` translated backward. Inverting only the two right wheels
(front-left and rear-left `false`, front-right and rear-right `true`) fixed it.
The navigation and trial scripts check this exact polarity before any motion.
After the fix, a 60 ms clockwise pulse produced about 3.9 degrees of rotation
and 4 mm of translation.

`scan_sweep.py` won't move without `--execute`. Its default pulses are 60 ms at
0.15 speed (also tested at 120 ms / 0.20). Each pulse needs fresh tracking, a
clearance check, confirmation that the wheels stopped, and a 12 cm translation
envelope, with a capture dwell between pulses. Motion commands are never
retried after an uncertain response. State queries can retry. Stop commands and
stopped state are checked, and the board's own deadline also expires each pulse.
This is a supervised clear-floor test routine, not autonomous navigation: the
forward sensor can't establish full swept-volume clearance.

```sh
WHEELS_HOST=<wheels-ip> REACHY_URL=http://reachy-mini.local:8042 \
python -m station.l515.scan_sweep --directory ./l515-output --execute --steps 5
```

## Verified behaviour

Tests (`station/l515/tests/test_persistent_rgbd.py`,
`test_continuous_mapping.py`) cover factory colour projection including
downsampling, accumulation and retention, saved-map restoration, object
association, loss without an origin reset, and recovery from a known 25 degree
turn.

On hardware, after the polarity fix: about 101 degrees of accumulated turn over
two 20-step rotation batches and a 5-step forward batch (8.3 cm measured
translation). The map grew to ~33,800 voxels and four provisional object
records. All baseline points stayed within 4 cm of the expanded map's geometry.
A mapper restart kept the same map ID, recovered tracking, and retained all
pre-restart points within 4 cm. This checks persistence, not absolute accuracy
against a surveyed reference. dimOS LCM depth, sensor odometry and coloured
global-map publications were received after the restart.
