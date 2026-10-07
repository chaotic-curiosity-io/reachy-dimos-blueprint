# Reachy/L515 calibration — 2026-09-08

Results from our build, kept as a reference for what a good fit looks like.
The procedure is in [README.md](README.md).

RGB intrinsics: 13 views at 1280×720, fixed-k3 Brown-Conrady model.
Fit RMS 0.120 px; mean leave-one-view-out RMS 0.126 px.

Fixed-head stereo: 6 stationary paired views, intrinsics held fixed.
Joint RMS 0.196 px. Leave-one-pair-out cross-camera prediction RMS:
Reachy mean 0.363 px, maximum 0.434 px; L515 mean 0.401 px, maximum 0.531 px.
Removing one pair changes translation by at most 0.335 mm and rotation by at
most 0.068 degrees. These are internal consistency metrics, not absolute accuracy.
Independent per-pair PnP transforms differ by up to 2.53 mm / 0.43 degrees.

The fitted color-camera separation is approximately 99.7 mm, conditional on
estimated square size 23.5 mm. Scale is not physically measured. The depth-to-
Reachy transform composes the fitted color transform with L515 factory extrinsics.
The first pair has no recorded head pose; subsequent pairs contain actual
before/after poses. Captures were stationary and bracketed, not hardware synced.
Reachy camera capture freshness is unavailable from its SDK endpoint.

Files: `reachy-rgb-intrinsics-candidate.json`, `paired-extrinsics-report.json`,
`paired-alignment-overlay.jpg` (generated locally, not committed: they are
specific to one camera pair and mount; `paired-extrinsics-report.example.json`
shows the report schema); reproducible scripts `fit_rgb.py`, `fit_pair.py`.
All transforms map source optical coordinates into Reachy optical coordinates:
`p_reachy = R @ p_source + t`, x right / y down / z forward, translation metres.

Status: applied to the experimental fixed-head RGB-D perception preview.
Live recognition produced a refrigerator label with depth-supported visible
surface bounds. Runtime rejects pose, timing and scene-stability failures;
network/CPU latency can intermittently suppress 3D while 2D recognition continues.
Valid only for this camera mode and fixed head/mount configuration. This does
not supply base_link pose, moving-head kinematics, or validated motion control.
The JSON remains labeled candidate to preserve the distinction from measured,
production calibration. No additional board captures are currently required.
