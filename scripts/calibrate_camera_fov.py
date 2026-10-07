#!/usr/bin/env python3
"""Measure the head camera's field of view, from the Mac, against the robot.

Why this exists: ``track_hfov_deg``/``track_vfov_deg`` are the only numbers
that turn pixels into angles, so the follow loop's turn magnitude is only as
good as they are. Guessing 70/55 makes the robot consistently over- or
under-turn. This measures them.

## Why it sweeps twice

A target that drifts steadily — a person shifting their weight over 30 s —
adds a bias to the slope that is itself linear, so it produces a *perfect*
straight-line fit at the wrong angle. R² cannot see it; measuring three
times and getting 74°, 90° and 110° is what it looks like from outside.

So each angle is visited twice, once ascending and once descending. A
linear drift enters the two passes with opposite sign and cancels in the
combined fit, and comparing the two passes gives an honest read on how
much the target actually moved. If they disagree, the tool says so instead
of printing a confident wrong number.

## The trick

You never need to know how far away the target is, or what angle it is at.
Hold a target still, sweep the HEAD through known angles, and watch where
the target lands in the image:

    cx = W/2 + f · tan(head_yaw − θ)           θ = the target's true bearing

θ is unknown but constant, and near the image centre tan is its argument,
so the slope hands you the focal length directly:

    d(cx)/d(head_yaw) = f  (per radian)   ⟹   HFOV = 2·atan(W / 2f)

Note this is NOT ``W / slope``. That linear shortcut is only valid for a
narrow lens; on a ~90° camera it reports ~120° instead.

Same for pitch and height (with a sign flip, since image y grows downward).
A straight-line fit over several head angles also tells you whether the
model is even valid: a low R² means something moved, or the lens is far
enough from a pinhole that one number won't describe it.

## Two methods

``--method flow`` (default) needs nothing from you. It turns the head a few
degrees and matches ORB features of the *static scene* between the two
frames: hundreds of correspondences, and the median shift is immune to a
person wandering through (they are a minority of features). It reads the
head's ACTUAL angle back from the daemon, so a neck that under-travels
cannot bias it either.

``--method detect`` tracks one detected object instead. It needs a target
that holds still — a person's bounding box wanders with their posture, and
a slow drift produces a perfect straight-line fit at the wrong angle — so
prefer ``flow`` unless the scene is featureless.

## Use

Put something the detector knows in front of the robot — you count ("person"),
so does a chair, a bottle, a potted plant, a TV. Stand roughly centred, a
couple of metres back, and KEEP STILL for the sweep (~30 s).

    python scripts/calibrate_camera_fov.py --host <robot-ip>            # flow (default)
    python scripts/calibrate_camera_fov.py --host <robot-ip> --method detect \
        --model ~/.config/reachy_wheels_app/models/yolo11n.onnx --target person

``--host`` falls back to ``$REACHY_HOST``. The robot must be running
``reachy_wheels_app`` (``robot/wheels_app``); ``--method detect`` also needs
that app's package importable on the station (this script adds
``robot/wheels_app`` to ``sys.path``) plus ``onnxruntime`` and the same ONNX
model the robot uses (default: the wheels app's model dir under ~/.config).

Add ``--apply`` to write the result straight into the robot's app settings.
Only the head moves — the wheels are never commanded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "robot" / "wheels_app"))


def http_json(url: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def http_bytes(url: str, timeout: float = 15.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def robust_fit(xs: list[float], ys: list[float], sigma: float = 2.0,
               min_keep: int = 4) -> tuple[float, float, float, list[int]]:
    """Least squares with one pass of outlier rejection.

    A detector run over a sweep does not always return the same object: a
    second person, a reflection, or a mirror-image false positive lands a
    point far off the line and quietly drags the slope — and the slope is
    the entire measurement. Fit, drop anything more than ``sigma`` standard
    deviations off, refit. Returns ``(a, b, r2, kept_indices)``.
    """
    a, b, _ = linear_fit(xs, ys)
    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    n = len(residuals)
    mean = sum(residuals) / n
    sd = (sum((r - mean) ** 2 for r in residuals) / n) ** 0.5
    if sd <= 1e-9:
        a, b, r2 = linear_fit(xs, ys)
        return a, b, r2, list(range(n))

    kept = [i for i, r in enumerate(residuals) if abs(r - mean) <= sigma * sd]
    if len(kept) < max(min_keep, 3) or len(kept) == n:
        a, b, r2 = linear_fit(xs, ys)
        return a, b, r2, list(range(n))

    a, b, r2 = linear_fit([xs[i] for i in kept], [ys[i] for i in kept])
    return a, b, r2, kept


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares ``y = a·x + b``; returns ``(a, b, r_squared)``."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, my, 0.0
    a = sxy / sxx
    b = my - a * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def calibrate_by_flow(args, base, head_state, look, fetch) -> int:
    """Measure focal length from how far the static scene slides per degree.

    A camera rotating by dθ about its vertical axis slides everything in
    view sideways by f·dθ pixels (near the centre, where tan is linear).
    Match ORB features of the room between two head angles, take the median
    horizontal shift, and f falls out — with no dependence on what is in the
    scene, whether anything in it moved, or whether the neck reached the
    angle it was told (the actual angle is read back from the daemon).
    """
    import cv2
    import numpy as np

    def grab():
        buf = fetch(f"{base}/api/camera")
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

    def true_yaw() -> float | None:
        st = head_state()
        return None if st is None else st[0] - st[1]

    orb = cv2.ORB_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    angles = [float(v) for v in args.yaws.split(",")]
    print("matching static scene features between head angles")
    print("(nothing is required of you — people in view are outliers "
          "the median ignores)\n")

    samples: list[tuple[float, float, int]] = []
    width = height = 0
    previous = None
    for cmd in angles:
        look(cmd, 0.0)
        time.sleep(args.settle)
        frame = grab()
        actual = true_yaw()
        if frame is None or actual is None:
            print(f"  yaw={cmd:+6.1f}  no frame / no pose readback")
            previous = None
            continue
        height, width = frame.shape[:2]
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(grey, None)
        if des is None or len(kp) < 40:
            print(f"  yaw={cmd:+6.1f}  too few features ({0 if des is None else len(kp)})")
            previous = None
            continue

        if previous is not None:
            p_kp, p_des, p_actual = previous
            d_yaw = actual - p_actual
            if abs(d_yaw) < 0.75:
                print(f"  yaw={cmd:+6.1f}  head barely moved ({d_yaw:+.2f}°) — skipped")
            else:
                matches = matcher.match(p_des, des)
                band = (0.2 * width, 0.8 * width)   # central, where tan≈linear
                shifts = [kp[m.trainIdx].pt[0] - p_kp[m.queryIdx].pt[0]
                          for m in matches
                          if band[0] <= p_kp[m.queryIdx].pt[0] <= band[1]
                          and abs(kp[m.trainIdx].pt[1] - p_kp[m.queryIdx].pt[1]) < 40]
                if len(shifts) < 25:
                    print(f"  yaw={cmd:+6.1f}  only {len(shifts)} usable matches")
                else:
                    dx = float(np.median(shifts))
                    px_per_deg = dx / d_yaw
                    samples.append((d_yaw, px_per_deg, len(shifts)))
                    print(f"  yaw={cmd:+6.1f}  Δhead={d_yaw:+6.2f}°  "
                          f"median shift {dx:+7.1f}px  "
                          f"→ {px_per_deg:6.2f} px/deg   ({len(shifts)} matches)")
        previous = (kp, des, actual)

    look(0.0, 0.0)

    if len(samples) < 3:
        print("\nnot enough usable pairs — is the scene textured enough, and "
              "is the daemon returning head poses?")
        return 1

    rates = sorted(s[1] for s in samples)
    px_per_deg = rates[len(rates) // 2]          # median across pairs
    spread = (rates[-1] - rates[0]) / max(1e-6, px_per_deg)
    focal_px = abs(px_per_deg) * 180.0 / math.pi
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * focal_px)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * focal_px)))

    print("\n" + "=" * 58)
    print(f"frame            {width} x {height}")
    print(f"px per real deg  {px_per_deg:.2f}   (median of {len(samples)} pairs, "
          f"spread {spread:.0%})")
    print(f"focal length     {focal_px:.0f} px")
    print(f"  →  track_hfov_deg = {hfov:.1f}")
    print(f"  →  track_vfov_deg = {vfov:.1f}")
    print("=" * 58)

    if spread > 0.25:
        print("\nWARNING: the per-pair estimates spread more than 25%. "
              "Something in the scene is moving a lot, or the head is not "
              "settling before each grab (try a longer --settle).")
    if not 20.0 <= hfov <= 160.0:
        print(f"\n{hfov:.0f}° is not a real lens — nothing applied.")
        return 1

    if args.apply:
        print(f"\napplied to the robot: " + str(http_json(
            f"{base}/api/track/config",
            {"track_hfov_deg": round(hfov, 1), "track_vfov_deg": round(vfov, 1)})))
    else:
        print(f"\nre-run with --apply to write these to the robot")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("REACHY_HOST", ""),
                    help="robot address running reachy_wheels_app (env REACHY_HOST), "
                         "e.g. <robot-ip> or reachy-mini.local")
    ap.add_argument("--port", type=int, default=8042)
    ap.add_argument("--model", default="~/.config/reachy_wheels_app/models/yolo11n.onnx",
                    help="the SAME ONNX model the robot uses")
    ap.add_argument("--target", default="person",
                    help="COCO class to track during the sweep")
    ap.add_argument("--yaws", default="-24,-16,-8,0,8,16,24",
                    help="head yaw angles to sample, degrees (+ = left)")
    ap.add_argument("--pitches", default="-12,-6,0,6,12",
                    help="head pitch angles to sample, degrees (+ = up)")
    ap.add_argument("--settle", type=float, default=1.2,
                    help="seconds to wait after each move before grabbing")
    ap.add_argument("--passes", type=int, default=2, choices=(1, 2),
                    help="2 = sweep up then back down, cancelling any steady "
                         "drift in the target (strongly recommended)")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--edge-margin", type=int, default=10,
                    help="discard boxes touching within this many px of a "
                         "frame edge — a clipped box's centre does not move "
                         "with the object and silently biases the fit")
    ap.add_argument("--daemon-port", type=int, default=8000,
                    help="Reachy daemon, for reading the head's ACTUAL pose")
    ap.add_argument("--assume-head-gain", type=float, default=None,
                    help="skip the head-fidelity measurement and use this")
    ap.add_argument("--method", choices=("flow", "detect"), default="flow",
                    help="flow = match static scene features (no cooperation "
                         "needed); detect = track one detected object")
    ap.add_argument("--apply", action="store_true",
                    help="POST the measured FOV into the robot's settings")
    args = ap.parse_args()
    if not args.host:
        ap.error("--host (or REACHY_HOST) is required")

    import numpy as np

    base = f"http://{args.host}:{args.port}"
    detector = None
    if args.method == "detect":
        from reachy_wheels_app.tracking.detect_onnx import OnnxDetector
        detector = OnnxDetector(str(Path(args.model).expanduser()),
                                imgsz=args.imgsz, min_score=args.min_score)

    def look(yaw: float, pitch: float) -> None:
        http_json(f"{base}/api/motion/look", {"yaw": yaw, "pitch": pitch})

    def sample(yaw: float, pitch: float, axis: str = "x"):
        """Centre of the best target at this head angle, or None.

        ``axis`` says which coordinate this sample is for, and only that
        axis's clipping disqualifies it: a person cut off at the top of the
        frame still has an honest horizontal centre, and throwing those
        samples away is what makes this tool need a bigger room than it
        actually does.
        """
        look(yaw, pitch)
        time.sleep(args.settle)
        try:
            jpeg = http_bytes(f"{base}/api/camera")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"  ! camera read failed: {exc}")
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        found = detector.detect(frame, targets=(args.target,))
        if not found:
            print(f"  yaw={yaw:+6.1f} pitch={pitch:+6.1f}  no {args.target} seen")
            return None
        best = max(found, key=lambda d: d.score)
        h_px, w_px = frame.shape[0], frame.shape[1]
        m = args.edge_margin
        x1, y1, x2, y2 = best.bbox
        clipped = ((x1 <= m or x2 >= w_px - m) if axis == "x"
                   else (y1 <= m or y2 >= h_px - m))
        if clipped:
            # The box runs off the frame, so its centre is the centre of the
            # *visible part*, which moves slower than the object does. Left
            # in, these points flatten the slope and inflate the FOV — the
            # exact way a large nearby object (a couch) ruins this measurement.
            print(f"  yaw={yaw:+6.1f} pitch={pitch:+6.1f}  clipped on {axis} "
                  f"at the frame edge — skipped")
            return None
        cx, cy = best.center
        print(f"  yaw={yaw:+6.1f} pitch={pitch:+6.1f}  "
              f"cx={cx:7.1f} cy={cy:7.1f}  score={best.score:.2f}")
        return cx, cy, frame.shape[1], frame.shape[0]

    import cv2  # noqa: E402 — after numpy, and only needed here

    daemon = f"http://{args.host}:{args.daemon_port}"

    def head_state() -> tuple[float, float] | None:
        """(head yaw, body yaw) in degrees, as the robot actually is."""
        try:
            h = http_json(f"{daemon}/api/state/present_head_pose")
            b = http_json(f"{daemon}/api/state/present_body_yaw")
        except Exception:  # noqa: BLE001
            return None
        try:
            return (math.degrees(float(h["yaw"])), math.degrees(float(b)))
        except (KeyError, TypeError, ValueError):
            return None

    # --- step 0: does the neck reach the angle we ask for? --------------
    # It does not, on this robot — and every degree of shortfall inflates
    # the FOV that the sweep reports, because the sweep assumes the head
    # moved as far as it was told.
    head_gain = args.assume_head_gain
    if head_gain is None:
        print("measuring neck fidelity (commanded vs actual head yaw):")
        pairs = []
        for cmd in (-10.0, -5.0, 0.0, 5.0, 10.0):
            look(cmd, 0.0)
            time.sleep(args.settle)
            st = head_state()
            if st is None:
                print("  (no head-pose readback from the daemon — assuming 1.0)")
                pairs = []
                break
            # present_head_pose is in the world frame; the shell's own yaw
            # has to come out to leave the neck's contribution.
            pairs.append((cmd, st[0] - st[1]))
            print(f"  commanded {cmd:+6.1f}   actual {pairs[-1][1]:+7.2f}")
        if len(pairs) >= 4:
            a, _, r2 = linear_fit([p[0] for p in pairs], [p[1] for p in pairs])
            if 0.4 <= a <= 1.6 and r2 > 0.9:
                head_gain = a
                print(f"  → neck gain {head_gain:.3f} (R²={r2:.4f})")
            else:
                print(f"  → implausible gain {a:.2f} (R²={r2:.3f}); assuming 1.0")
    if not head_gain:
        head_gain = 1.0
    print()

    if args.method == "flow":
        return calibrate_by_flow(args, base, head_state, look, http_bytes)

    print(f"calibrating against {base} using '{args.target}'")
    print("hold the target still…\n")

    # --- horizontal sweep: vary yaw, pitch level ------------------------
    print("horizontal sweep (yaw):")
    yaw_list = [float(v) for v in args.yaws.split(",")]
    width, height = 0, 0
    passes: list[tuple[list[float], list[float]]] = []
    orders = [yaw_list, list(reversed(yaw_list))][:max(1, args.passes)]
    for pass_no, order in enumerate(orders, start=1):
        if len(orders) > 1:
            print(f"  pass {pass_no} ({'ascending' if pass_no == 1 else 'descending'}):")
        p_yaws, p_cxs = [], []
        for yaw in order:
            got = sample(yaw, 0.0, axis="x")
            if got:
                p_cxs.append(got[0]); p_yaws.append(yaw)
                width, height = got[2], got[3]
        passes.append((p_yaws, p_cxs))

    yaws = [y for p in passes for y in p[0]]
    cxs = [c for p in passes for c in p[1]]

    # --- vertical sweep: vary pitch, yaw centred ------------------------
    print("\nvertical sweep (pitch):")
    pitches, cys = [], []
    for pitch in [float(v) for v in args.pitches.split(",")]:
        got = sample(0.0, pitch, axis="y")
        if got:
            cys.append(got[1]); pitches.append(pitch)

    look(0.0, 0.0)

    if len(yaws) < 4:
        print(f"\nonly {len(yaws)} usable horizontal samples — need at least 4 "
              "(three points fit a line no matter what, so the R² would be "
              "meaningless).")
        print("Pick a target that stays fully inside the frame across the "
              "whole sweep: a person standing a couple of metres back is "
              "ideal; a large nearby object is the worst case.")
        return 1

    # Fit each pass separately and average the slopes, rather than pooling
    # the points. If the target sits somewhere slightly different on the two
    # passes, the pooled data is two PARALLEL lines: the slope is still
    # right but R² collapses, and pooling would report a good measurement as
    # a bad one. Per-pass fits keep both numbers meaningful.
    pass_slopes, pass_r2, kept_x = [], [], []
    for p_yaws, p_cxs in passes:
        if len(p_yaws) >= 4:
            a, _, r2, kept = robust_fit(p_yaws, p_cxs)
            pass_slopes.append(a)
            pass_r2.append(r2)
            kept_x.extend(kept)

    if pass_slopes:
        slope_x = sum(pass_slopes) / len(pass_slopes)
        r2x = min(pass_r2)
    else:
        slope_x, _, r2x, kept_x = robust_fit(yaws, cxs)
    dropped = []
    if dropped:
        print("\ndiscarded as outliers (detector almost certainly found a "
              "different object at these angles): "
              + ", ".join(f"{y:+.0f}°" for y in dropped))
    if abs(slope_x) < 1e-6:
        print("\nthe target did not move in frame as the head turned — is the "
              "head actually moving? check /api/motion")
        return 1
    # The sweep measured pixels per COMMANDED degree; convert to pixels per
    # degree the head actually moved before touching the pinhole relation.
    slope_true = abs(slope_x) / head_gain
    focal_px = slope_true * 180.0 / math.pi
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * focal_px)))

    if not 20.0 <= hfov <= 160.0:
        print(f"\nmeasured {hfov:.0f}° horizontal, which is not a real lens.")
        print("Almost always one of:")
        print("  • the target was clipped by a frame edge for part of the "
              "sweep (use a smaller, more central target)")
        print("  • the detector jumped between two different objects")
        print("  • the head did not actually reach the commanded angles")
        print("Nothing was applied.")
        return 1

    print("\n" + "=" * 58)
    print(f"frame            {width} x {height}")
    print(f"horizontal slope {slope_x:+.2f} px per commanded deg   "
          f"worst-pass R²={r2x:.4f}   n={len(kept_x)}/{len(yaws)}")
    print(f"neck gain        {head_gain:.3f}  →  {slope_true:.2f} px per REAL deg")
    print(f"focal length     {focal_px:.0f} px")
    print(f"  →  track_hfov_deg = {hfov:.1f}")

    # One focal length governs both axes, so the vertical FOV follows from
    # the frame height — no separate sweep needed, and no drift to fight.
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * focal_px)))
    print(f"  →  track_vfov_deg = {vfov:.1f}  (same focal length, frame height)")

    if False and len(pitches) >= 3:
        slope_y, _, r2y, kept_y = robust_fit(pitches, cys)
        if abs(slope_y) > 1e-6:
            vfov = abs(height / slope_y)
            print(f"vertical slope   {slope_y:+.2f} px/deg   R²={r2y:.4f}   "
                  f"n={len(kept_y)}/{len(pitches)}")
            print(f"  →  track_vfov_deg = {vfov:.1f}")

    print("=" * 58)

    if len(kept_x) < 8:
        print(f"\nNOTE: only {len(kept_x)} usable points. Usable, but more would be "
              "better — widen --yaws or pick a target with a longer clear run.")
    drifted = False
    if len(pass_slopes) == 2:
        up, down = pass_slopes
        spread = abs(up - down) / max(1e-6, abs((up + down) / 2))
        print(f"\npass slopes: {up:+.2f} and {down:+.2f} px/deg "
              f"({spread:.0%} apart)")
        if spread > 0.15:
            drifted = True
            print("\nWARNING: the two passes disagree by more than 15%, which "
                  "means the target moved during the sweep. The combined fit "
                  "cancels most of a steady drift, so the number above is "
                  "still the best estimate — but re-run with a genuinely "
                  "static target before trusting it to a few degrees.")
        else:
            print("the two passes agree — the target held still, so this "
                  "number is trustworthy")

    if r2x < 0.97 and not drifted:
        print("\nWARNING: even within a single pass the fit is poor "
              f"(R²={r2x:.3f}). The target's detected box is wandering — a "
              "person's bounding box moves with their posture. A small rigid "
              "object (a bottle, a backpack, a potted plant) placed a couple "
              "of metres away gives a much sharper number.")

    if head_gain < 0.97 or head_gain > 1.03:
        print(f"\nNOTE: the neck reaches only {head_gain:.1%} of what it is "
              "asked for. Set MotionLimits.yaw_command_gain to this so the "
              "head lands where the controller intends and the tracked angle "
              "stays true.")

    if args.apply:
        out = http_json(f"{base}/api/track/config",
                        {"track_hfov_deg": round(hfov, 1),
                         "track_vfov_deg": round(vfov, 1)})
        print(f"\napplied to the robot: {out}")
    else:
        print(f"\nto apply:  curl -X POST {base}/api/track/config "
              f"-H 'Content-Type: application/json' -d "
              f"'{{\"track_hfov_deg\": {hfov:.1f}, "
              f"\"track_vfov_deg\": {vfov:.1f}}}'")
        print("or re-run this with --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
