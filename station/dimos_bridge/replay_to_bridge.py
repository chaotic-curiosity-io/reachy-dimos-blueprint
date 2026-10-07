#!/usr/bin/env python3
"""Fake robot: stream a saved recording to the station bridge over WebSocket.

A zero-robot offline demo. It exercises the *real* live path — bridge ingest,
capture-time pose pairing, the head->camera lever arm, the whole dimOS
pipeline — deterministically, with no robot. The station server
(``python -m station.dimos_bridge.server``) can't tell this from a live
``dimos_scanner`` app, so it is also the A/B rig for pose/drift changes: run
the same recording through the server before/after a change and compare the
saved maps.

It streams each frame and pose with its ORIGINAL recorded timestamp
(monotonic-shifted to "now" so staleness gates pass), so the server's
``pose_at(frame_ts)`` interpolation sees exactly the timing the robot produced —
including the fast pans that alias when a frame is paired with the latest pose.

Expected recording layout (one directory per recording)::

    <recording>/
      camera.mp4               head-camera video, one decoded frame per sample
      camera_timestamps.jsonl  one line per video frame:  {"ts": <unix s>, "value": <anything>}
      head_pose.jsonl          one line per FK sample:    {"ts": <unix s>, "value": [16 floats]}
                               (row-major 4x4 body->head pose, the same matrix
                               the robot streams as binary 0x03 pose messages)
      metadata.json            optional, ignored here

``camera_timestamps.jsonl`` may be missing (frames are then spaced at 5 Hz);
without ``head_pose.jsonl`` the server falls back to visual odometry.

Usage::

  # Terminal 1: the station server (waits for a "robot")
  python -m station.dimos_bridge.server --dimos-dir "$DIMOS_DIR" \
      --pose external --depth depthpro --extra --viz rerun
  # Terminal 2: this harness
  python -m station.dimos_bridge.replay_to_bridge /path/to/recording --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from station.dimos_bridge.protocol import (  # noqa: E402
    DEFAULT_BRIDGE_PORT,
    encode_frame,
    encode_pose,
    hello,
)


def _read_jsonl(path: Path) -> list[tuple[float, object]]:
    out: list[tuple[float, object]] = []
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out.append((float(rec["ts"]), rec["value"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return out


def _load_frames(rec: Path) -> tuple[list[np.ndarray], list[float]]:
    cap = cv2.VideoCapture(str(rec / "camera.mp4"))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    ts = [t for t, _ in _read_jsonl(rec / "camera_timestamps.jsonl")]
    if ts and len(ts) != len(frames):
        # Pair by index; warn on mismatch (mirror the replay script's behaviour).
        print(f"[replay] WARNING: {len(ts)} camera ts != {len(frames)} frames; "
              f"pairing by index up to min()")
    n = min(len(frames), len(ts)) if ts else len(frames)
    return frames[:n], (ts[:n] if ts else [i / 5.0 for i in range(n)])


async def _stream(rec: Path, host: str, port: int, camera_k: list[float] | None,
                  depth_model: str, speed: float) -> None:
    frames, frame_ts = _load_frames(rec)
    poses = _read_jsonl(rec / "head_pose.jsonl")
    if not frames:
        sys.exit(f"[replay] no frames decoded from {rec/'camera.mp4'}")
    if not poses:
        print(f"[replay] WARNING: no head_pose.jsonl in {rec} — server will fall "
              f"back to VO. Pick a recording with head poses to test --pose external.")

    # Merge frames + poses into one time-ordered event stream, then shift all
    # timestamps so the earliest lands at "now" (staleness gates + the server's
    # first-pose anchor want wall-clock-ish values).
    t0 = min([frame_ts[0]] + ([poses[0][0]] if poses else []))
    now = time.time()

    def shift(t: float) -> float:
        return now + (t - t0) / max(speed, 1e-6)

    events: list[tuple[float, str, object]] = []
    for i, ft in enumerate(frame_ts):
        events.append((ft, "frame", i))
    for pt, pv in poses:
        events.append((pt, "pose", pv))
    events.sort(key=lambda e: e[0])

    hello_cfg: dict = {"depth_model": depth_model, "pose_stream": bool(poses)}
    if camera_k is not None:
        hello_cfg["camera_K"] = camera_k
        h, w = frames[0].shape[:2]
        hello_cfg["camera_wh"] = [int(w), int(h)]

    url = f"ws://{host}:{port}"
    print(f"[replay] connecting to {url} — {len(frames)} frames, {len(poses)} poses, "
          f"speed x{speed}")
    async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
        await ws.send(hello("robot", name="replay", config=hello_cfg))
        n_sent_f = n_sent_p = 0
        wall_start = time.time()
        for src_ts, kind, val in events:
            target = shift(src_ts)
            dt = target - time.time()
            if dt > 0:
                await asyncio.sleep(dt)
            ts_ns = int(target * 1e9)
            if kind == "frame":
                bgr = frames[val]
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                h, w = bgr.shape[:2]
                await ws.send(encode_frame(buf.tobytes(), width=w, height=h, ts_ns=ts_ns))
                n_sent_f += 1
            else:
                m = np.asarray(val, dtype=float).reshape(4, 4)
                await ws.send(encode_pose(m.flatten(), ts_ns=ts_ns))
                n_sent_p += 1
        print(f"[replay] done — sent {n_sent_f} frames + {n_sent_p} poses in "
              f"{time.time() - wall_start:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path, help="recording dir (camera.mp4 + *.jsonl)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_BRIDGE_PORT)
    ap.add_argument("--camera-k", type=str, default=None,
                    help="fx,fy,cx,cy advertised in the hello (native resolution). "
                         "Omit to let the server use its calibration file.")
    ap.add_argument("--depth-model", default="depthpro",
                    help="depth_model advertised in the hello")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (2.0 = twice as fast)")
    args = ap.parse_args()

    rec = args.recording
    if not (rec / "camera.mp4").exists():
        sys.exit(f"no camera.mp4 in {rec}")

    camera_k = None
    if args.camera_k:
        vals = [float(x) for x in args.camera_k.split(",")]
        if len(vals) == 4:  # fx,fy,cx,cy -> full 9-vector row-major
            fx, fy, cx, cy = vals
            camera_k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        elif len(vals) == 9:
            camera_k = vals
        else:
            sys.exit("--camera-k needs fx,fy,cx,cy or a full 9-value K")

    try:
        asyncio.run(_stream(rec, args.host, args.port, camera_k,
                            args.depth_model, args.speed))
    except KeyboardInterrupt:
        print("[replay] interrupted")


if __name__ == "__main__":
    main()
