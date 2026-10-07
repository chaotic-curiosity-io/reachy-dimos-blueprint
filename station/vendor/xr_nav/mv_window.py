"""Sliding-window multi-view DA3 backend (Phase C).

The single-frame depth path can't make depth *mutually consistent across
views* — each frame's metric depth wobbles a little, and nearby objects (the
chair) fuzz as the head pans. DA3's multi-view pass fixes exactly that: run a
short overlapping window of keyframes through DA3 together, anchored to their
kinematic camera poses. With ``align_to_input_ext_scale`` DA3 adopts those
poses and rescales its depth to their metric scale (Umeyama), so every window
lands in the same gravity-correct, world-locked frame — no cross-window Sim3
stitcher and no loop closure needed at room scale (the kinematic anchor bounds
global drift; window-to-window seams are handled by the Phase-B ICP).

This module is intentionally decoupled from the pipeline: it owns a background
thread with a bounded queue of keyframes, calls ``estimator.infer_multi`` when
a window fills, unprojects + confidence-filters each view, and hands the fused
world-frame point batch back through a callback. The pipeline stays responsive
(live coarse map at its own rate); the refined windows arrive asynchronously.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger("xr_nav.mv_window")


@dataclass
class MVWindowConfig:
    window: int = 6              # frames per multi-view inference
    overlap: int = 2            # frames shared between consecutive windows
    stride: int = 2             # depth-map pixel stride when unprojecting
    conf_percentile: float = 40.0  # drop points below this per-window conf pct
    max_range_m: float = 6.0    # discard points beyond this (metres)
    min_baseline_m: float = 0.015  # skip a window whose views barely moved
    max_queue: int = 3          # drop-oldest beyond this many pending windows


@dataclass
class Keyframe:
    rgb: np.ndarray             # (H, W, 3)
    c2w: np.ndarray             # 4x4 kinematic camera-to-world (world-locked)
    K: np.ndarray               # 3x3 intrinsics at rgb resolution
    ts: float = 0.0


@dataclass
class WindowResult:
    points: np.ndarray          # (M, 3) world-frame
    colors: np.ndarray | None   # (M, 3) in [0,1] or None
    n_views: int = 0
    seq: int = 0


def _unproject(depth: np.ndarray, K: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Depth (H,W) + intrinsics -> camera-frame points (N,3) and the (v,u) mask
    indices used, so colours/conf can be sampled with the same subsample."""
    h, w = depth.shape
    u, v = np.meshgrid(
        np.arange(0, w, stride, dtype=np.float32),
        np.arange(0, h, stride, dtype=np.float32),
    )
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    K_inv = np.linalg.inv(K).astype(np.float32)
    rays = (K_inv @ pixels.T).T
    d = depth[::stride, ::stride].reshape(-1, 1).astype(np.float32)
    return rays * d, d[:, 0]


def fuse_window(views: list[dict], cfg: MVWindowConfig,
                colors_rgb: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Unproject + conf/range-filter each view of a DA3 multi-view result and
    stack into one world-frame point batch. ``views`` are ``infer_multi`` dicts
    ({depth, conf, K, c2w}); ``colors_rgb`` optionally supplies each view's RGB
    at the depth resolution for per-point colour."""
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    for i, vw in enumerate(views):
        depth, conf, K, c2w = vw["depth"], vw["conf"], vw["K"], vw["c2w"]
        if K is None or depth is None or not np.any(depth > 0):
            continue
        pts_cam, d = _unproject(depth, K, cfg.stride)
        conf_s = conf[::cfg.stride, ::cfg.stride].reshape(-1)
        # Per-window adaptive confidence gate + range gate + valid depth.
        thr = (np.percentile(conf_s[conf_s > 0], cfg.conf_percentile)
               if np.any(conf_s > 0) else 0.0)
        dist = np.linalg.norm(pts_cam, axis=1)
        mask = (d > 0) & (conf_s >= thr) & (dist <= cfg.max_range_m)
        if not np.any(mask):
            continue
        pts_cam = pts_cam[mask]
        world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        all_pts.append(world.astype(np.float32))
        if colors_rgb is not None and i < len(colors_rgb) and colors_rgb[i] is not None:
            # Subsample colours by the same stride so they align with the mask.
            col = colors_rgb[i][::cfg.stride, ::cfg.stride].reshape(-1, 3)[mask]
            all_cols.append(col.astype(np.float32))
    if not all_pts:
        return np.zeros((0, 3), np.float32), None
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0) if all_cols and len(all_cols) == len(all_pts) else None
    return pts, cols


def _window_baseline(frames: list[Keyframe]) -> float:
    """Max pairwise camera-center distance in a window (m) — a proxy for how
    well-conditioned the Umeyama scale fit will be."""
    centers = np.stack([f.c2w[:3, 3] for f in frames])
    if len(centers) < 2:
        return 0.0
    return float(np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1).max())


class MVWindowWorker:
    """Background sliding-window multi-view DA3 worker.

    Push keyframes with :meth:`submit`; when ``window`` have accumulated the
    worker runs multi-view inference on a copy, slides forward by
    ``window - overlap``, and delivers a :class:`WindowResult` via ``on_result``.
    Falls behind gracefully: the pending-window queue is bounded and drops the
    oldest, so a slow model never blocks the live pipeline.
    """

    def __init__(self, estimator, cfg: MVWindowConfig,
                 on_result: Callable[[WindowResult], None]):
        self._est = estimator
        self._cfg = cfg
        self._on_result = on_result
        self._pending: deque[list[Keyframe]] = deque(maxlen=max(1, cfg.max_queue))
        self._buf: list[Keyframe] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._seq = 0
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mv-window-worker")

    def start(self) -> None:
        self._thread.start()

    def submit(self, kf: Keyframe) -> None:
        """Add a keyframe; enqueue a window every (window - overlap) frames."""
        cfg = self._cfg
        with self._cond:
            self._buf.append(kf)
            if len(self._buf) >= cfg.window:
                window = self._buf[-cfg.window:]
                # Slide forward, retaining `overlap` frames for the next window.
                keep = cfg.overlap
                self._buf = self._buf[-keep:] if keep > 0 else []
                if len(self._pending) == self._pending.maxlen:
                    self._dropped += 1
                    logger.warning("mv-window queue full — dropping oldest "
                                   "(%d dropped total)", self._dropped)
                self._pending.append(window)
                self._cond.notify()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while not self._pending and not self._stop.is_set():
                    self._cond.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                window = self._pending.popleft()
            try:
                self._process(window)
            except Exception as e:  # noqa: BLE001 — one bad window must not kill the worker
                logger.warning("mv-window inference failed: %s", e)

    def _process(self, frames: list[Keyframe]) -> None:
        cfg = self._cfg
        baseline = _window_baseline(frames)
        if baseline < cfg.min_baseline_m:
            # Rotation-dominant / near-static window: the Umeyama translation
            # scale is ill-conditioned. Skip — the live path already covered it.
            logger.info("mv-window skipped (baseline %.1f mm < %.1f mm)",
                        baseline * 1000, cfg.min_baseline_m * 1000)
            return
        import cv2
        images = [f.rgb for f in frames]
        c2ws = [f.c2w for f in frames]
        Ks = [f.K for f in frames]
        views = self._est.infer_multi(images, c2w_list=c2ws, K_list=Ks)
        # Resize each source RGB to its view's depth resolution for colour.
        colors = []
        for f, vw in zip(frames, views):
            dh, dw = vw["depth"].shape
            img = f.rgb
            if img.shape[:2] != (dh, dw):
                img = cv2.resize(img, (dw, dh))
            colors.append((img.astype(np.float32) / 255.0)[..., :3])
        pts, cols = fuse_window(views, cfg, colors_rgb=colors)
        self._seq += 1
        if len(pts) == 0:
            return
        self._on_result(WindowResult(points=pts, colors=cols,
                                     n_views=len(frames), seq=self._seq))

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def dropped(self) -> int:
        return self._dropped
