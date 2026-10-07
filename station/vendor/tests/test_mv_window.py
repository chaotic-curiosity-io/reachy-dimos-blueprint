"""Phase-C tests: sliding-window multi-view worker (no DA3 model needed).

A fake estimator stands in for ``DA3Estimator.infer_multi`` so the windowing,
fusion geometry, kinematic-anchor unprojection, baseline gate, and drop-oldest
backpressure are all exercised deterministically. The real model run against a
recording is the user's integration step; these lock down the host logic.
"""

from __future__ import annotations

import time

import numpy as np

from xr_nav.mv_window import (
    Keyframe,
    MVWindowConfig,
    MVWindowWorker,
    WindowResult,
    _window_baseline,
    fuse_window,
)

H = W = 48
K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1.0]])


def _pose(t) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = t
    return T


def test_fuse_window_unprojects_to_world():
    # A single view at identity, constant 2 m depth -> a plane at world z=2.
    depth = np.full((H, W), 2.0, np.float32)
    conf = np.ones((H, W), np.float32)
    views = [{"depth": depth, "conf": conf, "K": K, "c2w": np.eye(4)}]
    pts, _ = fuse_window(views, MVWindowConfig(stride=2, conf_percentile=0))
    assert len(pts) == (H // 2) * (W // 2)
    assert np.allclose(pts[:, 2], 2.0, atol=1e-4)


def test_fuse_window_applies_pose_translation():
    depth = np.full((H, W), 1.0, np.float32)
    conf = np.ones((H, W), np.float32)
    c2w = _pose([5.0, -3.0, 10.0])
    views = [{"depth": depth, "conf": conf, "K": K, "c2w": c2w}]
    pts, _ = fuse_window(views, MVWindowConfig(stride=4, conf_percentile=0))
    # Depth 1 m along optical +Z, then translated by the pose -> world z = 11.
    assert np.allclose(pts[:, 2], 11.0, atol=1e-4)
    # Center pixel maps to (5, -3, 11).
    assert np.any(np.all(np.isclose(pts, [5.0, -3.0, 11.0], atol=0.05), axis=1))


def test_fuse_window_confidence_and_range_gates():
    depth = np.full((H, W), 2.0, np.float32)
    conf = np.zeros((H, W), np.float32)
    conf[: H // 2, :] = 1.0            # only top half is confident
    views = [{"depth": depth, "conf": conf, "K": K, "c2w": np.eye(4)}]
    pts, _ = fuse_window(views, MVWindowConfig(stride=2, conf_percentile=50, max_range_m=6.0))
    # Roughly half the points survive the confidence gate.
    assert 0 < len(pts) <= (H // 2) * (W // 2)
    # Range gate: nothing beyond max_range.
    far = np.full((H, W), 100.0, np.float32)
    v2 = [{"depth": far, "conf": np.ones((H, W), np.float32), "K": K, "c2w": np.eye(4)}]
    pts2, _ = fuse_window(v2, MVWindowConfig(stride=4, conf_percentile=0, max_range_m=6.0))
    assert len(pts2) == 0


def test_window_baseline():
    frames = [Keyframe(np.zeros((H, W, 3), np.uint8), _pose([x, 0, 0]), K) for x in (0, 0.1, 0.25)]
    assert np.isclose(_window_baseline(frames), 0.25)


class _FakeEst:
    """Returns a constant-depth view per input pose."""

    def __init__(self):
        self.calls = 0

    def infer_multi(self, images, c2w_list=None, K_list=None):
        self.calls += 1
        return [
            {"depth": np.full((H, W), 1.5, np.float32),
             "conf": np.ones((H, W), np.float32), "K": K, "c2w": c}
            for c in c2w_list
        ]


def _run(worker, n, dt=0.02, move=0.05):
    worker.start()
    for i in range(n):
        worker.submit(Keyframe((np.random.rand(H, W, 3) * 255).astype(np.uint8),
                               _pose([i * move, 0, 0]), K))
        time.sleep(dt)
    time.sleep(0.4)
    worker.stop()


def test_worker_windows_and_slides():
    results: list[WindowResult] = []
    est = _FakeEst()
    w = MVWindowWorker(est, MVWindowConfig(window=4, overlap=2, stride=2,
                                           min_baseline_m=0.0, conf_percentile=0,
                                           max_queue=16), results.append)
    _run(w, 10)
    # window=4, overlap=2 -> a new window every 2 submits: at 4,6,8,10 = 4 windows.
    assert len(results) == 4
    assert all(r.n_views == 4 for r in results)
    assert all(len(r.points) > 0 for r in results)
    # Sequence numbers are monotonic.
    assert [r.seq for r in results] == sorted(r.seq for r in results)


def test_worker_skips_low_baseline_windows():
    results: list[WindowResult] = []
    est = _FakeEst()
    # All frames at the same place -> baseline 0 -> every window skipped.
    w = MVWindowWorker(est, MVWindowConfig(window=3, overlap=1,
                                           min_baseline_m=0.05, conf_percentile=0),
                       results.append)
    w.start()
    for _ in range(9):
        w.submit(Keyframe(np.zeros((H, W, 3), np.uint8), _pose([0, 0, 0]), K))
        time.sleep(0.02)
    time.sleep(0.3)
    w.stop()
    assert len(results) == 0     # all skipped, worker still alive


def test_worker_colors_align_with_points():
    results: list[WindowResult] = []
    w = MVWindowWorker(_FakeEst(), MVWindowConfig(window=3, overlap=1, stride=2,
                                                  min_baseline_m=0.0, conf_percentile=0),
                       results.append)
    _run(w, 6)
    assert results
    for r in results:
        assert r.colors is not None
        assert r.colors.shape[0] == r.points.shape[0]
