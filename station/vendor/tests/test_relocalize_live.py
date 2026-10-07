"""Tests for xr_nav.relocalize_live.

Runs without pytest installed::

    python station/vendor/tests/test_relocalize_live.py

Two layers:
  * gating logic (sparse/empty/fitness rejection, success shape) — deterministic,
    via a fake ``relocalize`` injected into sys.modules so it doesn't depend on
    the stochastic RANSAC.
  * one real end-to-end synthetic-room solve through the actual relocalizer.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # station/vendor -> `import xr_nav`
if os.environ.get("DIMOS_DIR"):            # so `import dimos` works for the real test
    sys.path.insert(0, os.environ["DIMOS_DIR"])

from xr_nav.reference_map import ReferenceMap   # noqa: E402
from xr_nav.relocalize_live import run_relocalization, _nn_correspondences  # noqa: E402


def _reference_from_points(pts: np.ndarray) -> ReferenceMap:
    cols = np.tile(np.array([0.5, 0.5, 0.5], np.float32), (len(pts), 1))
    return ReferenceMap(points=pts.astype(np.float32), colors=cols,
                        voxel_state={}, voxel_size=0.05)


@contextlib.contextmanager
def _fake_relocalize(T, fitness):
    """Inject a deterministic ``relocalize`` so gating tests don't run RANSAC."""
    path = "dimos.mapping.relocalization.relocalize"
    saved = {k: sys.modules.get(k) for k in
             ("dimos", "dimos.mapping", "dimos.mapping.relocalization", path)}
    for pkg in ("dimos", "dimos.mapping", "dimos.mapping.relocalization"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    fake = types.ModuleType(path)
    fake.relocalize = lambda g, l: (np.asarray(T, float), float(fitness))
    sys.modules[path] = fake
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _rand_box(n, lo, hi, rng):
    return rng.uniform(lo, hi, size=(n, 3))


def test_rejects_sparse_scan():
    ref = _reference_from_points(np.random.default_rng(0).uniform(0, 5, (5000, 3)))
    res = run_relocalization(ref, np.zeros((10, 3)), min_local_points=2000)
    assert not res.success and res.method == "none" and "sparse" in res.message
    assert np.allclose(res.T_align, np.eye(4))
    print("ok: sparse scan rejected ->", res.message)


def test_rejects_empty_reference():
    ref = _reference_from_points(np.zeros((0, 3)))
    res = run_relocalization(ref, np.random.default_rng(1).uniform(0, 5, (3000, 3)))
    assert not res.success and "empty" in res.message
    print("ok: empty reference rejected ->", res.message)


def test_fitness_gate_rejects_low():
    pts = np.random.default_rng(2).uniform(0, 5, (4000, 3))
    ref = _reference_from_points(pts)
    with _fake_relocalize(np.eye(4), fitness=0.20):
        res = run_relocalization(ref, pts, fitness_threshold=0.45)
    assert not res.success and res.fitness == 0.20 and "rejected" in res.message
    assert np.allclose(res.T_align, np.eye(4))   # alignment unchanged on reject
    print("ok: low fitness rejected ->", res.message)


def test_success_returns_transform_and_correspondences():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 5, (4000, 3))
    ref = _reference_from_points(pts)
    T = np.eye(4); T[:3, 3] = [0.1, 0.2, 0.3]
    # local = inv(T) @ ref, so applying the returned T aligns it back onto ref.
    local = (np.linalg.inv(T)[:3, :3] @ pts.T).T + np.linalg.inv(T)[:3, 3]
    with _fake_relocalize(T, fitness=0.80):
        res = run_relocalization(ref, local, fitness_threshold=0.45)
    assert res.success and res.method == "geometric" and res.fitness == 0.80
    assert np.allclose(res.T_align, T)
    assert res.correspondences.ndim == 3 and res.correspondences.shape[1:] == (2, 3)
    assert len(res.correspondences) > 0   # ref==aligned-local here, so NN pairs exist
    print(f"ok: success path -> {len(res.correspondences)} correspondences, "
          f"fitness={res.fitness}")


def _synthetic_room(rng, spacing=0.06):
    """An asymmetric, z-up room: floor + 4 walls + one corner cabinet."""
    def grid(xr, yr, zr):
        xs = np.arange(*xr, spacing); ys = np.arange(*yr, spacing); zs = np.arange(*zr, spacing)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    parts = [
        grid((0, 4), (0, 3), (0, spacing)),          # floor
        grid((0, spacing), (0, 3), (0, 2.4)),        # wall x=0
        grid((4 - spacing, 4), (0, 3), (0, 2.4)),    # wall x=4
        grid((0, 4), (0, spacing), (0, 2.4)),        # wall y=0
        grid((0, 4), (3 - spacing, 3), (0, 2.4)),    # wall y=3
        grid((3.0, 3.8), (0.2, 1.0), (0, 1.5)),      # asymmetric cabinet (breaks yaw symmetry)
    ]
    pts = np.concatenate(parts, 0)
    pts += rng.normal(0, 0.004, pts.shape)           # mild sensor noise
    return pts


def test_real_synthetic_room_alignment():
    """End-to-end through the actual FPFH+RANSAC+ICP relocalizer (slow, seeded)."""
    try:
        import open3d as o3d
        from dimos.mapping.relocalization.relocalize import relocalize  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIP real test (deps unavailable): {e}")
        return
    with contextlib.suppress(Exception):
        o3d.utility.random.seed(0)

    rng = np.random.default_rng(7)
    room = _synthetic_room(rng)
    ref = _reference_from_points(room)

    yaw = np.deg2rad(15.0)
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    T_gt[:3, 3] = [0.4, -0.3, 0.0]
    A = np.linalg.inv(T_gt)
    local = (A[:3, :3] @ room.T).T + A[:3, 3]        # room as seen in the moved session frame

    res = run_relocalization(ref, local, fitness_threshold=0.30, min_local_points=1000)
    print(f"   real solve: success={res.success} fitness={res.fitness:.3f} "
          f"t={res.T_align[:3,3].round(3).tolist()}")
    assert res.success, f"expected success, got: {res.message}"

    # Primary checks: did we recover the known transform?
    t_err = float(np.linalg.norm(res.T_align[:3, 3] - T_gt[:3, 3]))
    R_err = res.T_align[:3, :3] @ T_gt[:3, :3].T
    rot_err = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
    # Proper alignment RMSE: NN to the *full* room via KDTree (no target subsampling).
    aligned = (res.T_align[:3, :3] @ local.T).T + res.T_align[:3, 3]
    room_pcd = o3d.geometry.PointCloud()
    room_pcd.points = o3d.utility.Vector3dVector(room)
    kdt = o3d.geometry.KDTreeFlann(room_pcd)
    d2 = [kdt.search_knn_vector_3d(p, 1)[2][0] for p in aligned[::17]]
    rmse = float(np.sqrt(np.mean(d2)))
    print(f"   recovery: translation_err={t_err:.3f}m  rotation_err={rot_err:.2f}°  "
          f"alignment_rmse={rmse:.3f}m")
    assert t_err < 0.08, f"translation error too high: {t_err:.3f}m"
    assert rot_err < 4.0, f"rotation error too high: {rot_err:.2f}°"
    assert rmse < 0.05, f"alignment rmse too high: {rmse:.3f}m"
    print("ok: real synthetic room aligned")


def test_visual_seed_recovers_transform():
    """build_feature_db + _visual_seed: ORB match -> depth backproject -> 3D-3D fit.

    Reference and live use the SAME synthetic textured image (perfect matches) but
    different poses; the seed must recover T (live -> map) = inv(live_pose)."""
    import tempfile

    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        print(f"SKIP visual seed test (cv2 unavailable): {e}")
        return
    from xr_nav.reference_map import ReferenceMap, ReferenceKeyframe
    from xr_nav.relocalize_live import _visual_seed, LiveKeyframe

    rng = np.random.default_rng(11)
    H, W = 240, 320
    img = rng.integers(0, 255, (H, W), dtype=np.uint8)
    for _ in range(40):  # blobs give ORB stable, well-localized corners
        cx, cy = int(rng.integers(20, W - 20)), int(rng.integers(20, H - 20))
        cv2.circle(img, (cx, cy), int(rng.integers(5, 15)), int(rng.integers(0, 255)), -1)
    depth = (np.linspace(1.0, 3.0, H, dtype=np.float32)[:, None]
             * np.ones((1, W), np.float32))
    fx = 250.0
    K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]], dtype=np.float64)

    d = Path(tempfile.mkdtemp())
    (d / "rgb").mkdir(); (d / "depth").mkdir()
    cv2.imwrite(str(d / "rgb/kf_000000.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(d / "depth/kf_000000.png"), (depth * 1000).astype(np.uint16))

    kf = ReferenceKeyframe(idx=0, pose_4x4=np.eye(4), intrinsics=K, image_hw=(H, W),
                           rgb_path=d / "rgb/kf_000000.png",
                           depth_path=d / "depth/kf_000000.png")
    ref = ReferenceMap(points=np.zeros((0, 3), np.float32),
                       colors=np.zeros((0, 3), np.float32),
                       voxel_state={}, voxel_size=0.05, keyframes=[kf])
    assert ref.build_feature_db() == 1 and ref.has_keyframe_features

    yaw = np.deg2rad(10.0); c, s = np.cos(yaw), np.sin(yaw)
    M = np.eye(4)
    M[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    M[:3, 3] = [0.2, -0.1, 0.05]
    live = LiveKeyframe(rgb=img, depth=depth, c2w=M, K=K)

    seed = _visual_seed(ref, [live], min_inliers=8)
    assert seed is not None, "visual seed returned None"
    aligned = (seed.T[:3, :3] @ seed.live_inlier_xyz.T).T + seed.T[:3, 3]
    rmse = float(np.sqrt(np.mean(np.sum((aligned - seed.ref_inlier_xyz) ** 2, axis=1))))
    t_err = float(np.linalg.norm(seed.T[:3, 3] - np.linalg.inv(M)[:3, 3]))
    print(f"   visual seed: inliers={seed.n_inliers} rmse={rmse:.3f}m "
          f"translation_err={t_err:.3f}m")
    assert seed.n_inliers >= 8
    assert rmse < 0.03, f"visual fit rmse too high: {rmse:.3f}m"
    assert t_err < 0.05, f"recovered translation off by {t_err:.3f}m"
    print("ok: visual seed recovered transform")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
