"""Phase-B tests: frame-to-map ICP recovers a perturbed pose.

The live loop trusts kinematic poses but folds in a small ICP correction
against the accumulated map to absorb backlash / servo lag / calibration
error. These tests build a synthetic room (two walls + floor), perturb the
frame pose by a few cm / deg, and assert ICP recovers it — for both
point-to-point and point-to-plane estimators — plus the correction-damping
and target-caching helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from xr_nav.icp import (
    ICPConfig,
    align_frame_to_map_cached,
    apply_correction,
    build_target,
    damp_correction,
)


def _room(n_per_face: int = 4000, seed: int = 0) -> np.ndarray:
    """Two perpendicular walls + a floor, mild noise — a room-shaped target."""
    rng = np.random.default_rng(seed)
    # floor z=0
    floor = np.column_stack([rng.uniform(-2, 2, n_per_face),
                             rng.uniform(-2, 2, n_per_face),
                             rng.normal(0, 0.004, n_per_face)])
    # wall x=-2 (facing +x)
    wall_x = np.column_stack([np.full(n_per_face, -2.0) + rng.normal(0, 0.004, n_per_face),
                              rng.uniform(-2, 2, n_per_face),
                              rng.uniform(0, 2, n_per_face)])
    # wall y=-2 (facing +y)
    wall_y = np.column_stack([rng.uniform(-2, 2, n_per_face),
                              np.full(n_per_face, -2.0) + rng.normal(0, 0.004, n_per_face),
                              rng.uniform(0, 2, n_per_face)])
    return np.vstack([floor, wall_x, wall_y]).astype(np.float32)


def _perturb(t=(0.0, 0.0, 0.0), yaw_deg=0.0) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


@pytest.mark.parametrize("estimation", ["point_to_point", "point_to_plane"])
def test_icp_recovers_small_perturbation(estimation):
    map_pts = _room()
    # The "frame" is the same geometry seen from a slightly wrong pose: shift
    # it 3 cm and rotate 2 deg, then check ICP pulls it back onto the map.
    bad = _perturb(t=(0.03, -0.02, 0.01), yaw_deg=2.0)
    frame_pts = apply_correction(map_pts, bad)

    cfg = ICPConfig(translation_only=False, estimation=estimation,
                    max_correspondence_distance=0.15, fitness_threshold=0.2)
    tgt = build_target(map_pts, cfg)
    T_corr, fitness = align_frame_to_map_cached(frame_pts, map_pts, cfg, target_pcd=tgt)

    assert fitness > 0.5
    # Correcting the frame should bring its mean back near the map's mean.
    fixed = apply_correction(frame_pts, T_corr)
    err_before = np.linalg.norm(frame_pts.mean(0) - map_pts.mean(0))
    err_after = np.linalg.norm(fixed.mean(0) - map_pts.mean(0))
    assert err_after < err_before
    assert err_after < 0.01     # within a cm


def test_icp_identity_when_aligned():
    map_pts = _room()
    cfg = ICPConfig(translation_only=False, estimation="point_to_plane",
                    max_correspondence_distance=0.15)
    tgt = build_target(map_pts, cfg)
    T_corr, fitness = align_frame_to_map_cached(map_pts, map_pts, cfg, target_pcd=tgt)
    # Already aligned -> near-identity correction.
    assert np.allclose(T_corr[:3, 3], 0.0, atol=0.005)


def test_build_target_none_when_too_small():
    cfg = ICPConfig(translation_only=False, min_map_points=500)
    assert build_target(np.zeros((10, 3), np.float32), cfg) is None


def test_damp_correction_scales_toward_identity():
    T = _perturb(t=(0.10, 0.0, 0.0), yaw_deg=10.0)
    half = damp_correction(T, 0.5)
    # Half the translation.
    assert np.allclose(half[:3, 3], [0.05, 0.0, 0.0], atol=1e-9)
    # Half the rotation angle.
    from scipy.spatial.transform import Rotation
    ang = np.degrees(Rotation.from_matrix(half[:3, :3]).magnitude())
    assert abs(ang - 5.0) < 1e-6
    # alpha=1 is a no-op.
    assert np.allclose(damp_correction(T, 1.0), T)
    # alpha=0 is identity.
    assert np.allclose(damp_correction(T, 0.0), np.eye(4))


def test_damped_icp_converges_over_iterations():
    """Applying a damped correction repeatedly should still converge — the
    low-pass slows each step but doesn't stall."""
    map_pts = _room()
    cfg = ICPConfig(translation_only=False, estimation="point_to_plane",
                    max_correspondence_distance=0.2, fitness_threshold=0.2)
    tgt = build_target(map_pts, cfg)
    frame = apply_correction(map_pts, _perturb(t=(0.05, 0.04, 0.0), yaw_deg=3.0))
    err0 = np.linalg.norm(frame.mean(0) - map_pts.mean(0))
    for _ in range(6):
        T_corr, fit = align_frame_to_map_cached(frame, map_pts, cfg, target_pcd=tgt)
        frame = apply_correction(frame, damp_correction(T_corr, 0.5))
    err1 = np.linalg.norm(frame.mean(0) - map_pts.mean(0))
    assert err1 < err0
    assert err1 < 0.01
