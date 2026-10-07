"""Per-frame depth scale alignment.

Computes an optimal scale factor so that a frame's depth, when
projected to world space, best matches the existing voxel map.
DA3 produces relative depth whose absolute scale can vary per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


@dataclass
class ScaleAlignConfig:
    """Configuration for per-frame depth scale alignment."""

    enabled: bool = True
    min_map_points: int = 500
    max_correspondence_distance: float = 0.50  # meters
    nn_subsample: int = 1000
    min_scale: float = 0.5
    max_scale: float = 2.0
    min_correspondences: int = 50


def compute_depth_scale(
    pts_cam: NDArray[np.float32],
    rot: NDArray[np.float32],
    trans: NDArray[np.float32],
    map_pts: NDArray[np.float32],
    config: ScaleAlignConfig,
) -> tuple[float, int]:
    """Compute optimal depth scale to align frame points to map.

    The scale multiplies camera-space points before the world transform:
        pts_world = R @ (s * pts_cam) + t

    For each NN correspondence (frame_pt, map_pt):
        d_frame = ||R @ pts_cam[i]||   (ray length at scale=1)
        d_map   = ||map_pt - t||       (map point distance from camera)
        ratio   = d_map / d_frame

    scale = median(ratios) — robust to outliers.

    Args:
        pts_cam: [N, 3] points in camera frame (unscaled).
        rot: [3, 3] rotation (pose[:3,:3] after ARKit→OpenCV flip).
        trans: [3] translation (pose[:3, 3]).
        map_pts: [M, 3] existing map points in world frame.
        config: Scale alignment parameters.

    Returns:
        (scale, num_correspondences). scale=1.0 if alignment skipped.
    """
    if len(map_pts) < config.min_map_points:
        return 1.0, 0

    # Project at scale=1 to get world-space points
    pts_world = (rot @ pts_cam.T).T + trans

    # Subsample for speed
    n = min(config.nn_subsample, len(pts_world))
    idx = np.random.choice(len(pts_world), n, replace=False)
    sub_world = pts_world[idx]
    sub_cam = pts_cam[idx]

    # NN correspondences
    tree = KDTree(map_pts)
    dists, nn_idx = tree.query(sub_world, k=1)

    mask = dists < config.max_correspondence_distance
    n_corr = int(mask.sum())
    if n_corr < config.min_correspondences:
        return 1.0, n_corr

    # Compute per-correspondence scale ratio
    matched_map = map_pts[nn_idx[mask]]
    matched_cam = sub_cam[mask]

    # Ray length at scale=1 in world frame
    rays_world = (rot @ matched_cam.T).T
    d_frame = np.linalg.norm(rays_world, axis=1)
    # Map point distance from camera center
    d_map = np.linalg.norm(matched_map - trans, axis=1)

    valid = d_frame > 1e-6
    if valid.sum() < config.min_correspondences:
        return 1.0, int(valid.sum())

    ratios = d_map[valid] / d_frame[valid]
    scale = float(np.median(ratios))

    # Clamp to reasonable range
    if scale < config.min_scale or scale > config.max_scale:
        return 1.0, n_corr

    return scale, n_corr
