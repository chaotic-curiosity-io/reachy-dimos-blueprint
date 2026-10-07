"""Keyframe selection to reject redundant or too-drifted frames.

Only accepts frames that add new geometry to the map AND have
enough overlap with existing geometry to verify alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


@dataclass
class KeyframeConfig:
    """Configuration for keyframe selection."""

    enabled: bool = True
    min_baseline: float = 0.02       # min translation from last keyframe (m)
    min_rotation_deg: float = 2.0    # min rotation from last keyframe (deg)
    max_nn_distance: float = 0.15    # reject if median NN to map exceeds this (m)
    min_overlap_ratio: float = 0.20  # min fraction of points within overlap_radius
    overlap_radius: float = 0.15     # radius for overlap test (m)
    nn_subsample: int = 500          # subsample frame pts for NN queries
    min_map_points: int = 200        # skip overlap check until map has enough pts


class KeyframeSelector:
    """Stateful selector: tracks last accepted pose to enforce baseline/rotation gates."""

    def __init__(self, config: KeyframeConfig) -> None:
        self._cfg = config
        self._last_pose: NDArray[np.float64] | None = None
        self._accepted: int = 0

    def should_accept(
        self,
        pose_4x4: NDArray[np.float64],
        pts_world: NDArray[np.float32],
        map_pts: NDArray[np.float32],
    ) -> tuple[bool, str]:
        """Decide whether to accept this frame.

        Args:
            pose_4x4: Camera-to-world (already flipped to OpenCV convention).
            pts_world: This frame's filtered world-space points.
            map_pts: Current voxel map points.

        Returns:
            (accepted, reason) for logging.
        """
        cfg = self._cfg

        # Bootstrap: always accept the first few frames
        if self._accepted < 2 or len(map_pts) < cfg.min_map_points:
            return True, "bootstrap"

        # --- Motion gate ---
        if self._last_pose is not None:
            dt = np.linalg.norm(pose_4x4[:3, 3] - self._last_pose[:3, 3])
            r_prev = Rotation.from_matrix(self._last_pose[:3, :3])
            r_curr = Rotation.from_matrix(pose_4x4[:3, :3])
            dr_deg = np.degrees((r_prev.inv() * r_curr).magnitude())

            if dt < cfg.min_baseline and dr_deg < cfg.min_rotation_deg:
                return False, f"too similar (dt={dt*1000:.0f}mm, dr={dr_deg:.1f}°)"

        # --- Overlap gate ---
        n_sample = min(cfg.nn_subsample, len(pts_world))
        if n_sample < 20:
            return False, "too few points"

        idx = np.random.choice(len(pts_world), n_sample, replace=False)
        sampled = pts_world[idx]

        tree = KDTree(map_pts)
        dists, _ = tree.query(sampled, k=1)
        median_nn = float(np.median(dists))
        overlap_ratio = float(np.mean(dists < cfg.overlap_radius))

        if median_nn > cfg.max_nn_distance:
            return False, f"too drifted (median_nn={median_nn:.3f}m)"

        if overlap_ratio < cfg.min_overlap_ratio:
            return False, f"low overlap ({overlap_ratio*100:.0f}%)"

        return True, f"ok (nn={median_nn:.3f}m, overlap={overlap_ratio*100:.0f}%)"

    def mark_accepted(self, pose_4x4: NDArray[np.float64]) -> None:
        """Record that this frame was accepted."""
        self._last_pose = pose_4x4.copy()
        self._accepted += 1
