"""Frame-to-model alignment to correct VIO drift.

Supports two modes:
  - Full ICP (Open3D point-to-point) — rigid rotation + translation
  - Translation-only — median NN offset, for cases where VIO rotation
    is reliable but translation drifts

Includes a DriftTracker that detects systematic translation bias
over a sliding window of recent corrections.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


@dataclass
class ICPConfig:
    """Configuration for frame-to-model alignment."""

    enabled: bool = True
    translation_only: bool = True    # constrain to translation-only correction
    max_correspondence_distance: float = 0.20  # meters
    max_iteration: int = 30          # for full ICP mode
    fitness_threshold: float = 0.3   # reject if fitness below this
    min_map_points: int = 500
    nn_subsample: int = 2000         # subsample frame pts for translation-only
    # Full-ICP estimation method: "point_to_point" or "point_to_plane". The
    # latter converges better on the walls/floor that dominate a room scan
    # (it slides along surfaces instead of fighting tangential mismatch) but
    # needs target normals. Ignored in translation_only mode.
    estimation: str = "point_to_point"
    target_voxel: float = 0.03       # downsample target before ICP (m); 0 = off
    normal_radius: float = 0.10      # normal-estimation radius for point_to_plane


class DriftTracker:
    """Track cumulative translation corrections to detect systematic drift."""

    def __init__(self, window_size: int = 10) -> None:
        self._corrections: deque[NDArray[np.float64]] = deque(maxlen=window_size)

    def add(self, T_corr: NDArray[np.float64]) -> None:
        self._corrections.append(T_corr[:3, 3].copy())

    def systematic_drift(self) -> NDArray[np.float64]:
        """Return the average correction vector (systematic bias)."""
        if len(self._corrections) < 3:
            return np.zeros(3, dtype=np.float64)
        return np.mean(list(self._corrections), axis=0)


def align_frame_to_map(
    frame_pts: NDArray[np.float32],
    map_pts: NDArray[np.float32],
    config: ICPConfig,
) -> tuple[NDArray[np.float64], float]:
    """Align a frame's world-space points to the existing map.

    Dispatches to translation-only or full ICP based on config.

    Returns:
        (T_corr, fitness) where T_corr is a 4x4 correction matrix.
    """
    if config.translation_only:
        return _align_translation_only(frame_pts, map_pts, config)
    return _align_icp(frame_pts, map_pts, config)


def _align_translation_only(
    frame_pts: NDArray[np.float32],
    map_pts: NDArray[np.float32],
    config: ICPConfig,
) -> tuple[NDArray[np.float64], float]:
    """Translation-only correction via median NN offset."""
    identity = np.eye(4, dtype=np.float64)

    if len(map_pts) < config.min_map_points:
        return identity, 0.0

    # Subsample frame points
    n = min(config.nn_subsample, len(frame_pts))
    idx = np.random.choice(len(frame_pts), n, replace=False)
    src = frame_pts[idx]

    tree = KDTree(map_pts)
    dists, nn_idx = tree.query(src, k=1)

    mask = dists < config.max_correspondence_distance
    fitness = float(mask.sum()) / len(src)

    if fitness < config.fitness_threshold:
        return identity, fitness

    # Median offset is robust to outliers
    offsets = map_pts[nn_idx[mask]] - src[mask]
    t_corr = np.median(offsets, axis=0)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = t_corr.astype(np.float64)
    return T, fitness


def _align_icp(
    frame_pts: NDArray[np.float32],
    map_pts: NDArray[np.float32],
    config: ICPConfig,
    target_pcd=None,
) -> tuple[NDArray[np.float64], float]:
    """Full rigid ICP via Open3D.

    ``target_pcd`` may be a pre-built (downsampled, normal-carrying) Open3D
    target to avoid rebuilding it every frame — see :func:`build_target`.
    """
    import open3d as o3d  # type: ignore[import-untyped]

    identity = np.eye(4, dtype=np.float64)

    if len(map_pts) < config.min_map_points:
        return identity, 0.0

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(frame_pts.astype(np.float64))

    point_to_plane = config.estimation == "point_to_plane"
    tgt = target_pcd if target_pcd is not None else build_target(map_pts, config)
    if tgt is None or len(tgt.points) < config.min_map_points:
        return identity, 0.0

    if point_to_plane:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance=config.max_correspondence_distance,
        init=identity,
        estimation_method=estimator,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config.max_iteration,
        ),
    )

    fitness = result.fitness
    T_corr = np.asarray(result.transformation, dtype=np.float64)

    if fitness < config.fitness_threshold:
        return identity, fitness

    return T_corr, fitness


def build_target(map_pts: NDArray[np.float32], config: ICPConfig):
    """Build an Open3D target cloud (optionally voxel-downsampled + normals).

    Cache this across frames and rebuild every N inserts — the map churns
    slowly under ``--no-prune``, so a per-frame rebuild is wasted work. Returns
    None if the map is too small.
    """
    import open3d as o3d  # type: ignore[import-untyped]

    if len(map_pts) < config.min_map_points:
        return None
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(map_pts.astype(np.float64))
    if config.target_voxel > 0:
        tgt = tgt.voxel_down_sample(config.target_voxel)
    if config.estimation == "point_to_plane":
        tgt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.normal_radius, max_nn=30
            )
        )
    return tgt


def align_frame_to_map_cached(
    frame_pts: NDArray[np.float32],
    map_pts: NDArray[np.float32],
    config: ICPConfig,
    target_pcd=None,
) -> tuple[NDArray[np.float64], float]:
    """Like :func:`align_frame_to_map` but accepts a pre-built ICP target.

    Translation-only mode ignores ``target_pcd`` (it uses a scipy KDTree).
    """
    if config.translation_only:
        return _align_translation_only(frame_pts, map_pts, config)
    return _align_icp(frame_pts, map_pts, config, target_pcd=target_pcd)


def damp_correction(
    T_corr: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]:
    """Scale a rigid correction toward identity by ``alpha`` in [0, 1].

    Low-passes the per-frame ICP correction so a single noisy solve can't jerk
    the whole map: slerp the rotation from identity and scale the translation.
    """
    from scipy.spatial.transform import Rotation

    if alpha >= 1.0:
        return T_corr
    rotvec = Rotation.from_matrix(T_corr[:3, :3]).as_rotvec()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_rotvec(rotvec * alpha).as_matrix()
    out[:3, 3] = T_corr[:3, 3] * alpha
    return out


def apply_correction(
    pts: NDArray[np.float32],
    T_corr: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Apply a 4x4 rigid correction to world-space points."""
    rot = T_corr[:3, :3].astype(np.float32)
    trans = T_corr[:3, 3].astype(np.float32)
    return (rot @ pts.T).T + trans
