"""One-shot relocalization of a live scan against a saved reference map.

Wraps the geometric relocalizer (``dimos.mapping.relocalization.relocalize``)
with the bits a live, fixed-base scanner needs: build an Open3D cloud from the
accumulated scan, gate the result on registration fitness, and surface point
correspondences for the Rerun overlay. The visual keyframe layer (Phase B) seeds
this solve via ``seed_T`` and contributes its own 2D-feature correspondences.

``relocalize(global_map, local_map)`` returns ``T`` mapping *local -> global*,
i.e. session -> map, which is exactly the ``T_align`` the pipeline applies to the
live pose so the moved robot localizes inside the saved map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from xr_nav.reference_map import ReferenceMap

# Floor below which a scan is too sparse to register reliably. Far lower than the
# dimos RelocalizationModule's 50k (that's dense Go2 lidar); a fixed-base Reachy
# head-sweep at a 0.03 m voxel yields a few thousand points and fitness gates the
# rest.
DEFAULT_MIN_LOCAL_POINTS = 2_000
# Correspondence search radius for the viz overlay (≈ relocalize.py's FINE_VOXEL×1.5).
CORRESPONDENCE_MAX_DIST = 0.15
MAX_CORRESPONDENCE_LINES = 400


@dataclass
class LiveKeyframe:
    """One live frame captured during the recalibration sweep (session frame)."""

    rgb: NDArray[np.uint8]             # [H, W, 3] RGB
    depth: NDArray[np.float32]         # [H, W] metres
    c2w: NDArray[np.float64]           # 4x4 camera-to-world, session frame
    K: NDArray[np.float64]             # 3x3 intrinsics


@dataclass
class _VisualSeed:
    T: NDArray[np.float64]                       # 4x4 session -> map
    n_inliers: int
    live_inlier_xyz: NDArray[np.float64]         # [K, 3] session frame
    ref_inlier_xyz: NDArray[np.float64]          # [K, 3] map frame
    ref_kf_idx: int


@dataclass
class RelocResult:
    success: bool
    T_align: NDArray[np.float64]                 # 4x4 session -> map (identity on failure)
    fitness: float
    method: str                                  # "geometric" | "visual+icp" | "seeded-icp"
    message: str
    # [M, 2, 3] pairs of (aligned-local, reference) points in the map frame, for viz.
    correspondences: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 2, 3), dtype=np.float64)
    )
    n_local_points: int = 0
    n_visual_inliers: int = 0


def run_relocalization(
    reference: ReferenceMap,
    local_xyz: NDArray[np.floating],
    *,
    local_colors: NDArray[np.floating] | None = None,
    fitness_threshold: float = 0.45,
    min_local_points: int = DEFAULT_MIN_LOCAL_POINTS,
    seed_T: NDArray[np.floating] | None = None,
    live_keyframes: list | None = None,  # Phase B: visual-seed inputs
) -> RelocResult:
    """Align ``local_xyz`` (session frame) onto ``reference`` (map frame).

    Phase A path is purely geometric. ``seed_T`` / ``live_keyframes`` are accepted
    now so the Phase B visual layer can slot in without changing call sites.
    """
    identity = np.eye(4, dtype=np.float64)
    n_local = int(len(local_xyz))
    if n_local < min_local_points:
        return RelocResult(
            success=False, T_align=identity, fitness=0.0, method="none",
            message=f"scan too sparse ({n_local} < {min_local_points} pts) — sweep more",
            n_local_points=n_local,
        )
    if len(reference) == 0:
        return RelocResult(
            success=False, T_align=identity, fitness=0.0, method="none",
            message="reference map is empty", n_local_points=n_local,
        )

    import open3d as o3d

    local = o3d.geometry.PointCloud()
    local.points = o3d.utility.Vector3dVector(np.asarray(local_xyz, dtype=np.float64))
    if local_colors is not None and len(local_colors) == n_local:
        local.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(local_colors, dtype=np.float64), 0.0, 1.0)
        )

    # --- Visual keyframe seed (Phase B) -----------------------------------
    # Match live frames against the reference keyframes, backproject matched
    # features to 3D via metric depth, and solve a 3D-3D rigid fit. A good seed
    # lets a cheap seeded ICP replace the expensive cold FPFH search and yields
    # the literal image-feature correspondences to draw across the two frames.
    visual = None
    if live_keyframes and reference.has_keyframe_features:
        try:
            visual = _visual_seed(reference, live_keyframes)
        except Exception as e:  # noqa: BLE001
            print(f"[reloc] visual seed failed: {e}")
    seed = (np.asarray(seed_T, dtype=np.float64) if seed_T is not None
            else (visual.T if visual is not None else None))

    if seed is not None:
        T_icp, fit_icp = _seeded_icp(reference.o3d_cloud, local, seed)
        if fit_icp >= fitness_threshold:
            method = "visual+icp" if visual is not None else "seeded-icp"
            corr = (_visual_correspondences(visual, T_icp) if visual is not None
                    else _nn_correspondences(np.asarray(local_xyz, np.float64),
                                             reference.points.astype(np.float64), T_icp))
            n_inl = visual.n_inliers if visual is not None else 0
            return RelocResult(
                success=True, T_align=T_icp, fitness=fit_icp, method=method,
                message=(f"aligned via {method} (fitness {fit_icp:.3f}"
                         + (f", {n_inl} visual inliers" if visual is not None else "") + ")"),
                correspondences=corr, n_local_points=n_local, n_visual_inliers=n_inl,
            )
        print(f"[reloc] seed rejected by ICP (fitness {fit_icp:.3f} < "
              f"{fitness_threshold:.2f}); falling back to FPFH")

    # --- Geometric fallback (Phase A: cold FPFH + RANSAC + ICP) -----------
    try:
        from dimos.mapping.relocalization.relocalize import relocalize
    except Exception as e:  # noqa: BLE001
        return RelocResult(
            success=False, T_align=identity, fitness=0.0, method="none",
            message=f"relocalize import failed: {e}", n_local_points=n_local,
        )

    try:
        T, fitness = relocalize(reference.o3d_cloud, local)
    except Exception as e:  # noqa: BLE001
        return RelocResult(
            success=False, T_align=identity, fitness=0.0, method="geometric",
            message=f"relocalize() raised: {e}", n_local_points=n_local,
        )

    T = np.asarray(T, dtype=np.float64)
    fitness = float(fitness)
    if fitness < fitness_threshold:
        return RelocResult(
            success=False, T_align=identity, fitness=fitness, method="geometric",
            message=f"fitness {fitness:.3f} < {fitness_threshold:.2f} — rejected",
            n_local_points=n_local,
        )

    corr = _nn_correspondences(
        np.asarray(local_xyz, dtype=np.float64), reference.points.astype(np.float64), T
    )
    return RelocResult(
        success=True, T_align=T, fitness=fitness, method="geometric",
        message=f"aligned (fitness {fitness:.3f}, {len(corr)} correspondences)",
        correspondences=corr, n_local_points=n_local,
    )


def _nn_correspondences(
    local_xyz: NDArray[np.float64],
    reference_xyz: NDArray[np.float64],
    T: NDArray[np.float64],
    max_dist: float = CORRESPONDENCE_MAX_DIST,
    max_lines: int = MAX_CORRESPONDENCE_LINES,
) -> NDArray[np.float64]:
    """Mutual-ish nearest-neighbour pairs between the aligned scan and reference.

    Returns ``[M, 2, 3]`` (aligned-local point, matched reference point), both in
    the map frame — the inliers that make the alignment visible as short link lines.
    """
    if len(local_xyz) == 0 or len(reference_xyz) == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)

    import open3d as o3d

    aligned = (T[:3, :3] @ local_xyz.T).T + T[:3, 3]
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(reference_xyz)
    kdt = o3d.geometry.KDTreeFlann(ref_pcd)

    # Sample the aligned scan so the overlay stays light regardless of scan size.
    step = max(1, len(aligned) // (max_lines * 3))
    pairs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for p in aligned[::step]:
        k, idx, d2 = kdt.search_knn_vector_3d(p, 1)
        if k == 1 and d2[0] <= max_dist * max_dist:
            pairs.append((p, reference_xyz[idx[0]]))
            if len(pairs) >= max_lines:
                break
    if not pairs:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.asarray([[a, b] for a, b in pairs], dtype=np.float64)


def _visual_seed(
    reference: ReferenceMap,
    live_keyframes: list,
    *,
    ratio: float = 0.75,
    min_inliers: int = 12,
    ransac_max_dist: float = 0.10,
) -> "_VisualSeed | None":
    """Match live frames to reference keyframes and solve a 3D-3D rigid seed.

    ORB + Lowe-ratio matching picks the best reference keyframe; matched
    keypoints backproject to 3D (live via metric depth, reference precomputed),
    and a 3D-3D RANSAC (Umeyama) recovers the session -> map seed transform.
    """
    import cv2
    import open3d as o3d

    from xr_nav.reference_map import backproject_pixels_to_world

    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best: tuple[int, NDArray, NDArray, int] | None = None
    for lkf in live_keyframes:
        rgb = np.asarray(lkf.rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        kps, desc = orb.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            continue
        live_xy = np.array([kp.pt for kp in kps], dtype=np.float32)
        live_xyz, live_valid = backproject_pixels_to_world(
            live_xy, np.asarray(lkf.depth), lkf.K, lkf.c2w)
        for kf in reference.keyframes:
            feats = kf.features
            if feats is None or len(feats.descriptors) < 2:
                continue
            good = []
            for m_n in bf.knnMatch(desc, feats.descriptors, k=2):
                if len(m_n) != 2:
                    continue
                m, nn = m_n
                if m.distance < ratio * nn.distance and live_valid[m.queryIdx]:
                    good.append(m)
            if len(good) < min_inliers:
                continue
            if best is None or len(good) > best[0]:
                lp = np.array([live_xyz[m.queryIdx] for m in good])
                rp = np.array([feats.points3d_world[m.trainIdx] for m in good])
                best = (len(good), lp, rp, kf.idx)

    if best is None:
        return None
    _, lp, rp, ref_idx = best

    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(lp)
    dst = o3d.geometry.PointCloud(); dst.points = o3d.utility.Vector3dVector(rp)
    corres = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(len(lp)), np.arange(len(lp))]).astype(np.int32))
    reg = o3d.pipelines.registration
    result = reg.registration_ransac_based_on_correspondence(
        src, dst, corres, ransac_max_dist,
        reg.TransformationEstimationPointToPoint(False), 3,
        criteria=reg.RANSACConvergenceCriteria(100_000, 0.999))
    inliers = np.asarray(result.correspondence_set)
    if len(inliers) < min_inliers:
        return None
    return _VisualSeed(
        T=np.asarray(result.transformation, dtype=np.float64),
        n_inliers=int(len(inliers)),
        live_inlier_xyz=lp[inliers[:, 0]], ref_inlier_xyz=rp[inliers[:, 1]],
        ref_kf_idx=int(ref_idx),
    )


def _seeded_icp(target_map_cloud, source_local_cloud, seed_T, *,
                max_dist: float = 0.20, iters: int = 60) -> tuple[NDArray[np.float64], float]:
    """Point-to-point ICP from a seed. Returns (T session->map, fitness)."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    result = reg.registration_icp(
        source_local_cloud, target_map_cloud, max_dist,
        np.asarray(seed_T, dtype=np.float64),
        reg.TransformationEstimationPointToPoint(),
        reg.ICPConvergenceCriteria(max_iteration=iters))
    return np.asarray(result.transformation, dtype=np.float64), float(result.fitness)


def _visual_correspondences(visual: "_VisualSeed", T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inlier feature matches as [K, 2, 3] (aligned-live, reference), map frame."""
    if visual is None or len(visual.live_inlier_xyz) == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)
    aligned = (T[:3, :3] @ visual.live_inlier_xyz.T).T + T[:3, 3]
    return np.stack([aligned, visual.ref_inlier_xyz], axis=1)
