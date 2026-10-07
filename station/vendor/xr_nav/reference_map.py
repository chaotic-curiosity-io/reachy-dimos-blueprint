"""A previously-saved map loaded as a fixed relocalization target.

``--load-map`` resumes a capture by merging saved voxels into the *live* world
frame, so it only lines up when the robot restarts at the same physical pose.
Relocalization instead needs the saved map held still as a registration target
while the new (moved) session is aligned onto it. ``ReferenceMap`` is that
target: the saved voxel cloud plus, optionally, the reference keyframes (RGB +
pose + intrinsics) that the visual feature layer matches live frames against.

Built on the shared bundle/keyframe schema:
  * map bundle  -> ``xr_nav.map_io.load_map_bundle`` + ``VoxelMap.load_state``
  * keyframes   -> the ``index.jsonl`` written by ``xr_nav.keyframe_recorder``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from xr_nav.map_io import load_map_bundle
from xr_nav.voxel_map import VoxelMap


@dataclass
class KeyframeFeatures:
    """ORB features of one reference keyframe, backprojected to the map frame."""

    keypoints_xy: NDArray[np.float32]   # [M, 2] pixel coords
    descriptors: NDArray[np.uint8]      # [M, 32] ORB descriptors
    points3d_world: NDArray[np.float64] # [M, 3] world/map-frame 3D (valid rows only)
    valid: NDArray[np.bool_]            # [M] had a usable depth


@dataclass
class ReferenceKeyframe:
    """One saved keyframe of the reference map.

    ``depth_path`` is populated only for keyframe sets captured with
    ``--keyframe-save-depth`` (Phase B); when absent the visual layer renders
    depth by projecting the reference cloud into this camera instead.
    """

    idx: int
    pose_4x4: NDArray[np.float64]      # camera-to-world (reference/map frame)
    intrinsics: NDArray[np.float64]    # 3x3 K
    image_hw: tuple[int, int]
    rgb_path: Path
    depth_path: Path | None = None
    timestamp: float = 0.0
    features: KeyframeFeatures | None = None


@dataclass
class ReferenceMap:
    """Fixed reference cloud (+ optional keyframes) to relocalize against."""

    points: NDArray[np.float32]        # [N, 3] in the saved map's world frame
    colors: NDArray[np.float32]        # [N, 3] RGB in [0, 1]
    voxel_state: dict                  # raw VoxelMap.to_state() dict (for --reloc-merge)
    voxel_size: float
    keyframes: list[ReferenceKeyframe] = field(default_factory=list)
    _o3d_cache: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: Path,
        keyframes_dir: Path | None = None,
        min_observations: int = 1,
    ) -> "ReferenceMap":
        bundle = load_map_bundle(bundle_path)
        vmap = VoxelMap()
        vmap.load_state(bundle["voxel_map"], replace=True)
        pts, cols = vmap.to_points_colored(min_observations=min_observations)
        keyframes = (
            load_reference_keyframes(keyframes_dir)
            if keyframes_dir is not None else []
        )
        print(
            f"[reference] {len(pts)} voxels"
            + (f", {len(keyframes)} keyframes" if keyframes else "")
            + f" (voxel_size={vmap.voxel_size:.3f}m) from {Path(bundle_path).name}"
        )
        return cls(
            points=pts,
            colors=cols,
            voxel_state=bundle["voxel_map"],
            voxel_size=float(vmap.voxel_size),
            keyframes=keyframes,
        )

    @property
    def o3d_cloud(self):  # type: ignore[no-untyped-def]
        """Open3D PointCloud of the reference (cached). Empty cloud if no points."""
        if self._o3d_cache is None:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            if len(self.points) > 0:
                pcd.points = o3d.utility.Vector3dVector(self.points.astype(np.float64))
                if len(self.colors) == len(self.points):
                    pcd.colors = o3d.utility.Vector3dVector(
                        np.clip(self.colors.astype(np.float64), 0.0, 1.0)
                    )
            self._o3d_cache = pcd
        return self._o3d_cache

    @property
    def has_keyframe_features(self) -> bool:
        return any(kf.features is not None for kf in self.keyframes)

    def build_feature_db(self, orb_nfeatures: int = 1200) -> int:
        """Extract ORB features per reference keyframe and backproject to 3D.

        Depth comes from the saved sidecar when present, else is rendered by
        projecting the reference cloud into that camera. Returns the number of
        keyframes that ended up with usable features. Idempotent / lazy: skips
        keyframes already built.
        """
        if not self.keyframes:
            return 0
        import cv2

        orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        built = 0
        for kf in self.keyframes:
            if kf.features is not None:
                built += 1
                continue
            bgr = cv2.imread(str(kf.rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            kps, desc = orb.detectAndCompute(gray, None)
            if desc is None or len(kps) == 0:
                continue
            xy = np.array([kp.pt for kp in kps], dtype=np.float32)
            depth = self._keyframe_depth(kf, bgr.shape[:2])
            if depth is None:
                continue
            pts3d, valid = backproject_pixels_to_world(
                xy, depth, kf.intrinsics, kf.pose_4x4)
            if not valid.any():
                continue
            kf.features = KeyframeFeatures(
                keypoints_xy=xy[valid], descriptors=desc[valid],
                points3d_world=pts3d[valid], valid=valid[valid],
            )
            built += 1
        print(f"[reference] built ORB feature DB for {built}/{len(self.keyframes)} keyframes")
        return built

    def _keyframe_depth(self, kf: ReferenceKeyframe, hw: tuple[int, int]):  # type: ignore[no-untyped-def]
        if kf.depth_path is not None:
            d = _load_depth_png(kf.depth_path)
            if d is not None:
                return d if d.shape == tuple(hw) else _resize_depth(d, hw)
        if len(self.points) == 0:
            return None
        return _render_depth_from_cloud(self.points, kf.pose_4x4, kf.intrinsics, hw)

    def __len__(self) -> int:
        return len(self.points)


def load_reference_keyframes(keyframes_dir: Path) -> list[ReferenceKeyframe]:
    """Parse ``index.jsonl`` written by ``KeyframeRecorder`` into typed records.

    Lenient by design: a malformed or truncated trailing line (Ctrl-C mid-write)
    is skipped, and a missing depth sidecar just leaves ``depth_path=None``.
    """
    keyframes_dir = Path(keyframes_dir)
    index = keyframes_dir / "index.jsonl"
    if not index.exists():
        print(f"[reference] WARNING: no index.jsonl in {keyframes_dir}")
        return []

    out: list[ReferenceKeyframe] = []
    with open(index, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line
            rgb_path = keyframes_dir / rec["rgb"]
            depth_rel = rec.get("depth")
            depth_path = keyframes_dir / depth_rel if depth_rel else None
            if depth_path is not None and not depth_path.exists():
                depth_path = None
            hw = rec.get("image_hw", [0, 0])
            out.append(
                ReferenceKeyframe(
                    idx=int(rec.get("idx", len(out))),
                    pose_4x4=np.asarray(rec["pose"], dtype=np.float64),
                    intrinsics=np.asarray(rec["intrinsics"], dtype=np.float64),
                    image_hw=(int(hw[0]), int(hw[1])),
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    timestamp=float(rec.get("timestamp", 0.0)),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Depth I/O + backprojection (shared by the reference build and the live layer)
# ---------------------------------------------------------------------------


def _load_depth_png(path: Path) -> NDArray[np.float32] | None:
    """Load a 16-bit-mm depth PNG written by KeyframeRecorder, in metres."""
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return (raw.astype(np.float32) / 1000.0)


def _resize_depth(depth: NDArray[np.float32], hw: tuple[int, int]) -> NDArray[np.float32]:
    import cv2

    H, W = hw
    return cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)


def _render_depth_from_cloud(
    points_world: NDArray[np.floating],
    c2w: NDArray[np.floating],
    K: NDArray[np.floating],
    hw: tuple[int, int],
) -> NDArray[np.float32]:
    """Z-buffer the reference cloud into a keyframe camera (fallback depth).

    Sparse (one cloud point per pixel at most) but enough to give most ORB
    keypoints — which sit on textured surfaces — a 3D anchor via window sampling.
    """
    H, W = hw
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    pts = np.asarray(points_world, dtype=np.float64)
    cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    z = cam[:, 2]
    front = z > 0.05
    cam, z = cam[front], z[front]
    if len(z) == 0:
        return np.zeros((H, W), dtype=np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.round(fx * cam[:, 0] / cam[:, 2] + cx).astype(np.int64)
    v = np.round(fy * cam[:, 1] / cam[:, 2] + cy).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inb], v[inb], z[inb]
    depth = np.zeros((H, W), dtype=np.float32)
    # Write nearest (min z) per pixel: sort far->near so the nearest lands last.
    order = np.argsort(-z)
    depth[v[order], u[order]] = z[order].astype(np.float32)
    return depth


def backproject_pixels_to_world(
    xy: NDArray[np.floating],
    depth: NDArray[np.floating],
    K: NDArray[np.floating],
    c2w: NDArray[np.floating],
    win: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Backproject pixels ``xy`` through ``depth`` + ``K`` + ``c2w`` to world points.

    Samples the median positive depth in a small window so sparse (rendered)
    depth still anchors keypoints. Returns ``(points3d [M,3], valid [M])``.
    """
    K = np.asarray(K, dtype=np.float64)
    c2w = np.asarray(c2w, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape[:2]
    out = np.zeros((len(xy), 3), dtype=np.float64)
    valid = np.zeros(len(xy), dtype=bool)
    for i, (u, v) in enumerate(xy):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        y0, y1 = max(0, vi - win), min(H, vi + win + 1)
        x0, x1 = max(0, ui - win), min(W, ui + win + 1)
        patch = depth[y0:y1, x0:x1]
        pos = patch[patch > 0]
        if pos.size == 0:
            continue
        d = float(np.median(pos))
        cam = np.array([(u - cx) * d / fx, (v - cy) * d / fy, d], dtype=np.float64)
        out[i] = c2w[:3, :3] @ cam + c2w[:3, 3]
        valid[i] = True
    return out, valid
