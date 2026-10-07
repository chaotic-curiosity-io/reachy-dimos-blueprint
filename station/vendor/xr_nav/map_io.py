"""Shared save/load helpers for spatial maps.

Centralises the save/load logic that previously lived inside individual demo
scripts (mac_iphone_spatial_foxglove.py, mac_viture_spatial_foxglove.py). Every
spatial-map producer in this repo and downstream consumers like
the station bridge import from here, so the .pkl bundle schema and
PLY/PCD export format stay in lockstep.

Public API:
    save_map_bundle(path, voxel_map, object_db, extra=None)
    load_map_bundle(path) -> dict
    save_voxel_map_ply(path, voxel_map, min_observations=1) -> int
    save_voxel_map_pcd(path, voxel_map, min_observations=1) -> int
    MapArtifactWriter — single call point for pkl + sibling PLY/PCD
    NullObjectDB — stand-in when detection is disabled
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

MAP_BUNDLE_VERSION = 2

# Default coordinate frame for a bundle when the producer doesn't declare one:
# the dimos pipeline anchors the world at the first camera pose in OpenCV
# optical convention (X-right, Y-down, Z-forward), so the world "up" axis is -Y
# and the cloud is not gravity-aligned. Multi-device merges may re-anchor.
DEFAULT_MAP_FRAME = {"convention": "opencv-optical", "up_axis": "-y", "gravity_aligned": False}


def _default_v2_fields(state: dict) -> dict:
    """Fill the v2 provenance/lineage fields a bundle may omit.

    Applied on save (producer passed no ``meta``) and on load (migrating a v1
    bundle): a bundle with no declared sources is a single ``legacy`` source, so
    downstream provenance code always has at least one entry to index into.
    """
    return {
        "map_id": state.get("map_id"),
        "revision": state.get("revision"),
        "parent_revision": state.get("parent_revision"),
        "frame": state.get("frame") or dict(DEFAULT_MAP_FRAME),
        "sources": state.get("sources") or [{
            "idx": 0,
            "session_id": None,
            "device": "legacy",
            "depth_source": "legacy",
            "depth_scale_applied": 1.0,
            "reloc": None,
            "n_frames": None,
            "contributed_at": state.get("saved_at"),
        }],
    }


class _VoxelMapLike(Protocol):
    def to_state(self) -> dict: ...
    def to_points_colored(
        self, min_observations: int = 1
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _ObjectDBLike(Protocol):
    def to_state(self) -> dict: ...


class NullObjectDB:
    """Minimal stand-in so bundles still save when detection is disabled."""

    def to_state(self) -> dict:
        return {
            "pending": {},
            "permanent": {},
            "track_id_map": {},
            "confidence": {},
            "config": {},
        }


def save_map_bundle(
    path: Path,
    voxel_map: _VoxelMapLike,
    object_db: _ObjectDBLike,
    extra: dict | None = None,
    *,
    meta: dict | None = None,
) -> None:
    """Pickle a versioned bundle of voxel + object state.

    Pickle is used (not npz) because ObjectDB Object instances carry nested
    PointCloud2 / Vector3 / open3d objects that don't serialize cleanly to
    pure-numpy. Bundles are tied to the dimos Object class layout — re-saving
    after schema changes is the recovery path.

    ``meta`` carries v2 lineage/provenance — ``map_id``, ``revision``,
    ``parent_revision``, ``frame``, ``sources`` — for maps that belong to a
    persistent revision chain (a downstream map store / merge pipeline).
    Omit it and the bundle still saves as valid v2 with a single ``legacy``
    source, so single-scan producers need no changes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": MAP_BUNDLE_VERSION,
        "saved_at": time.time(),
        "voxel_map": voxel_map.to_state(),
        "object_db": object_db.to_state(),
        "extra": extra or {},
    }
    state.update(_default_v2_fields(meta or {}))
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_vox = len(state["voxel_map"]["keys"])
    n_perm = len(state["object_db"]["permanent"])
    n_pend = len(state["object_db"]["pending"])
    print(
        f"[save] wrote {path} — {n_vox} voxels, "
        f"{n_perm} permanent + {n_pend} pending objects"
    )


def load_map_bundle(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"--load-map: bundle not found at {path}")
    with open(path, "rb") as f:
        state = pickle.load(f)
    v = state.get("version")
    if v is not None and v > MAP_BUNDLE_VERSION:
        # Forward-incompatible: a newer schema may add fields this build can't
        # honour. Re-save from the producing build is the recovery path.
        sys.exit(
            f"--load-map: bundle version {v} is newer than supported "
            f"{MAP_BUNDLE_VERSION} — upgrade xr_nav or re-save the map"
        )
    if v != MAP_BUNDLE_VERSION:
        # v1 (and any unversioned legacy) upgrades in memory: synthesize the
        # provenance/lineage fields as one legacy source. The voxel state loads
        # fine because VoxelMap.load_state treats source_bits/last_seen as
        # optional. Next save rewrites it as v2.
        print(f"[load] migrating bundle v{v} -> v{MAP_BUNDLE_VERSION} in memory")
        state.update(_default_v2_fields(state))
        state["version"] = MAP_BUNDLE_VERSION
    n_vox = len(state["voxel_map"]["keys"])
    n_perm = len(state["object_db"]["permanent"])
    n_pend = len(state["object_db"]["pending"])
    age = time.time() - state.get("saved_at", time.time())
    print(
        f"[load] read {path} — {n_vox} voxels, "
        f"{n_perm} permanent + {n_pend} pending objects, "
        f"saved {age/60:.1f}m ago"
    )
    return state


def _voxel_map_to_o3d(voxel_map: _VoxelMapLike, min_observations: int):
    """Build an Open3D PointCloud from voxel centroids + colors. Returns (pcd, n)."""
    import open3d as o3d

    pts, cols = voxel_map.to_points_colored(min_observations=min_observations)
    pcd = o3d.geometry.PointCloud()
    if len(pts) == 0:
        return pcd, 0
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols.astype(np.float64), 0.0, 1.0))
    return pcd, len(pts)


def save_voxel_map_ply(
    path: Path,
    voxel_map: _VoxelMapLike,
    min_observations: int = 1,
) -> int:
    """Write voxel centroids as a binary PLY. Returns point count written.

    Sibling-file convention: foo.pkl <-> foo.ply. Viewable in MeshLab,
    CloudCompare, Blender, Open3D without a Python round-trip.
    """
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd, n = _voxel_map_to_o3d(voxel_map, min_observations)
    if n == 0:
        print(f"[save] PLY skipped (0 voxels >= min_obs={min_observations})")
        return 0
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)
    print(f"[save] wrote {path} — {n} points (PLY)")
    return n


def save_voxel_map_pcd(
    path: Path,
    voxel_map: _VoxelMapLike,
    min_observations: int = 1,
) -> int:
    """Write voxel centroids as a binary PCD. Returns point count written.

    PCD is Open3D's native format — slightly faster reload in Python pipelines
    than PLY. Use PLY for cross-tool portability, PCD for in-pipeline reuse.
    """
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd, n = _voxel_map_to_o3d(voxel_map, min_observations)
    if n == 0:
        print(f"[save] PCD skipped (0 voxels >= min_obs={min_observations})")
        return 0
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)
    print(f"[save] wrote {path} — {n} points (PCD)")
    return n


class MapArtifactWriter:
    """Single emit point for pkl + sibling PLY + sibling PCD.

    Collapses the three duplicated save sites in mac_iphone_spatial_foxglove.py
    (periodic save, on-exit save, error save) into one call. Each artifact is
    independent — any of pkl/ply/pcd can be on or off.

    Sibling-file behaviour: when ``save_cloud_with_map`` is True and
    ``save_map`` is set, ``<save_map>.ply`` and/or ``<save_map>.pcd`` are also
    written next to the pkl on every ``write()`` invocation.
    """

    def __init__(
        self,
        save_map: Path | None = None,
        save_ply: Path | None = None,
        save_pcd: Path | None = None,
        save_cloud_with_map: bool = False,
        cloud_min_observations: int = 1,
        meta: dict | None = None,
    ) -> None:
        self.save_map = Path(save_map) if save_map else None
        self.save_ply = Path(save_ply) if save_ply else None
        self.save_pcd = Path(save_pcd) if save_pcd else None
        self.save_cloud_with_map = save_cloud_with_map
        self.cloud_min_observations = int(cloud_min_observations)
        # v2 lineage/provenance stamped onto every pkl this writer emits
        # (e.g. one contribution source describing the current scan).
        self.meta = meta

    @property
    def enabled(self) -> bool:
        return any(
            (self.save_map, self.save_ply, self.save_pcd, self.save_cloud_with_map)
        )

    def _sibling(self, suffix: str) -> Path | None:
        if not (self.save_cloud_with_map and self.save_map):
            return None
        return self.save_map.with_suffix(suffix)

    def write(
        self,
        voxel_map: _VoxelMapLike,
        object_db: _ObjectDBLike,
        extra: dict | None = None,
    ) -> None:
        """Write whichever artifacts are configured. Errors are caught + logged
        per-artifact so a single failure doesn't lose the others."""
        if self.save_map is not None:
            try:
                save_map_bundle(self.save_map, voxel_map, object_db, extra,
                                meta=self.meta)
            except Exception as e:
                print(f"[save] pkl FAILED: {e}", file=sys.stderr)

        ply_path = self.save_ply or self._sibling(".ply")
        if ply_path is not None:
            try:
                save_voxel_map_ply(ply_path, voxel_map, self.cloud_min_observations)
            except Exception as e:
                print(f"[save] ply FAILED: {e}", file=sys.stderr)

        pcd_path = self.save_pcd or self._sibling(".pcd")
        if pcd_path is not None:
            try:
                save_voxel_map_pcd(pcd_path, voxel_map, self.cloud_min_observations)
            except Exception as e:
                print(f"[save] pcd FAILED: {e}", file=sys.stderr)
