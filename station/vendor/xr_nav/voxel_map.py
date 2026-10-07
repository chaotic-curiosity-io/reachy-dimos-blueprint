"""Hash-map voxel grid with raycast-based free space clearing.

Ported from dimos/hardware/sensors/lidar/fastlio2/cpp/voxel_map.hpp.
Pure Python + numba for the inner raycast loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray


@dataclass
class RaycastConfig:
    """Config for raycast-based free space clearing."""

    subsample: int = 4         # raycast every Nth point
    max_misses: int = 5        # erase after N consecutive misses (0 = disabled)
    fov_up_rad: float = 0.35   # ~20° above horizontal (fixed for Viture)
    fov_down_rad: float = -0.52  # ~-30° below horizontal
    # World-frame up axis used for the vertical-FOV erase gate:
    #   "y"    ARKit world (Y-up) — Viture / phone AR poses
    #   "z"    FLU world (Z-up) — Reachy base / robot body frames
    #   "-y"   camera-optical world (Y-down) — VO anchored at the first camera
    #   "none" no FOV gate; erase on miss count alone
    up_axis: str = "y"


# Voxel-key packing: 21 bits per axis, offset so negative indices stay positive.
_KEY_OFF = 1 << 20
_KEY_MASK = (1 << 21) - 1


@dataclass
class _Voxel:
    """Internal voxel storage.

    ``source_bits`` is a bitmask (bit i set = voxel observed by contribution
    source i) and ``last_seen`` the unix timestamp of the most recent
    supporting observation. Both are multi-device-map provenance: which device
    contributed this geometry (source i indexes the bundle's ``sources`` list)
    and how stale it is (for change-detection carving). Defaults — bit 0 set,
    ``last_seen=0`` — reproduce a single legacy source when callers don't pass
    provenance, so pre-provenance producers keep working unchanged.
    """

    x: float
    y: float
    z: float
    confidence: float
    count: int
    miss_count: int
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    source_bits: int = 1
    last_seen: float = 0.0


class VoxelMap:
    """Hash-map-based 3D voxel grid.

    Supports O(1) insert/update, confidence-weighted centroid averaging,
    distance-based pruning, and 3D DDA raycasting for free space clearing.
    """

    def __init__(self, voxel_size: float = 0.10, max_range: float = 6.0) -> None:
        self.voxel_size = voxel_size
        self.max_range = max_range
        self._inv = 1.0 / voxel_size
        self._map: dict[tuple[int, int, int], _Voxel] = {}

    def insert(
        self,
        points: NDArray[np.floating],
        confidences: NDArray[np.floating] | None = None,
        max_drift: float = 0.0,
        colors: NDArray[np.floating] | None = None,
        source_idx: int = 0,
        timestamp: float = 0.0,
    ) -> tuple[int, int]:
        """Insert points into the map, merging into existing voxels.

        Running weighted-average update for centroid position and color.
        Resets miss_count for hit voxels.

        When ``max_drift > 0``, the batch centroid of points mapping to a
        *new* voxel cell is checked against the 26-neighbour cube of
        **pre-existing** voxels. If a pre-existing neighbour centroid is
        within ``max_drift`` metres the whole group is considered a drift
        duplicate and skipped.

        Args:
            colors: Optional [N, 3] float32 RGB in [0, 1].
            source_idx: Contribution-source index (0-15) whose provenance bit is
                OR-ed into every voxel this batch touches. Multi-device maps pass
                the merging device's source index; the default 0 keeps
                single-source producers on bit 0.
            timestamp: Unix time of these observations; every touched voxel's
                ``last_seen`` advances to at least this (staleness for carving).

        Returns:
            (inserted, skipped) counts.
        """
        if len(points) == 0:
            return 0, 0
        src_bit = (1 << int(source_idx)) & 0xFFFF or 1

        pts = np.asarray(points, dtype=np.float32)
        if confidences is not None:
            conf = np.asarray(confidences, dtype=np.float64)
        else:
            conf = np.ones(len(pts), dtype=np.float64)

        has_color = colors is not None
        if has_color:
            cols = np.asarray(colors, dtype=np.float32)

        vmap = self._map
        check_drift = max_drift > 0.0
        max_drift_sq = max_drift * max_drift
        prior_keys: set[tuple[int, int, int]] = set(vmap.keys()) if check_drift else set()

        # Aggregate the whole batch per voxel key in numpy, then merge each
        # touched voxel into the map once. Equivalent to the old per-point
        # incremental weighted mean (weighted means are associative), but the
        # Python-level loop shrinks from N points to the (much smaller) number
        # of unique voxels hit this frame. Keys are packed 21 bits/axis into a
        # single int64 (grid range ±2^20 cells) so np.unique runs on a flat
        # array instead of row-comparing an (N, 3) one.
        keys_int = np.floor(pts.astype(np.float64) * self._inv).astype(np.int64)
        # Clamp to the packable range (±2^20 cells) so degenerate points can't
        # alias onto other keys; anything that far out is garbage anyway.
        np.clip(keys_int, -_KEY_OFF, _KEY_OFF - 1, out=keys_int)
        packed = (((keys_int[:, 0] + _KEY_OFF) << 42)
                  | ((keys_int[:, 1] + _KEY_OFF) << 21)
                  | (keys_int[:, 2] + _KEY_OFF))
        uniq_packed, inverse = np.unique(packed, return_inverse=True)
        uniq = np.empty((len(uniq_packed), 3), dtype=np.int64)
        uniq[:, 0] = (uniq_packed >> 42) - _KEY_OFF
        uniq[:, 1] = ((uniq_packed >> 21) & _KEY_MASK) - _KEY_OFF
        uniq[:, 2] = (uniq_packed & _KEY_MASK) - _KEY_OFF
        nuniq = len(uniq)
        w_sum = np.bincount(inverse, weights=conf, minlength=nuniq)
        pts64 = pts.astype(np.float64)
        wpts = pts64 * conf[:, None]
        wpos = np.empty((nuniq, 3), dtype=np.float64)
        pos_sum = np.empty((nuniq, 3), dtype=np.float64)
        for ax in range(3):
            wpos[:, ax] = np.bincount(inverse, weights=wpts[:, ax], minlength=nuniq)
            # Unweighted sums as a fallback when a group's total confidence is 0.
            pos_sum[:, ax] = np.bincount(inverse, weights=pts64[:, ax], minlength=nuniq)
        if has_color:
            wcols = cols.astype(np.float64) * conf[:, None]
            wcol = np.empty((nuniq, 3), dtype=np.float64)
            for ax in range(3):
                wcol[:, ax] = np.bincount(inverse, weights=wcols[:, ax], minlength=nuniq)
        counts = np.bincount(inverse, minlength=nuniq)

        inserted = 0
        skipped = 0
        for i in range(nuniq):
            key = (int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2]))
            cw = float(w_sum[i])
            cnt = int(counts[i])
            if cw > 0.0:
                gx, gy, gz = wpos[i] / cw
                if has_color:
                    gr, gg, gb = wcol[i] / cw
            else:
                gx, gy, gz = pos_sum[i] / cnt
                gr = gg = gb = 0.0
            if not has_color:
                gr = gg = gb = 0.0

            v = vmap.get(key)
            if v is not None:
                n = v.confidence
                n1 = n + cw
                if n1 > 0.0:
                    v.x = (v.x * n + gx * cw) / n1
                    v.y = (v.y * n + gy * cw) / n1
                    v.z = (v.z * n + gz * cw) / n1
                    if has_color:
                        v.r = (v.r * n + gr * cw) / n1
                        v.g = (v.g * n + gg * cw) / n1
                        v.b = (v.b * n + gb * cw) / n1
                v.confidence = n1
                v.count += cnt
                v.miss_count = 0
                v.source_bits |= src_bit
                if timestamp > v.last_seen:
                    v.last_seen = timestamp
                inserted += cnt
            elif check_drift and self._near_prior(
                gx, gy, gz, key[0], key[1], key[2], max_drift_sq, prior_keys
            ):
                skipped += cnt
            else:
                vmap[key] = _Voxel(x=gx, y=gy, z=gz, confidence=cw,
                                   count=cnt, miss_count=0, r=gr, g=gg, b=gb,
                                   source_bits=src_bit, last_seen=timestamp)
                inserted += cnt

        return inserted, skipped

    def _near_prior(
        self,
        px: float, py: float, pz: float,
        kx: int, ky: int, kz: int,
        max_drift_sq: float,
        prior_keys: set[tuple[int, int, int]],
    ) -> bool:
        """Check the 26-cell neighbourhood for a pre-existing voxel within distance."""
        vmap = self._map
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nk = (kx + dx, ky + dy, kz + dz)
                    if nk in prior_keys:
                        nv = vmap[nk]
                        ex = px - nv.x
                        ey = py - nv.y
                        ez = pz - nv.z
                        if ex * ex + ey * ey + ez * ez < max_drift_sq:
                            return True
        return False

    def raycast_clear(
        self,
        origin: NDArray[np.floating] | tuple[float, float, float],
        points: NDArray[np.floating],
        config: RaycastConfig | None = None,
    ) -> None:
        """Cast rays from origin through sampled points, marking intermediate voxels.

        Voxels that accumulate enough misses and fall within the sensor FOV
        are erased.
        """
        cfg = config or RaycastConfig()
        if len(points) == 0 or cfg.max_misses <= 0:
            return

        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        pts = np.asarray(points, dtype=np.float32)

        # Phase 1: walk rays, increment miss_count for intermediate voxels
        # Collect keys of voxels that exist, pass to numba for the DDA walk
        # Since numba can't access a Python dict, we do it in Python but
        # use numba for the heavy DDA math to compute which keys to mark.
        vmap = self._map
        inv = self._inv
        vs = self.voxel_size

        for i in range(0, len(pts), cfg.subsample):
            px, py, pz = pts[i]
            hit_keys, n_keys = _raycast_single_keys(ox, oy, oz, px, py, pz, inv, vs)
            for j in range(n_keys):
                k = (int(hit_keys[j, 0]), int(hit_keys[j, 1]), int(hit_keys[j, 2]))
                v = vmap.get(k)
                if v is not None and v.miss_count < 255:
                    v.miss_count += 1

        # Phase 2: erase voxels that exceeded miss threshold within the
        # sensor's vertical FOV. The elevation is measured about the world
        # up axis configured in ``cfg.up_axis`` ("y" = ARKit, "z" = FLU,
        # "-y" = camera-optical world, "none" = skip the FOV gate).
        fov_up = cfg.fov_up_rad
        fov_down = cfg.fov_down_rad
        up = cfg.up_axis
        to_delete = []
        for key, v in vmap.items():
            if v.miss_count > cfg.max_misses:
                if up == "none":
                    to_delete.append(key)
                    continue
                dx = v.x - ox
                dy = v.y - oy
                dz = v.z - oz
                if up == "z":
                    vert = dz
                    horiz = math.sqrt(dx * dx + dy * dy)
                elif up == "-y":
                    vert = -dy
                    horiz = math.sqrt(dx * dx + dz * dz)
                else:  # "y" (ARKit world: Y-up, X-Z floor plane)
                    vert = dy
                    horiz = math.sqrt(dx * dx + dz * dz)
                if horiz < 1e-6:
                    to_delete.append(key)
                    continue
                elev = math.atan2(vert, horiz)
                if fov_down <= elev <= fov_up:
                    to_delete.append(key)

        for key in to_delete:
            del vmap[key]

    def prune(self, px: float, py: float, pz: float) -> None:
        """Remove voxels farther than max_range from the given position."""
        r2 = self.max_range * self.max_range
        to_delete = [
            key for key, v in self._map.items()
            if (v.x - px) ** 2 + (v.y - py) ** 2 + (v.z - pz) ** 2 > r2
        ]
        for key in to_delete:
            del self._map[key]

    def to_points(self, min_observations: int = 1) -> NDArray[np.float32]:
        """Export voxel centroids as an (N, 3) array.

        Args:
            min_observations: Only include voxels observed at least this many times.
        """
        voxels = [
            (v.x, v.y, v.z) for v in self._map.values()
            if v.count >= min_observations
        ]
        if not voxels:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(voxels, dtype=np.float32)

    def to_points_colored(self, min_observations: int = 1) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Export voxel centroids and their averaged RGB colors.

        Returns:
            (points [N, 3], colors [N, 3]) with colors in [0, 1].
        """
        pts = []
        cols = []
        for v in self._map.values():
            if v.count >= min_observations:
                pts.append((v.x, v.y, v.z))
                cols.append((v.r, v.g, v.b))
        if not pts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)

    def to_points_with_confidence(self, min_observations: int = 1) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Export voxel centroids and confidences."""
        pts = []
        confs = []
        for v in self._map.values():
            if v.count >= min_observations:
                pts.append((v.x, v.y, v.z))
                confs.append(v.confidence)
        if not pts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.array(pts, dtype=np.float32), np.array(confs, dtype=np.float32)

    def to_points_with_provenance(
        self, min_observations: int = 1
    ) -> tuple[NDArray[np.float32], NDArray[np.uint16], NDArray[np.float32]]:
        """Export centroids with their provenance channels.

        Returns ``(points [N,3], source_bits [N], last_seen [N])`` — the inputs
        a session merge needs to pick scale anchors (voxels carrying a trusted
        source's bit) and to age geometry for change-detection carving.
        """
        pts, bits, seen = [], [], []
        for v in self._map.values():
            if v.count >= min_observations:
                pts.append((v.x, v.y, v.z))
                bits.append(v.source_bits)
                seen.append(v.last_seen)
        if not pts:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.uint16),
                    np.empty((0,), dtype=np.float32))
        return (np.array(pts, dtype=np.float32),
                np.array(bits, dtype=np.uint16),
                np.array(seen, dtype=np.float32))

    @property
    def size(self) -> int:
        return len(self._map)

    def to_state(self) -> dict:
        """Serialize the voxel map to a numpy-friendly dict for save/load.

        Output has fixed-shape arrays so callers can hand the dict straight to
        ``np.savez_compressed`` if they want a portable on-disk representation
        instead of pickling.
        """
        if not self._map:
            return {
                "voxel_size": float(self.voxel_size),
                "max_range": float(self.max_range),
                "keys": np.zeros((0, 3), dtype=np.int64),
                "centroids": np.zeros((0, 3), dtype=np.float32),
                "colors": np.zeros((0, 3), dtype=np.float32),
                "confidence": np.zeros((0,), dtype=np.float32),
                "count": np.zeros((0,), dtype=np.int32),
                "miss_count": np.zeros((0,), dtype=np.int32),
                "source_bits": np.zeros((0,), dtype=np.uint16),
                "last_seen": np.zeros((0,), dtype=np.float32),
            }
        items = list(self._map.items())
        keys = np.array([k for k, _ in items], dtype=np.int64)
        vox = [v for _, v in items]
        centroids = np.array([(v.x, v.y, v.z) for v in vox], dtype=np.float32)
        colors = np.array([(v.r, v.g, v.b) for v in vox], dtype=np.float32)
        confidence = np.array([v.confidence for v in vox], dtype=np.float32)
        count = np.array([v.count for v in vox], dtype=np.int32)
        miss_count = np.array([v.miss_count for v in vox], dtype=np.int32)
        source_bits = np.array([v.source_bits for v in vox], dtype=np.uint16)
        last_seen = np.array([v.last_seen for v in vox], dtype=np.float32)
        return {
            "voxel_size": float(self.voxel_size),
            "max_range": float(self.max_range),
            "keys": keys,
            "centroids": centroids,
            "colors": colors,
            "confidence": confidence,
            "count": count,
            "miss_count": miss_count,
            "source_bits": source_bits,
            "last_seen": last_seen,
        }

    def load_state(self, state: dict, replace: bool = True) -> None:
        """Repopulate from a dict produced by ``to_state``.

        When ``replace`` is True the existing voxels are cleared first; otherwise
        loaded voxels overwrite same-key cells and other cells are preserved.
        Voxel size / max_range from the saved state win — callers that mix
        sessions with different settings should reconstruct the map manually.
        """
        if replace:
            self._map.clear()
        self.voxel_size = float(state["voxel_size"])
        self.max_range = float(state["max_range"])
        self._inv = 1.0 / self.voxel_size
        keys = state["keys"]
        centroids = state["centroids"]
        colors = state["colors"]
        confidence = state["confidence"]
        count = state["count"]
        miss_count = state["miss_count"]
        n = len(keys)
        # Provenance channels are optional so pre-provenance bundles still load:
        # a voxel without a recorded source is treated as legacy source 0.
        source_bits = state.get("source_bits")
        if source_bits is None:
            source_bits = np.ones(n, dtype=np.uint16)
        last_seen = state.get("last_seen")
        if last_seen is None:
            last_seen = np.zeros(n, dtype=np.float32)
        for i in range(n):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            self._map[k] = _Voxel(
                x=float(centroids[i, 0]),
                y=float(centroids[i, 1]),
                z=float(centroids[i, 2]),
                confidence=float(confidence[i]),
                count=int(count[i]),
                miss_count=int(miss_count[i]),
                r=float(colors[i, 0]),
                g=float(colors[i, 1]),
                b=float(colors[i, 2]),
                source_bits=int(source_bits[i]),
                last_seen=float(last_seen[i]),
            )

    def clear(self) -> None:
        self._map.clear()


# ---------------------------------------------------------------------------
# Numba-accelerated 3D DDA raycast (returns array of voxel keys to mark)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _raycast_single_keys(
    ox: float, oy: float, oz: float,
    px: float, py: float, pz: float,
    inv: float, voxel_size: float,
) -> tuple[NDArray[np.int64], int]:
    """Amanatides & Woo 3D DDA: intermediate voxel keys along the ray.

    Walks from (ox,oy,oz) to (px,py,pz). Returns ``(keys, n)`` where
    ``keys[:n]`` are the voxel coordinates traversed *excluding* the endpoint
    voxel (which was hit).
    """
    empty = np.empty((0, 3), dtype=np.int64)
    dx = px - ox
    dy = py - oy
    dz = pz - oz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-6:
        return empty, 0
    dx /= length
    dy /= length
    dz /= length

    cx = int(math.floor(ox * inv))
    cy = int(math.floor(oy * inv))
    cz = int(math.floor(oz * inv))
    ex = int(math.floor(px * inv))
    ey = int(math.floor(py * inv))
    ez = int(math.floor(pz * inv))

    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    sz = 1 if dz >= 0 else -1

    # tMax: parametric distance to next voxel boundary per axis
    if abs(dx) < 1e-10:
        tMaxX = 1e30
    else:
        boundary_x = (cx + 1 if dx > 0 else cx) * voxel_size
        tMaxX = (boundary_x - ox) / dx

    if abs(dy) < 1e-10:
        tMaxY = 1e30
    else:
        boundary_y = (cy + 1 if dy > 0 else cy) * voxel_size
        tMaxY = (boundary_y - oy) / dy

    if abs(dz) < 1e-10:
        tMaxZ = 1e30
    else:
        boundary_z = (cz + 1 if dz > 0 else cz) * voxel_size
        tMaxZ = (boundary_z - oz) / dz

    tDeltaX = 1e30 if abs(dx) < 1e-10 else abs(voxel_size / dx)
    tDeltaY = 1e30 if abs(dy) < 1e-10 else abs(voxel_size / dy)
    tDeltaZ = 1e30 if abs(dz) < 1e-10 else abs(voxel_size / dz)

    max_steps = int(length * inv) + 3
    keys = np.empty((max_steps, 3), dtype=np.int64)
    n = 0
    for _ in range(max_steps):
        if cx == ex and cy == ey and cz == ez:
            break

        keys[n, 0] = cx
        keys[n, 1] = cy
        keys[n, 2] = cz
        n += 1

        if tMaxX < tMaxY and tMaxX < tMaxZ:
            cx += sx
            tMaxX += tDeltaX
        elif tMaxY < tMaxZ:
            cy += sy
            tMaxY += tDeltaY
        else:
            cz += sz
            tMaxZ += tDeltaZ

    return keys, n
