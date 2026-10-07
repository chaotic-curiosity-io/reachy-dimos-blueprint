"""Bundle v2 provenance + migration.

The multi-device map needs to know, per voxel, which contribution source
observed it and when — ``source_bits`` / ``last_seen`` on the voxel state — and
the bundle needs lineage fields (map_id / revision / sources / frame). These
tests pin the round-trip through ``VoxelMap`` insert/save/load, the v1->v2
in-memory migration, and the forward-version guard, so a schema change that
silently drops provenance can't slip through.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pytest

from xr_nav.map_io import (
    MAP_BUNDLE_VERSION,
    NullObjectDB,
    load_map_bundle,
    save_map_bundle,
)
from xr_nav.voxel_map import VoxelMap


def _cloud(n: int = 800, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, 3).astype(np.float32)


def test_insert_stamps_source_bit_and_last_seen() -> None:
    vm = VoxelMap(voxel_size=0.05)
    vm.insert(_cloud(), source_idx=3, timestamp=1234.0)
    pts, bits, seen = vm.to_points_with_provenance()
    assert len(pts) == vm.size > 0
    assert (bits == (1 << 3)).all()
    assert (seen == 1234.0).all()


def test_second_source_ors_bits_and_advances_last_seen() -> None:
    vm = VoxelMap(voxel_size=0.05)
    pts = _cloud()
    vm.insert(pts, source_idx=0, timestamp=100.0)
    vm.insert(pts, source_idx=4, timestamp=250.0)  # same cells, newer, other device
    _, bits, seen = vm.to_points_with_provenance()
    assert (bits == ((1 << 0) | (1 << 4))).all()
    assert (seen == 250.0).all()          # last_seen takes the max
    # An older observation must not roll last_seen backwards.
    vm.insert(pts, source_idx=0, timestamp=50.0)
    _, _, seen2 = vm.to_points_with_provenance()
    assert (seen2 == 250.0).all()


def test_default_insert_is_legacy_source_zero() -> None:
    vm = VoxelMap(voxel_size=0.05)
    vm.insert(_cloud())  # no provenance args
    _, bits, _ = vm.to_points_with_provenance()
    assert (bits == 1).all()  # bit 0 == legacy single source


def test_state_round_trip_preserves_provenance() -> None:
    vm = VoxelMap(voxel_size=0.05)
    vm.insert(_cloud(), source_idx=2, timestamp=999.0)
    state = vm.to_state()
    assert state["source_bits"].dtype == np.uint16
    assert "last_seen" in state
    vm2 = VoxelMap()
    vm2.load_state(state)
    _, bits, seen = vm2.to_points_with_provenance()
    assert (bits == (1 << 2)).all() and (seen == 999.0).all()


def test_load_state_without_provenance_defaults_to_legacy() -> None:
    vm = VoxelMap(voxel_size=0.05)
    vm.insert(_cloud())
    state = vm.to_state()
    # Simulate a pre-provenance state dict (older producer).
    del state["source_bits"]
    del state["last_seen"]
    vm2 = VoxelMap()
    vm2.load_state(state)  # must not raise
    _, bits, seen = vm2.to_points_with_provenance()
    assert (bits == 1).all() and (seen == 0.0).all()


def test_save_load_bundle_carries_v2_meta(tmp_path: Path) -> None:
    vm = VoxelMap(voxel_size=0.05)
    vm.insert(_cloud(), source_idx=0, timestamp=time.time())
    path = tmp_path / "home_v7.pkl"
    save_map_bundle(
        path, vm, NullObjectDB(), extra={"frames": 42},
        meta={
            "map_id": "home", "revision": 7, "parent_revision": 6,
            "sources": [{"idx": 0, "device": "robot", "depth_source": "da3metric"}],
        },
    )
    b = load_map_bundle(path)
    assert b["version"] == MAP_BUNDLE_VERSION == 2
    assert b["map_id"] == "home" and b["revision"] == 7 and b["parent_revision"] == 6
    assert b["sources"][0]["device"] == "robot"
    assert b["frame"]["convention"] == "opencv-optical"  # default frame filled in


def test_bundle_without_meta_is_valid_v2_single_source(tmp_path: Path) -> None:
    vm = VoxelMap(voxel_size=0.05)
    vm.insert(_cloud())
    path = tmp_path / "plain.pkl"
    save_map_bundle(path, vm, NullObjectDB())  # no meta — legacy producer shape
    b = load_map_bundle(path)
    assert b["version"] == 2
    assert len(b["sources"]) == 1 and b["sources"][0]["idx"] == 0


def test_v1_bundle_migrates_in_memory(tmp_path: Path) -> None:
    vm = VoxelMap(voxel_size=0.05)
    vm.insert(_cloud())
    v1 = {
        "version": 1,
        "saved_at": time.time(),
        "voxel_map": vm.to_state(),
        "object_db": NullObjectDB().to_state(),
        "extra": {},
    }
    path = tmp_path / "old_v1.pkl"
    path.write_bytes(pickle.dumps(v1))
    b = load_map_bundle(path)
    assert b["version"] == 2
    assert b["sources"][0]["device"] == "legacy"
    assert b["frame"]["up_axis"] == "-y"
    # Voxels still load (provenance defaulted to legacy).
    vm2 = VoxelMap()
    vm2.load_state(b["voxel_map"])
    assert vm2.size == vm.size


def test_future_version_is_rejected(tmp_path: Path) -> None:
    future = {
        "version": MAP_BUNDLE_VERSION + 1,
        "saved_at": time.time(),
        "voxel_map": VoxelMap().to_state(),
        "object_db": NullObjectDB().to_state(),
        "extra": {},
    }
    path = tmp_path / "future.pkl"
    path.write_bytes(pickle.dumps(future))
    with pytest.raises(SystemExit):
        load_map_bundle(path)
