"""Shared argparse registration for map I/O + keyframe capture.

Every spatial-map producer (iPhone, Viture, xr-nav scripts, Reachy scanner)
calls these to register the same flag set with identical semantics. Adding a
flag here automatically propagates to all consumers.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_map_io_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("map I/O")
    g.add_argument(
        "--save-map",
        type=Path,
        default=None,
        help="On exit, save the voxel map + tracked objects to this .pkl bundle. "
        "Pair with --load-map to resume a previous capture",
    )
    g.add_argument(
        "--load-map",
        type=Path,
        default=None,
        help="Load a previously-saved .pkl map bundle before starting. "
        "Note: the loaded map is in the prior session's world frame, so "
        "it only aligns when the new VO session starts at the same physical "
        "pose (e.g., re-running the same clip from the start)",
    )
    g.add_argument(
        "--save-map-every-n",
        type=int,
        default=0,
        help="If >0 and --save-map is set, also write the bundle every N frames "
        "in addition to on-exit. Useful for crash safety on long sessions",
    )
    g.add_argument(
        "--save-ply",
        type=Path,
        default=None,
        help="Standalone PLY export of voxel centroids (no .pkl needed). "
        "Viewable in MeshLab/CloudCompare/Blender",
    )
    g.add_argument(
        "--save-pcd",
        type=Path,
        default=None,
        help="Standalone PCD export (Open3D-native, slightly faster reload)",
    )
    g.add_argument(
        "--save-cloud-with-map",
        action="store_true",
        help="When --save-map is set, also emit <save-map>.ply and <save-map>.pcd "
        "next to the pkl on every write",
    )
    g.add_argument(
        "--cloud-min-observations",
        type=int,
        default=None,
        help="Min voxel observation count for PLY/PCD export. If unset, falls back "
        "to --voxel-min-observations (when available) or 1",
    )
    g.add_argument(
        "--session-meta",
        type=str,
        default=None,
        help="JSON describing this capture as a contribution session (device, "
        "depth source, pose frame, reloc seed). Stamped into the v2 map bundle's "
        "provenance and written as session.json next to --save-keyframes so a "
        "multi-device merge pipeline can align + fuse this scan later",
    )


def add_keyframe_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("keyframe capture")
    g.add_argument(
        "--save-keyframes",
        type=Path,
        default=None,
        help="Directory to write keyframe RGB + pose + intrinsics. Created if missing",
    )
    g.add_argument(
        "--save-keyframes-every-n",
        type=int,
        default=1,
        help="Subsample stride for scripts that don't use a KeyframeSelector "
        "(record every Nth processed frame). The xr-nav pipeline records every "
        "selector-accepted keyframe regardless of this value",
    )
    g.add_argument(
        "--keyframe-rgb-format",
        choices=["png", "jpg"],
        default="png",
        help="png = lossless (default); jpg = ~3x smaller, lossy",
    )
    g.add_argument(
        "--keyframe-save-depth",
        action="store_true",
        help="Also save each keyframe's depth map (16-bit PNG, mm). Lets the "
        "visual relocalization layer backproject reference keypoints to exact 3D",
    )


def add_reloc_args(parser: argparse.ArgumentParser) -> None:
    """Relocalization against a previously-saved reference map.

    Unlike ``--load-map`` (which merges saved voxels into the live session's own
    world frame and only aligns when the robot restarts at the exact same physical
    pose), ``--reference-map`` keeps the saved map as a *fixed* registration target
    and estimates the rigid transform that places the new (moved) session into it.
    """
    g = parser.add_argument_group("relocalization")
    g.add_argument(
        "--reference-map",
        type=Path,
        default=None,
        help="Path to a saved .pkl map bundle to relocalize against. Loaded as a "
        "fixed reference cloud (shown statically), separate from the live map",
    )
    g.add_argument(
        "--reference-keyframes",
        type=Path,
        default=None,
        help="Directory of saved keyframes (rgb + index.jsonl) for the reference "
        "map, used for the visual (2D feature) relocalization layer",
    )
    g.add_argument(
        "--relocalize",
        action="store_true",
        help="Enable on-demand relocalization. When an external trigger fires "
        "(RELOC_REQUEST), capture a short local scan and align it to the reference",
    )
    g.add_argument(
        "--reloc-fitness-threshold",
        type=float,
        default=0.45,
        help="Minimum registration fitness to accept a relocalization (else reject "
        "and keep the prior alignment). Matches the dimos RelocalizationModule default",
    )
    g.add_argument(
        "--reloc-capture-frames",
        type=int,
        default=40,
        help="Number of frames to accumulate into the local scan after a trigger "
        "before running the solve (≈ sweep duration × fps)",
    )
    g.add_argument(
        "--reloc-merge",
        action="store_true",
        help="After a successful relocalization, seed the live map with the "
        "reference voxels so mapping continues on top of the saved map",
    )


def resolve_cloud_min_observations(args: argparse.Namespace) -> int:
    """Pick the right min-obs for PLY/PCD export.

    Precedence: --cloud-min-observations > --voxel-min-observations > 1.
    Scripts that don't expose --voxel-min-observations are unaffected.
    """
    if getattr(args, "cloud_min_observations", None) is not None:
        return int(args.cloud_min_observations)
    return int(getattr(args, "voxel_min_observations", 1))
