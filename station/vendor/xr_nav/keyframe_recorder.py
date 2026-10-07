"""Append-only writer for the RGB frames that fed the depth model.

Persists the source RGB + camera pose + intrinsics for every accepted keyframe
so DA3 (or any future depth model) can be re-run offline, bad frames can be
audited, and the voxel map can be reconstructed from raw inputs.

Disk layout::

    <out_dir>/
        rgb/kf_000042.png        # zero-padded frame index, lossless PNG by default
        index.jsonl              # one JSON object per accepted frame

JSONL was chosen over CSV because pose is 4x4 and intrinsics is 3x3; one-line
append-per-frame is crash-safe (a Ctrl-C mid-run leaves a valid prefix), and
the line-buffered file handle survives without re-rewriting the whole index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

import cv2
import numpy as np
from numpy.typing import NDArray


class KeyframeRecorder:
    def __init__(
        self,
        out_dir: Path,
        rgb_format: str = "png",
        save_depth: bool = False,
    ) -> None:
        if rgb_format not in ("png", "jpg"):
            raise ValueError(f"rgb_format must be 'png' or 'jpg', got {rgb_format!r}")
        self.out_dir = Path(out_dir)
        self.rgb_dir = self.out_dir / "rgb"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_format = rgb_format
        # Depth enables the visual relocalization layer to backproject reference
        # keypoints to exact 3D. Stored as 16-bit PNG in millimetres (lossless,
        # ~1/4 the size of float32 .npy and viewable in any image tool).
        self.save_depth = save_depth
        self.depth_dir = self.out_dir / "depth"
        if save_depth:
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        # Line-buffered so each record() flushes to disk even if the process dies.
        self._index: IO[str] = open(
            self.out_dir / "index.jsonl", "a", buffering=1, encoding="utf-8"
        )
        self._count = 0

    def record(
        self,
        *,
        idx: int,
        rgb: NDArray[np.uint8],
        pose_4x4: NDArray[np.floating],
        intrinsics: NDArray[np.floating],
        timestamp: float,
        depth: NDArray[np.floating] | None = None,
        reason: str = "",
    ) -> None:
        """Write one keyframe. Idempotent file naming by ``idx``."""
        if rgb.ndim == 2:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif rgb.ndim == 3 and rgb.shape[2] == 3:
            # Caller passes RGB; cv2.imwrite expects BGR.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"rgb must be HxW or HxWx3 uint8, got shape {rgb.shape}")

        fname = f"kf_{idx:06d}.{self.rgb_format}"
        rgb_path = self.rgb_dir / fname
        if self.rgb_format == "jpg":
            cv2.imwrite(str(rgb_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            cv2.imwrite(str(rgb_path), bgr)

        h, w = rgb.shape[:2]
        record = {
            "idx": int(idx),
            "timestamp": float(timestamp),
            "rgb": f"rgb/{fname}",
            "pose": np.asarray(pose_4x4, dtype=np.float64).tolist(),
            "intrinsics": np.asarray(intrinsics, dtype=np.float64).tolist(),
            "image_hw": [int(h), int(w)],
            "reason": reason,
        }
        if self.save_depth and depth is not None:
            dname = f"kf_{idx:06d}.png"
            depth_mm = np.clip(np.nan_to_num(np.asarray(depth), nan=0.0,
                                             posinf=0.0, neginf=0.0) * 1000.0,
                               0, 65535).astype(np.uint16)
            cv2.imwrite(str(self.depth_dir / dname), depth_mm)
            record["depth"] = f"depth/{dname}"
        self._index.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if not self._index.closed:
            self._index.close()
            print(f"[keyframes] wrote {self._count} frames to {self.out_dir}")

    def __enter__(self) -> "KeyframeRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
