"""On-robot ONNX object detection (COCO-80) through onnxruntime.

onnxruntime is already in the robot's shared apps venv (the face
recognition app pulls it in), so this backend adds a model file rather than
a dependency stack. At 320x320 a YOLO-nano export runs at a handful of Hz
on the robot's CPU, which is enough for a follow loop that only needs to
correct heading a few times a second.

The pre/post-processing here is deliberately generic: both common YOLO ONNX
output layouts are handled, so a ``yolo export model=yolo11n.pt format=onnx
imgsz=320`` (v8/v11 style, ``(1, 4+nc, N)``) and an older v5-style export
(``(1, N, 5+nc)``) both work without a flag. Nothing in this file is
specific to a vendor's runtime wrapper — it is numpy and an ORT session.

Model discovery, in order: the ``track_model_path`` setting, the
``WHEELS_TRACK_MODEL`` env var, then the first ``*.onnx`` under
``~/.config/reachy_wheels_app/models/``. Missing model → a
``DetectorUnavailable`` that tells you the export command to run.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from .detectors import DetectorUnavailable
from .types import Detection
from .vocab import COCO_CLASSES

_log = logging.getLogger(__name__)

MODEL_DIR = Path(os.path.expanduser("~/.config/reachy_wheels_app/models"))

_HOWTO = (
    "no ONNX detection model found. Put one at {dir}/ (any .onnx), or set "
    "track_model_path. To make one: "
    "`pip install ultralytics && yolo export model=yolo11n.pt format=onnx "
    "imgsz=320` — then copy yolo11n.onnx to the robot."
).format(dir=MODEL_DIR)


def resolve_model_path(configured: str = "") -> Path:
    for candidate in (configured, os.environ.get("WHEELS_TRACK_MODEL", "")):
        if candidate:
            path = Path(os.path.expanduser(candidate))
            if path.is_file():
                return path
            raise DetectorUnavailable(f"detection model not found at {path}")
    if MODEL_DIR.is_dir():
        found = sorted(MODEL_DIR.glob("*.onnx"))
        if found:
            return found[0]
    raise DetectorUnavailable(_HOWTO)


def _providers() -> list[str]:
    # CoreML when this runs on a Mac (bench/replay); the robot is CPU.
    if sys.platform == "darwin":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def letterbox(frame, size: int):
    """Resize into a ``size x size`` square, preserving aspect with padding.

    Returns ``(padded_rgb_float_chw, scale, pad_x, pad_y)`` — the last three
    undo the transform when mapping boxes back to frame pixels.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    # BGR (the SDK's frame format) → RGB, NCHW float32 in 0..1.
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold: float, max_out: int = 50):
    """Plain greedy NMS in numpy — no torchvision on the robot."""
    import numpy as np

    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_out:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolo(output, num_classes: int):
    """Normalise either YOLO ONNX layout to ``(boxes_cxcywh, scores, class_ids)``.

    v8/v11 exports are channels-first ``(4+nc, N)``; v5-era exports are
    ``(N, 5+nc)``. We identify the channel axis by its *width* — it has to
    be ``4+nc`` or ``5+nc`` — rather than by assuming N is the larger axis,
    which is true of real exports but not of anything else.
    """
    import numpy as np

    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unexpected detector output shape {np.asarray(output).shape}")

    widths = {4 + num_classes, 5 + num_classes}
    if arr.shape[1] in widths:
        pass                       # already (N, C)
    elif arr.shape[0] in widths:
        arr = arr.T                # channels-first → (N, C)
    elif arr.shape[0] < arr.shape[1]:
        # Neither axis matches our class count (a sidecar .names of the
        # wrong length, say). Fall back to "N is the big axis" and let the
        # score threshold sort out the mess.
        arr = arr.T
    channels = arr.shape[1]

    if channels == 5 + num_classes:      # v5: cx cy w h obj cls...
        boxes = arr[:, :4]
        scores = arr[:, 4:5] * arr[:, 5:]
    else:                                 # v8/v11: cx cy w h cls...
        boxes = arr[:, :4]
        scores = arr[:, 4:]
    if scores.size == 0:
        return boxes[:0], np.zeros(0), np.zeros(0, dtype=int)
    class_ids = scores.argmax(axis=1)
    best = scores[np.arange(scores.shape[0]), class_ids]
    return boxes, best, class_ids


class OnnxDetector:
    """COCO-80 detection on the robot's CPU."""

    name = "onnx"

    def __init__(self, model_path: str = "", *, imgsz: int = 320,
                 min_score: float = 0.35, nms_iou: float = 0.45,
                 class_names: Sequence[str] = COCO_CLASSES):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorUnavailable(
                "onnxruntime is not installed in this venv — "
                "pip install onnxruntime"
            ) from exc

        self.path = resolve_model_path(model_path)
        self.imgsz = max(96, int(imgsz))
        self.min_score = float(min_score)
        self.nms_iou = float(nms_iou)
        self.class_names = tuple(self._load_names(self.path, class_names))

        options = ort.SessionOptions()
        # The follow loop is latency-bound on one frame at a time and shares
        # the robot with voice; a thread stampede here starves the mic pump.
        options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        try:
            self._session = ort.InferenceSession(
                str(self.path), options, providers=_providers())
        except Exception as exc:  # noqa: BLE001
            raise DetectorUnavailable(
                f"could not load {self.path.name}: {exc}") from exc
        spec = self._session.get_inputs()[0]
        self._input = spec.name
        # A YOLO export bakes its input size in (shape [1, 3, S, S]). Trust
        # the file over the setting: a `track_imgsz` that disagrees with the
        # model is an instant runtime failure, and the number the user can
        # see in settings is the easier one to get wrong.
        baked = self._baked_size(spec.shape)
        if baked and baked != self.imgsz:
            _log.info("model %s is fixed at %dpx — using that, not the "
                      "configured %dpx", self.path.name, baked, self.imgsz)
            self.imgsz = baked
        _log.info("onnx detector ready: %s (%dpx, %d classes)",
                  self.path.name, self.imgsz, len(self.class_names))

    @staticmethod
    def _baked_size(shape) -> int | None:
        """The square input size a fixed-shape model demands, if it has one."""
        try:
            h, w = shape[2], shape[3]
        except (IndexError, TypeError):
            return None
        if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
            return h
        return None

    @staticmethod
    def _load_names(path: Path, default: Sequence[str]) -> Sequence[str]:
        """A sidecar ``<model>.names`` (one label per line) overrides COCO."""
        sidecar = path.with_suffix(".names")
        try:
            lines = [ln.strip() for ln in sidecar.read_text().splitlines()]
        except OSError:
            return default
        labels = [ln for ln in lines if ln]
        return labels or default

    @property
    def vocabulary(self) -> Sequence[str]:
        return self.class_names

    def detect(self, frame, targets: Sequence[str] = ()) -> list[Detection]:
        import numpy as np

        if frame is None or getattr(frame, "ndim", 0) != 3:
            return []
        h, w = frame.shape[:2]
        blob, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        outputs = self._session.run(None, {self._input: blob})
        boxes, scores, class_ids = decode_yolo(outputs[0], len(self.class_names))

        wanted = {str(t).strip().lower() for t in targets if str(t).strip()}
        keep_mask = scores >= self.min_score
        if wanted:
            # Filtering before NMS keeps the sort cheap and stops a
            # high-scoring chair from suppressing the person behind it.
            wanted_ids = {i for i, name in enumerate(self.class_names)
                          if name.lower() in wanted}
            if wanted_ids:
                keep_mask &= np.isin(class_ids, list(wanted_ids))
        boxes, scores, class_ids = boxes[keep_mask], scores[keep_mask], class_ids[keep_mask]
        if len(boxes) == 0:
            return []

        # cxcywh (letterboxed pixels) → xyxy (original frame pixels)
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = np.stack([
            (cx - bw / 2 - pad_x) / scale, (cy - bh / 2 - pad_y) / scale,
            (cx + bw / 2 - pad_x) / scale, (cy + bh / 2 - pad_y) / scale,
        ], axis=1)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)

        return [
            Detection(
                bbox=(float(xyxy[i][0]), float(xyxy[i][1]),
                      float(xyxy[i][2]), float(xyxy[i][3])),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
                class_name=self.class_names[int(class_ids[i])]
                if 0 <= int(class_ids[i]) < len(self.class_names) else "",
            )
            for i in nms(xyxy, scores, self.nms_iou)
        ]

    def close(self) -> None:
        self._session = None
