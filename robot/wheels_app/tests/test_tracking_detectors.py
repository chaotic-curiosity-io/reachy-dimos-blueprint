"""Detector plumbing: output decoding, NMS, backend choice, wire parsing.

No model file and no network — the parts that would need either are the
parts that are already someone else's tested code.
"""

from __future__ import annotations

import numpy as np
import pytest

from reachy_wheels_app.tracking.detect_onnx import decode_yolo, letterbox, nms
from reachy_wheels_app.tracking.detect_remote import parse_detections
from reachy_wheels_app.tracking.detectors import (
    BACKENDS,
    DetectorUnavailable,
    NullDetector,
    build_detector,
    filter_detections,
)
from reachy_wheels_app.tracking.types import Detection


# --- ONNX pre/post-processing --------------------------------------------

def test_letterbox_pads_to_a_square_without_distorting():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blob, scale, pad_x, pad_y = letterbox(frame, 320)
    assert blob.shape == (1, 3, 320, 320)
    assert scale == pytest.approx(0.5)
    assert (pad_x, pad_y) == (0, 40)          # 640→320 wide, 480→240 tall
    assert 0.0 <= blob.min() and blob.max() <= 1.0


def test_letterbox_handles_a_portrait_frame():
    blob, scale, pad_x, pad_y = letterbox(np.zeros((640, 480, 3), np.uint8), 320)
    assert blob.shape == (1, 3, 320, 320)
    assert (pad_x, pad_y) == (40, 0)


def test_decode_handles_the_v8_channels_first_layout():
    # (1, 4 + nc, N) — what `yolo export format=onnx` emits today.
    out = np.zeros((1, 84, 10), dtype=np.float32)
    out[0, :4, 3] = [100, 100, 40, 60]
    out[0, 4 + 16, 3] = 0.9                    # class 16 = dog
    boxes, scores, class_ids = decode_yolo(out, 80)
    assert boxes.shape == (10, 4)
    assert scores[3] == pytest.approx(0.9)
    assert class_ids[3] == 16


def test_decode_handles_the_v5_objectness_layout():
    # (1, N, 5 + nc) with a separate objectness column.
    out = np.zeros((1, 10, 85), dtype=np.float32)
    out[0, 2, :4] = [50, 50, 20, 20]
    out[0, 2, 4] = 0.8                          # objectness
    out[0, 2, 5] = 0.5                          # class 0 = person
    boxes, scores, class_ids = decode_yolo(out, 80)
    assert scores[2] == pytest.approx(0.4)      # obj * cls
    assert class_ids[2] == 0


def test_decode_rejects_a_shape_it_cannot_read():
    with pytest.raises(ValueError):
        decode_yolo(np.zeros((2, 3, 4, 5)), 80)


def test_nms_drops_overlaps_and_keeps_the_best():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [400, 400, 500, 500]],
                     dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.45) == [0, 2]


def test_nms_on_nothing_is_nothing():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


# --- backend selection ----------------------------------------------------

def test_unknown_backend_is_refused_by_name():
    with pytest.raises(DetectorUnavailable) as exc:
        build_detector({"track_detector": "magic"})
    assert "magic" in str(exc.value)
    for backend in BACKENDS:
        assert backend in str(exc.value)


def test_remote_backend_without_a_url_says_so():
    with pytest.raises(DetectorUnavailable) as exc:
        build_detector({"track_detector": "remote", "track_remote_url": ""})
    assert "track_remote_url" in str(exc.value)


def test_onnx_backend_without_a_model_explains_how_to_get_one(monkeypatch, tmp_path):
    from reachy_wheels_app.tracking import detect_onnx
    monkeypatch.setattr(detect_onnx, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(detect_onnx, "_HOWTO", "no model; run yolo export")
    monkeypatch.delenv("WHEELS_TRACK_MODEL", raising=False)
    with pytest.raises(DetectorUnavailable) as exc:
        detect_onnx.resolve_model_path("")
    assert "yolo export" in str(exc.value)


def test_a_configured_but_missing_model_path_names_the_path(tmp_path):
    from reachy_wheels_app.tracking import detect_onnx
    with pytest.raises(DetectorUnavailable) as exc:
        detect_onnx.resolve_model_path(str(tmp_path / "nope.onnx"))
    assert "nope.onnx" in str(exc.value)


def test_null_detector_is_a_usable_no_op():
    null = NullDetector()
    assert null.detect(None) == [] and null.vocabulary == ()
    null.close()


# --- filtering ------------------------------------------------------------

def det(name, score=0.9):
    return Detection(bbox=(0, 0, 10, 10), score=score, class_name=name)


def test_filter_keeps_only_the_wanted_classes():
    kept = filter_detections([det("person"), det("chair")], {"person"})
    assert [d.class_name for d in kept] == ["person"]


def test_filter_applies_the_score_floor():
    kept = filter_detections([det("person", 0.2), det("person", 0.8)],
                             {"person"}, min_score=0.5)
    assert len(kept) == 1 and kept[0].score == 0.8


def test_empty_target_set_means_keep_everything():
    assert len(filter_detections([det("person"), det("chair")], set())) == 2


# --- remote wire format ---------------------------------------------------

def test_parses_our_own_reply_shape():
    out = parse_detections({"detections": [
        {"bbox": [1, 2, 11, 22], "score": 0.7, "class_name": "red mug"}]})
    assert len(out) == 1
    assert out[0].bbox == (1.0, 2.0, 11.0, 22.0)
    assert out[0].class_name == "red mug"


def test_parses_roboflow_inference_centre_and_extent():
    out = parse_detections({"predictions": [
        {"x": 100, "y": 100, "width": 40, "height": 20,
         "confidence": 0.6, "class": "dog"}]})
    assert out[0].bbox == (80.0, 90.0, 120.0, 110.0)
    assert out[0].class_name == "dog"


def test_low_scoring_rows_are_dropped_at_the_wire():
    payload = {"detections": [{"bbox": [0, 0, 1, 1], "score": 0.1}]}
    assert parse_detections(payload, min_score=0.5) == []


def test_malformed_rows_are_skipped_not_fatal():
    out = parse_detections({"detections": [
        "nonsense",
        {"score": 0.9},                              # no box at all
        {"bbox": [10, 10, 5, 5], "score": 0.9},      # inverted box
        {"bbox": [0, 0, 4, 4], "score": 0.9, "class_name": "cup"},
    ]})
    assert len(out) == 1 and out[0].class_name == "cup"


def test_an_empty_reply_is_simply_no_detections():
    assert parse_detections({}) == []
