"""OnnxDetector against a real onnxruntime session, end to end.

The unit tests cover letterbox/decode/NMS in isolation; this one runs an
actual (tiny, synthetic) ONNX graph through the whole detector so the part
that is easy to get wrong — mapping a box from letterboxed model pixels
back to original frame pixels — is proven rather than reasoned about.

Skipped when onnx (the graph builder) isn't installed; onnxruntime alone is
what the robot needs at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="onnx needed to synthesise a model")
pytest.importorskip("onnxruntime")

from reachy_wheels_app.tracking.detect_onnx import OnnxDetector  # noqa: E402
from reachy_wheels_app.tracking.vocab import COCO_CLASSES  # noqa: E402

IMGSZ = 320
NUM_CLASSES = 80


def build_constant_model(path, boxes_scores):
    """A graph that ignores its input and emits a fixed (1, 84, N) tensor.

    That is the v8/v11 export layout: rows are cx, cy, w, h then one score
    per class, in letterboxed model pixels.
    """
    from onnx import TensorProto, helper, numpy_helper

    n = len(boxes_scores)
    out = np.zeros((1, 4 + NUM_CLASSES, n), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(boxes_scores):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + class_id, i] = score

    const = helper.make_node(
        "Constant", [], ["output"],
        value=numpy_helper.from_array(out, name="const"))
    graph = helper.make_graph(
        [const], "fake_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                       [1, 3, IMGSZ, IMGSZ])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, 4 + NUM_CLASSES, n])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9          # match what onnxruntime 1.16+ accepts
    onnx.save(model, str(path))
    return path


@pytest.fixture()
def detector_factory(tmp_path):
    def build(boxes_scores, **kwargs):
        path = build_constant_model(tmp_path / "fake.onnx", boxes_scores)
        return OnnxDetector(str(path), imgsz=IMGSZ, **kwargs)
    return build


def test_a_centred_box_comes_back_centred_in_frame_pixels(detector_factory):
    # 640x480 letterboxes to 320x320 with scale 0.5 and pad_y 40, so the
    # frame centre (320, 240) lands at model (160, 160).
    detector = detector_factory([(160, 160, 100, 200, 0, 0.9)])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    out = detector.detect(frame)
    assert len(out) == 1
    det = out[0]
    assert det.class_name == "person"
    assert det.score == pytest.approx(0.9, abs=1e-3)
    cx, cy = det.center
    assert cx == pytest.approx(320, abs=1.0)
    assert cy == pytest.approx(240, abs=1.0)
    # 100x200 model pixels at scale 0.5 → 200x400 frame pixels.
    assert det.width == pytest.approx(200, abs=1.0)
    assert det.height == pytest.approx(400, abs=1.0)


def test_an_off_centre_box_keeps_its_side(detector_factory):
    detector = detector_factory([(240, 160, 40, 40, 0, 0.9)])   # right of centre
    out = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out[0].center[0] > 320


def test_boxes_are_clipped_to_the_frame(detector_factory):
    detector = detector_factory([(10, 10, 200, 200, 0, 0.9)])
    det = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))[0]
    assert det.bbox[0] >= 0 and det.bbox[1] >= 0
    assert det.bbox[2] <= 640 and det.bbox[3] <= 480


def test_the_score_floor_is_applied(detector_factory):
    detector = detector_factory([(160, 160, 40, 40, 0, 0.2)], min_score=0.5)
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_class_filtering_happens_before_anything_reaches_the_caller(detector_factory):
    detector = detector_factory([
        (100, 160, 40, 40, 0, 0.9),     # person
        (220, 160, 40, 40, 16, 0.95),   # dog
    ])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert {d.class_name for d in detector.detect(frame)} == {"person", "dog"}
    only_dogs = detector.detect(frame, targets=("dog",))
    assert [d.class_name for d in only_dogs] == ["dog"]


def test_overlapping_duplicates_are_suppressed(detector_factory):
    detector = detector_factory([
        (160, 160, 100, 100, 0, 0.9),
        (162, 162, 100, 100, 0, 0.8),   # same object, lower score
        (40, 40, 30, 30, 0, 0.7),       # elsewhere
    ])
    out = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(out) == 2
    assert out[0].score == pytest.approx(0.9, abs=1e-3)


def test_a_sidecar_names_file_overrides_the_coco_labels(detector_factory, tmp_path):
    (tmp_path / "fake.names").write_text("\n".join(["widget"] + list(COCO_CLASSES[1:])))
    detector = detector_factory([(160, 160, 40, 40, 0, 0.9)])
    assert detector.vocabulary[0] == "widget"
    assert detector.detect(np.zeros((480, 640, 3), np.uint8))[0].class_name == "widget"


def test_a_junk_frame_is_ignored_rather_than_raising(detector_factory):
    detector = detector_factory([(160, 160, 40, 40, 0, 0.9)])
    assert detector.detect(None) == []
    assert detector.detect(np.zeros((480, 640), dtype=np.uint8)) == []


def test_the_model_file_overrides_a_mismatched_imgsz_setting(tmp_path):
    """A fixed-shape export plus a disagreeing `track_imgsz` is an instant
    runtime failure; the file is the authority."""
    path = build_constant_model(tmp_path / "fake.onnx", [(80, 80, 20, 20, 0, 0.9)])
    detector = OnnxDetector(str(path), imgsz=640)     # deliberately wrong
    assert detector.imgsz == IMGSZ
    assert detector.detect(np.zeros((480, 640, 3), np.uint8))
