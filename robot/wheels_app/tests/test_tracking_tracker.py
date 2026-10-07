"""Track association. Roboflow ``trackers`` when installed; these tests pin
the fallback's contract, which is the one that must hold everywhere."""

from __future__ import annotations

from reachy_wheels_app.tracking.tracker import TargetTracker, _GreedyIouTracker
from reachy_wheels_app.tracking.types import Detection


def det(x, y, w=100, h=200, name="person", score=0.9):
    return Detection(bbox=(x, y, x + w, y + h), score=score, class_name=name)


def test_an_object_keeps_its_id_as_it_moves():
    tracker = _GreedyIouTracker()
    first = tracker.update([det(100, 100)])[0]
    later = tracker.update([det(120, 100)])[0]
    assert first.track_id == later.track_id


def test_a_second_object_gets_a_different_id():
    tracker = _GreedyIouTracker()
    out = tracker.update([det(0, 0), det(400, 0)])
    assert out[0].track_id != out[1].track_id


def test_ids_survive_a_distractor_walking_through():
    # This is the whole reason for tracking: the robot must keep following
    # the person it locked onto, not whichever box scores highest now.
    tracker = _GreedyIouTracker()
    mine = tracker.update([det(100, 100)])[0].track_id
    out = tracker.update([det(105, 100), det(400, 100, score=0.99)])
    assert out[0].track_id == mine
    assert out[1].track_id != mine


def test_a_track_never_adopts_a_detection_of_another_class():
    tracker = _GreedyIouTracker()
    person = tracker.update([det(100, 100, name="person")])[0].track_id
    chair = tracker.update([det(100, 100, name="chair")])[0].track_id
    assert chair != person


def test_tracks_age_out_after_enough_empty_frames():
    tracker = _GreedyIouTracker(lost_track_buffer=2)
    first = tracker.update([det(100, 100)])[0].track_id
    for _ in range(4):
        tracker.update([])
    assert tracker.update([det(100, 100)])[0].track_id != first


def test_unconfirmed_tracks_are_marked_minus_one():
    tracker = _GreedyIouTracker(minimum_consecutive_frames=2)
    assert tracker.update([det(100, 100)])[0].track_id == -1
    assert tracker.update([det(100, 100)])[0].track_id > 0


def test_target_tracker_always_produces_a_working_backend():
    tracker = TargetTracker()
    assert tracker.backend
    out = tracker.update([det(50, 50)])
    assert len(out) == 1 and out[0].track_id is not None


def test_target_tracker_forced_onto_the_fallback():
    tracker = TargetTracker(prefer_roboflow=False)
    assert tracker.backend == "fallback-iou"
    assert tracker.update([]) == []


def test_unknown_algorithm_falls_back_to_sort():
    assert TargetTracker(algorithm="nonsense").algorithm == "sort"
