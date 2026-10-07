"""Phrase → detector labels. The bit that decides whether "follow the doggy"
finds a dog or silently follows a chair."""

from __future__ import annotations

from reachy_wheels_app.tracking.vocab import (
    COCO_CLASSES,
    known_height_m,
    resolve_target,
)


def labels(phrase):
    found, _ = resolve_target(phrase)
    return found


def test_first_person_phrases_mean_a_person():
    for phrase in ("me", "follow me", "the person", "that guy", "human",
                   "somebody", "keep up with me"):
        assert labels(phrase) == {"person"}, phrase


def test_everyday_words_map_onto_coco_labels():
    assert labels("doggy") == {"dog"}
    assert labels("kitty") == {"cat"}
    assert labels("the ball") == {"sports ball"}
    assert labels("my coffee mug") == {"cup"}
    assert labels("the sofa") == {"couch"}


def test_multi_word_synonyms_beat_their_single_words():
    assert labels("coffee cup") == {"cup"}
    assert labels("water bottle") == {"bottle"}


def test_exact_coco_labels_pass_straight_through():
    for label in ("person", "sports ball", "potted plant", "teddy bear"):
        assert labels(label) == {label}


def test_plurals_and_filler_are_tolerated():
    assert labels("the bottles over there") == {"bottle"}
    assert labels("please follow that dog") == {"dog"}


def test_unknown_things_report_failure_rather_than_guessing():
    found, ok = resolve_target("a unicorn")
    assert found == set() and ok is False


def test_filler_only_phrases_resolve_to_nothing():
    assert resolve_target("the thing over there")[1] is False
    assert resolve_target("")[1] is False


def test_resolution_is_scoped_to_the_given_vocabulary():
    # An open-vocab backend passes its own (or no) label set; a closed one
    # must never be handed a label it cannot produce.
    found, ok = resolve_target("dog", vocabulary=("person",))
    assert found == set() and ok is False


def test_height_priors_exist_only_where_a_single_number_is_honest():
    assert known_height_m("person") > 1.0
    assert known_height_m("cup") < 0.3
    assert known_height_m("frisbee") is None


def test_coco_list_is_the_standard_eighty():
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[0] == "person"
