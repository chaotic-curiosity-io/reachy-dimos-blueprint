"""Turning what a person *said* into what a detector can *find*.

Two jobs:

* ``resolve_target("follow me")`` → ``{"person"}``, so a spoken phrase from
  the voice agent (or typed in the UI) picks class labels without the caller
  having to know COCO by heart.
* ``known_height_m("person")`` → 1.7, so the follow controller can turn a
  box height in pixels into a rough metres-away estimate.

An open-vocabulary remote detector (``detect_remote``) takes the raw phrase
instead — ``resolve_target`` is a best-effort mapping for closed-vocabulary
detectors, and reports when it had to give up.
"""

from __future__ import annotations

# Standard COCO-80 label order — the output vocabulary of essentially every
# off-the-shelf YOLO/RT-DETR export, and the index space the ONNX detector
# maps class ids through.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Everyday words → COCO labels. Keys are matched against the whole phrase
# and against each of its words, longest key first.
_SYNONYMS: dict[str, str] = {
    "me": "person", "you": "person", "human": "person", "people": "person",
    "human being": "person", "man": "person", "woman": "person",
    "guy": "person", "girl": "person", "boy": "person", "kid": "person",
    "child": "person", "someone": "person", "somebody": "person",
    "my face": "person", "owner": "person",
    "doggy": "dog", "doggie": "dog", "puppy": "dog", "pup": "dog",
    "kitty": "cat", "kitten": "cat", "feline": "cat",
    "ball": "sports ball", "balls": "sports ball", "soccer ball": "sports ball",
    "tennis ball": "sports ball", "basketball": "sports ball",
    "mug": "cup", "coffee": "cup", "coffee cup": "cup", "tea cup": "cup",
    "glass": "wine glass", "water bottle": "bottle", "can": "bottle",
    "phone": "cell phone", "mobile": "cell phone", "iphone": "cell phone",
    "computer": "laptop", "macbook": "laptop", "notebook": "laptop",
    "screen": "tv", "monitor": "tv", "television": "tv", "display": "tv",
    "plant": "potted plant", "houseplant": "potted plant",
    "sofa": "couch", "settee": "couch", "seat": "chair", "stool": "chair",
    "table": "dining table", "desk": "dining table",
    "bag": "backpack", "rucksack": "backpack", "purse": "handbag",
    "controller": "remote", "clicker": "remote",
    "plush": "teddy bear", "stuffed animal": "teddy bear", "bear": "teddy bear",
    "fridge": "refrigerator", "bin": "bowl",
}

# Filler that carries no target information, stripped before matching so
# "follow the person over there" and "person" resolve the same.
_STOPWORDS = frozenset({
    "the", "a", "an", "that", "this", "there", "here", "over", "please",
    "follow", "following", "track", "tracking", "go", "to", "towards",
    "toward", "at", "my", "your", "his", "her", "their", "it", "and",
    "keep", "up", "with", "on", "near", "of",
})

# Rough real-world heights (metres) for a monocular range estimate. Only
# classes where a single number is honest enough to steer a stop distance.
_HEIGHTS_M: dict[str, float] = {
    "person": 1.70, "dog": 0.50, "cat": 0.30, "chair": 0.90, "couch": 0.80,
    "bottle": 0.25, "cup": 0.10, "wine glass": 0.18, "laptop": 0.25,
    "tv": 0.55, "potted plant": 0.45, "backpack": 0.45, "sports ball": 0.22,
    "teddy bear": 0.30, "cell phone": 0.15, "dining table": 0.75,
    "suitcase": 0.60, "refrigerator": 1.70, "bicycle": 1.00,
}


def known_height_m(class_name: str) -> float | None:
    """Typical standing height in metres, or None when we shouldn't guess."""
    return _HEIGHTS_M.get(class_name.strip().lower())


def normalise(phrase: str) -> str:
    return " ".join(str(phrase or "").lower().replace("_", " ").split())


def resolve_target(phrase: str, vocabulary=COCO_CLASSES) -> tuple[set[str], bool]:
    """Map a spoken/typed phrase onto labels in ``vocabulary``.

    Returns ``(labels, exact)``. ``exact`` is False when nothing matched and
    the caller got the empty set — the session surfaces that as "I don't know
    how to look for X" rather than silently following the wrong thing.
    """
    text = normalise(phrase)
    if not text:
        return set(), False

    known = {normalise(c): c for c in vocabulary}
    matched: set[str] = set()

    def take(label: str) -> None:
        canonical = known.get(normalise(label))
        if canonical:
            matched.add(canonical)

    # 1. whole phrase, as a label or a synonym
    if text in known:
        return {known[text]}, True
    if text in _SYNONYMS:
        take(_SYNONYMS[text])
        if matched:
            return matched, True

    # 2. multi-word synonyms/labels appearing inside the phrase, longest
    #    first so "coffee cup" wins over "coffee".
    haystack = f" {text} "
    for key in sorted(set(_SYNONYMS) | set(known), key=len, reverse=True):
        if " " in key and f" {key} " in haystack:
            take(_SYNONYMS.get(key, key))
            if matched:
                return matched, True

    # 3. single significant words
    for word in text.split():
        if word in _STOPWORDS:
            continue
        singular = word[:-1] if len(word) > 3 and word.endswith("s") else word
        for candidate in (word, singular):
            if candidate in _SYNONYMS:
                take(_SYNONYMS[candidate])
            if candidate in known:
                take(candidate)
        if matched:
            break

    return matched, bool(matched)
