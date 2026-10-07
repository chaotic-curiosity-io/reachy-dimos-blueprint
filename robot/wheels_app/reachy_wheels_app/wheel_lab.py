"""Manual per-wheel driving: mixes, presets and the tuning we can persist.

Rotating in place is the manoeuvre mecanum wheels are worst at. Every roller
is dragged sideways at once, so the pivot only stays put while all four
corners keep grip — and on this chassis (top-heavy, weight biased toward the
robot's end) the light corners slip first and the turn walks instead of
spinning. ``move(omega=…)`` cannot express any of that: it normalises the
four wheels into one symmetric mix.

So this module is deliberately dumb. It carries no control law and no
correction — it just lets a human name four numbers, send them, and watch.
The point is to find out empirically which mix pivots cleanly on this
chassis; only once that is known does it make sense to bake the answer into
the rotate primitives.

Board-side semantics worth remembering while reading a mix:
  * a mix value is a signed fraction of the lab's speed and is NOT
    renormalised (unlike ``move``), so 0.6 really is 60% of it;
  * the board rescales magnitude onto ``pins.MIN_DUTY``..1, so 0.1 is not a
    tenth of the torque of 1.0 — below the dead zone a motor only buzzes;
  * per-wheel trim is applied board-side on top of all of that.
"""

from __future__ import annotations

WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
WHEEL_LABELS = {"front_left": "FL", "front_right": "FR",
                "rear_left": "RL", "rear_right": "RR"}

# The board clamps trim to this band too; mirrored here so the UI can show
# the limits and the API can reject nonsense without a round trip.
TRIM_MIN, TRIM_MAX = 0.3, 1.5


def _cw(fl: float, fr: float, rl: float, rr: float) -> dict:
    return {"front_left": fl, "front_right": fr,
            "rear_left": rl, "rear_right": rr}


# Every preset is written in the CLOCKWISE sense (seen from above); the UI
# offers a mirror button rather than duplicating each one. The board's
# `rotate_cw` is exactly ALL FOUR — it is here as the baseline to beat.
PRESETS = (
    {"key": "all_four", "label": "all four (= rotate_cw)",
     "mix": _cw(1.0, -1.0, 1.0, -1.0),
     "note": "What the built-in rotate does today. Four wheels scrubbing at "
             "once: most torque, most slip."},
    {"key": "diag_fl_rr", "label": "diagonal FL+RR",
     "mix": _cw(1.0, 0.0, 0.0, -1.0),
     "note": "Two opposite corners only. Halves the scrubbing and often "
             "pivots tighter, at the cost of torque."},
    {"key": "diag_fr_rl", "label": "diagonal FR+RL",
     "mix": _cw(0.0, -1.0, 1.0, 0.0),
     "note": "The other diagonal. If one diagonal turns better than the "
             "other, that difference is grip or trim, not kinematics."},
    {"key": "fronts_only", "label": "front pair only",
     "mix": _cw(1.0, -1.0, 0.0, 0.0),
     "note": "Pivots about the rear axle rather than the centre — useful for "
             "telling a weight-distribution problem from a trim problem."},
    {"key": "rears_only", "label": "rear pair only",
     "mix": _cw(0.0, 0.0, 1.0, -1.0),
     "note": "Pivots about the front axle. Compare with the front pair: the "
             "loaded end is the one that actually turns the chassis."},
    {"key": "rear_biased", "label": "rear-biased 0.6 / 1.0",
     "mix": _cw(0.6, -0.6, 1.0, -1.0),
     "note": "Gives the rear more authority. Start here if the chassis "
             "creeps forward while it turns."},
    {"key": "front_biased", "label": "front-biased 1.0 / 0.6",
     "mix": _cw(1.0, -1.0, 0.6, -0.6),
     "note": "The mirror of the above; which one is right depends on where "
             "the robot's mass actually sits on the chassis."},
    {"key": "left_tank", "label": "left side only (tank)",
     "mix": _cw(1.0, 0.0, 1.0, 0.0),
     "note": "Not a pivot at all — it swings the chassis around its right "
             "side. Worth feeling once, to calibrate the others against."},
    {"key": "right_tank", "label": "right side only (tank)",
     "mix": _cw(0.0, -1.0, 0.0, -1.0),
     "note": "The mirror swing. Together these show how much of a bad "
             "rotate is really one side doing all the work."},
)

PRESETS_BY_KEY = {p["key"]: p for p in PRESETS}


def clean_mix(raw: dict | None) -> dict:
    """A full four-wheel mix, clamped to [-1, 1], missing corners at rest."""
    raw = raw or {}
    return {name: max(-1.0, min(1.0, float(raw.get(name) or 0.0)))
            for name in WHEEL_NAMES}


def unknown_wheels(raw: dict | None) -> list:
    """Names in `raw` the chassis has no wheel for (a typo, not a corner)."""
    return sorted(set(raw or {}) - set(WHEEL_NAMES))


def clamp_trim(value: float) -> float:
    return max(TRIM_MIN, min(TRIM_MAX, float(value)))


def mirrored(mix: dict) -> dict:
    """The same mix in the opposite rotational sense."""
    return {name: -value for name, value in clean_mix(mix).items()}


def describe(mix: dict, speed: float) -> str:
    """One line a human can paste back as feedback about what they drove."""
    clean = clean_mix(mix)
    parts = " · ".join(f"{WHEEL_LABELS[name]} {clean[name]:+.2f}"
                       for name in WHEEL_NAMES)
    return f"{parts} @ speed {float(speed):.2f}"


def pins_snippet(trim: dict, invert: dict) -> str:
    """The chassis `pins.py` WHEELS block matching the current tuning.

    The board forgets /tune on reset, so a trim that turns out to be right
    has to be copied into the Wheels project by hand — printing the exact
    block removes the transcription step, and the transcription error.
    """
    trim = trim or {}
    invert = invert or {}
    lines = ["WHEELS = {"]
    for name in WHEEL_NAMES:
        lines.append('    %-16s{"pins": …, "invert": %s, "trim": %.2f},'
                     % ('"%s":' % name, bool(invert.get(name, True)),
                        float(trim.get(name, 1.0))))
    lines.append("}")
    return "\n".join(lines)
