"""Voice control for the wheels app — Gemini Live audio+video session.

Built on proven robot-side Gemini Live plumbing from an earlier Reachy Mini
voice app: mic → 16 kHz PCM in, 24 kHz PCM out to the speaker with mic-mute
while the model talks (the robot hears its own speaker otherwise), camera →
JPEG frames, and tool calls as a silent side channel. Here the tools drive
the Wheels chassis.

Import safety: this package's modules that touch ``google.genai`` or the
robot SDK (``gemini_live``, ``io_harness``) are imported lazily by
``main.py`` only. ``tools`` and ``board`` are pure and offline-testable.
"""
