"""VoiceBoard — thread-safe status + transcript ring shared between the
Gemini worker thread and the settings FastAPI handlers.

Pure module (no SDK, no google.genai) so the /api/voice endpoints are
offline-testable.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class VoiceBoard:
    def __init__(self, max_lines: int = 60):
        self._lock = threading.Lock()
        self._lines: deque[dict] = deque(maxlen=max_lines)
        self._connected = False
        self._detail = "voice not started"

    def set_status(self, connected: bool, detail: str = "") -> None:
        with self._lock:
            self._connected = connected
            self._detail = detail

    def push(self, who: str, text: str) -> None:
        """who: 'user' | 'reachy' | 'tool' | 'system'."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            # Live transcription streams a word or two at a time — glue
            # consecutive fragments from the same speaker into one line so
            # the UI reads as sentences, not confetti.
            if self._lines and self._lines[-1]["who"] == who \
                    and who in ("user", "reachy") \
                    and time.time() - self._lines[-1]["t"] < 4.0:
                self._lines[-1]["text"] = (self._lines[-1]["text"] + " " + text).strip()
                self._lines[-1]["t"] = time.time()
            else:
                self._lines.append({"t": time.time(), "who": who, "text": text})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "detail": self._detail,
                "transcript": list(self._lines),
            }
