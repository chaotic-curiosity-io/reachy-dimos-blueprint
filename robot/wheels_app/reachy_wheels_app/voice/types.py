"""Media frame types shared by the IO harness and the Gemini provider.

Trimmed from an earlier Reachy Mini voice app's types — the wheels app only
needs the frame containers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioFrame:
    """16-bit PCM little-endian, mono."""
    data: bytes
    sample_rate: int
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class VideoFrame:
    """Encoded video frame; default is JPEG."""
    data: bytes
    mime_type: str = "image/jpeg"
    width: int = 0
    height: int = 0
    timestamp: float = field(default_factory=time.time)
