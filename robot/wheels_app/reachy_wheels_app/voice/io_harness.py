"""RobotIOHarness — pumps robot mic + camera into the Gemini driver, and
routes Gemini's audio response to the robot's speaker.

Adapted from a standard Reachy Mini Gemini Live harness, minus face tracking.
All format conversion lives here so the provider stays clean:

  * mic    : (samples, 2) float32 @ 16kHz  → mono int16 PCM bytes @ 16kHz
  * speaker: 24kHz mono int16 PCM bytes    → (samples, 1) float32 @ output rate
  * camera : (H, W, 3) uint8 BGR           → JPEG bytes via PIL
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from .types import AudioFrame, VideoFrame

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

    from .gemini_live import GeminiDriver

_log = logging.getLogger(__name__)

GEMINI_INPUT_RATE = 16_000   # robot mic native rate — pass-through, no resample
GEMINI_OUTPUT_RATE = 24_000  # Gemini emits 24kHz mono int16


class RobotIOHarness:
    """Owns the mic-poll, camera-poll, and speaker-write tasks."""

    def __init__(
        self,
        mini: "ReachyMini",
        loop: asyncio.AbstractEventLoop,
        *,
        enable_video: bool = True,
        speak_responses: bool = True,
        video_fps: float = 1.0,
        mic_poll_seconds: float = 0.05,
    ):
        self._mini = mini
        self._provider: "GeminiDriver | None" = None
        self._loop = loop
        self._enable_video = enable_video
        self._speak_responses = speak_responses
        self._video_period = 1.0 / max(0.1, video_fps)
        self._mic_poll = mic_poll_seconds
        self._tasks: list[asyncio.Task] = []
        self._output_rate: int = GEMINI_OUTPUT_RATE
        self._jpeg_quality = 75

    # --- lifecycle ------------------------------------------------------

    def set_provider(self, provider: "GeminiDriver") -> None:
        self._provider = provider

    def start_tasks(self) -> None:
        if self._provider is None:
            raise RuntimeError("set_provider() must be called before start_tasks()")
        try:
            self._output_rate = int(
                self._mini.media.get_output_audio_samplerate()
            ) or GEMINI_OUTPUT_RATE
        except Exception:  # noqa: BLE001
            _log.warning("get_output_audio_samplerate failed; using %d", GEMINI_OUTPUT_RATE)

        self._tasks.append(self._loop.create_task(self._mic_pump()))
        if self._enable_video:
            self._tasks.append(self._loop.create_task(self._camera_pump()))

    async def stop_tasks(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    # --- audio out (Gemini → speaker) ------------------------------------

    async def audio_output_callback(self, data: bytes, mime: str) -> None:
        if not self._speak_responses or not data:
            return
        try:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if self._output_rate != GEMINI_OUTPUT_RATE:
                from scipy.signal import resample
                target_len = int(round(len(samples) * self._output_rate / GEMINI_OUTPUT_RATE))
                if target_len > 0:
                    samples = resample(samples, target_len).astype(np.float32)
            samples = samples.reshape(-1, 1)
            await self._loop.run_in_executor(
                None, self._mini.media.push_audio_sample, samples
            )
        except Exception:  # noqa: BLE001
            _log.exception("audio_output_callback failed")

    # --- pumps ------------------------------------------------------------

    async def _mic_pump(self) -> None:
        """Robot mic is 16kHz stereo float32; average to mono int16 LE,
        which is what Gemini Live's `audio/pcm;rate=16000` expects."""
        try:
            while True:
                got = await self._loop.run_in_executor(None, self._safe_get_audio_sample)
                if got is None or len(got) == 0:
                    await asyncio.sleep(self._mic_poll)
                    continue
                pcm = self._stereo_float32_to_mono_int16_bytes(got)
                if pcm:
                    await self._provider.push_audio(
                        AudioFrame(data=pcm, sample_rate=GEMINI_INPUT_RATE)
                    )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _log.exception("mic pump crashed")

    async def _camera_pump(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            _log.warning("Pillow not installed — disabling video pump")
            return

        _log.info("camera pump starting (%.2fs period)", self._video_period)
        try:
            while True:
                t0 = time.time()
                frame = await self._loop.run_in_executor(None, self._safe_get_frame)
                if frame is not None:
                    jpeg = self._encode_jpeg(frame, Image)
                    if jpeg:
                        h, w = frame.shape[:2]
                        await self._provider.push_video(VideoFrame(
                            data=jpeg, mime_type="image/jpeg", width=w, height=h,
                        ))
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, self._video_period - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _log.exception("camera pump crashed")

    # --- conversion helpers ----------------------------------------------

    def _safe_get_audio_sample(self):
        try:
            return self._mini.media.get_audio_sample()
        except Exception:  # noqa: BLE001
            return None

    def _safe_get_frame(self):
        try:
            return self._mini.media.get_frame()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _stereo_float32_to_mono_int16_bytes(samples: np.ndarray) -> bytes:
        if samples.ndim == 2 and samples.shape[1] >= 2:
            mono = samples.mean(axis=1)
        elif samples.ndim == 2:
            mono = samples[:, 0]
        else:
            mono = samples
        mono = np.clip(mono, -1.0, 1.0)
        return (mono * 32767.0).astype(np.int16).tobytes()

    def _encode_jpeg(self, frame: np.ndarray, Image) -> bytes:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            return b""
        # SDK media manager returns BGR (GStreamer appsink video/x-raw,
        # format=BGR) — swap to RGB before encoding or Gemini sees red and
        # blue inverted.
        try:
            rgb = np.ascontiguousarray(frame[:, :, 2::-1])
            img = Image.fromarray(rgb, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._jpeg_quality)
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            _log.exception("jpeg encode failed")
            return b""
