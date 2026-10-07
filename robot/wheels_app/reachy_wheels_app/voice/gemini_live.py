"""Gemini Live session that talks AND drives.

Adapted from an earlier Reachy Mini Gemini Live voice app. What survives
unchanged is the hard-won plumbing:

  * bidirectional audio (16 kHz PCM in, 24 kHz PCM out) + JPEG video in
  * mic muted while the model speaks, with a grace window computed from
    chunk arrival times — on this robot 3.0 s is the margin that stops the
    mic re-capturing the speaker's tail audio as a fresh user turn
  * tool calls answered promptly so the session never stalls
  * per-turn error recovery in the receive loop

What changed: the tool channel now carries drive/stop/wheels_state calls
executed against the Wheels chassis (blocking HTTP → run_in_executor), and
transcripts go to a VoiceBoard callback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .board import VoiceBoard
from .tools import (
    DRIVE_TOOL_DECLS,
    FOLLOW_TOOL_DECLS,
    MOTION_TOOL_DECLS,
    DriveLimits,
    describe_tool_result,
    dispatch_tool,
    response_scheduling,
)
from .types import AudioFrame, VideoFrame

_log = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """You are Reachy Mini, a small companion robot: a head and body shell riding on a four-wheeled mecanum base. You hear the user, see through your head camera, speak, and move with tools.

Conversation: reply in ONE short sentence — at most 15 words. Match the user's tone. ALWAYS speak English; never switch languages unless the user explicitly asks you to.

YOUR BODY — numbers that matter, use them:
  • Head turns ±{head_yaw:.0f}° and the body shell another ±{body_yaw:.0f}° under it: you can face anything within about ±{envelope:.0f}° of wheel-forward WITHOUT driving. Sweep big — one turn_body of 90° beats four of 20°. turn_body replies tell you how much sweep remains each way.
  • Wheels at default speed: forward/strafe ≈ {cm_per_s:.0f} cm per second, spinning ≈ {deg_per_s:.0f}° per second. One drive command runs at most {max_s:.0f} s ≈ {max_cm:.0f} cm or {max_deg:.0f}°.
  • Prefer asking for what you mean: drive with distance_cm (or degrees for spins) and the duration is computed for you; if one command can't cover it, the reply says how far you got — just call drive again.
  • Everything is open-loop — no odometry. Distances are estimates: verify with your camera after each leg.

HOW YOU ARE BUILT — respect the mount:
  • You are TOP-HEAVY on a narrow wheelbase. Accelerate gently: translations at speed 0.4–0.6; only rotation needs 0.7+. Never chain sudden direction reversals.
  • Electronics and your antenna stick out ~10 cm past your shell. Keep that much side clearance, especially rotating near objects.
  • Your camera sits high and CANNOT see the ground within ~20 cm of your own wheels. You are often on a countertop or table: treat any visible surface edge as a cliff — stop a full body-length before it, never drive toward one, and keep any reverse to a few cm.

HOW YOU MOVE — three tiers, always the smallest one that does the job:
  1. `look` (head, instant): glance at and track things. Use it constantly and freely while chatting about the surroundings — look at what you or the user mention, scan left/centre/right when asked what's around. Never drive just to see something that a glance would show you.
  2. `turn_body` (body shell): face something beyond head range, or square up toward where you're about to go. Its reply tells you when the shell is at its limit — that's your cue for tier 3.
  3. `drive` (wheels): actually travel — approach, follow, back off, or rotate beyond the ±{envelope:.0f}° envelope (rotate_ccw = left, rotate_cw = right). Use it when the user asks you to move/come/follow, or when you need a closer look at something genuinely far away.
Every tool runs in the background — speak in the same breath, never wait for a result. You'll be interrupted if something actually failed; only then tell the user.

Your camera moves with your head: after a look/turn/drive, the next frames show the new view — glance first, then describe or drive.

Driving specifics:
  • `center` your head and body before driving forward far, so forward means where you face.
  • The base can strafe sideways without turning; prefer strafing to sidestep obstacles.
  • If anyone says "stop", "wait", "whoa" or sounds alarmed, call `stop` IMMEDIATELY before speaking.
  • You cannot see behind you. Reverse only slowly and briefly, and say you're backing up blind.
  • Don't drive into anything: if the way ahead doesn't look clear on camera, say so instead of driving. Judge distance from the camera: things filling the frame are close; leave ~30 cm margin.
  • Use `wheels_state` if asked whether you're moving.

{following}

Hard rules — NON-NEGOTIABLE:
  • NEVER speak tool names, argument names, numbers, or braces out loud.
  • NEVER narrate that you are calling a tool. Tools are an internal channel — just act, then talk about the world.
  • A speech-only answer is correct when no movement was requested."""


# Two ways to follow something, depending on whether the vision stack is
# wired in. Kept apart because the manual version is a *worse* behaviour
# that the model must not fall back to when the good one is available.
_FOLLOWING_TRACKED = """Following something (a person, a pet, a ball) — you have a vision tracker, USE IT:
  • Call `follow` with what they said in plain words ("me", "the dog", "the ball"). It locks onto that object with your camera and keeps facing and approaching it by itself, correcting several times a second, until stopped. Keep talking normally the whole time — you are not busy.
  • While a follow runs, do NOT also call drive/turn_body/look to chase: you would fight your own tracker. Only `stop`, `stop_following`, or a new `follow` change it.
  • If the reply says there is no detector class for what they asked, tell them plainly what you can follow instead — never quietly follow something else.
  • It ends itself if it loses the target for good, or after a few minutes. Say so when that happens."""

_FOLLOWING_MANUAL = """Following something (a person, a pet): look at it, turn_body to face it, then drive forward in short legs (~50 cm), re-checking the camera between legs and steering with rotate/strafe. Stop when it's close or lost — say so when lost."""


def build_system_prompt(limits: DriveLimits, motion_limits=None,
                        can_follow: bool = False) -> str:
    """Fill the prompt with this robot's actual envelope + calibration."""
    head_yaw = getattr(motion_limits, "head_yaw", 40.0)
    body_yaw = getattr(motion_limits, "body_yaw", 120.0)
    return _PROMPT_TEMPLATE.format(
        following=_FOLLOWING_TRACKED if can_follow else _FOLLOWING_MANUAL,
        head_yaw=head_yaw,
        body_yaw=body_yaw,
        envelope=head_yaw + body_yaw,
        cm_per_s=limits.cm_per_s * limits.default_speed / limits.ref_speed,
        deg_per_s=limits.deg_per_s * limits.default_speed / limits.ref_speed,
        max_s=limits.max_duration,
        max_cm=limits.cm_per_s * limits.max_duration,
        max_deg=limits.deg_per_s * limits.max_duration,
    )


# Default prompt with stock limits — kept for tests / introspection.
SYSTEM_PROMPT = build_system_prompt(DriveLimits())


AudioOutputCallback = Callable[[bytes, str], Awaitable[None] | None]


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str = "gemini-3.1-flash-live-preview"
    voice: str | None = "Aoede"
    # BCP-47 code pinning the output speech language. Live models sometimes
    # drift languages mid-conversation off a mis-heard word; pinning here +
    # the system-prompt rule holds them steady. None → model's default.
    language: str | None = "en-US"


class GeminiDriver:
    """Owns the Live session; pumps in audio/video, speaks back, drives."""

    def __init__(
        self,
        config: GeminiConfig,
        wheels_client,
        board: VoiceBoard,
        audio_output_callback: AudioOutputCallback | None = None,
        limits: DriveLimits | None = None,
        motion=None,  # RobotMotion → unlocks look/turn_body/center (tiers 1–2)
        follow=None,  # FollowManager → unlocks follow/stop_following
        system_prompt: str | None = None,  # None → built from actual limits
        mute_during_response: bool = True,
        unmute_grace_seconds: float = 3.0,
    ):
        self._config = config
        self._wheels = wheels_client
        self._motion = motion
        self._follow = follow
        self._board = board
        self._audio_cb = audio_output_callback
        self._limits = limits or DriveLimits()
        self._system_prompt = system_prompt or build_system_prompt(
            self._limits, getattr(motion, "limits", None),
            can_follow=follow is not None)
        self._mute_during_response = mute_during_response
        self._unmute_grace_seconds = unmute_grace_seconds

        self._client = None
        self._session_ctx = None
        self._session: Any = None
        self._receive_task: asyncio.Task | None = None
        self._expected_speaker_finish_at: float = 0.0

    # --- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "voice mode requires google-genai (pip install google-genai "
                "into /venvs/apps_venv on the robot)"
            ) from e

        self._client = genai.Client(api_key=self._config.api_key)

        speech_config = None
        if self._config.voice or self._config.language:
            voice_config = None
            if self._config.voice:
                voice_config = genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=self._config.voice,
                    ),
                )
            speech_config = genai_types.SpeechConfig(
                voice_config=voice_config,
                language_code=self._config.language or None,
            )

        live_cfg = genai_types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            realtime_input_config=genai_types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=self._system_prompt)],
            ),
            speech_config=speech_config,
            tools=[{"function_declarations": (
                DRIVE_TOOL_DECLS
                + (MOTION_TOOL_DECLS if self._motion else [])
                + (FOLLOW_TOOL_DECLS if self._follow else [])
            )}],
        )

        self._session_ctx = self._client.aio.live.connect(
            model=self._config.model, config=live_cfg
        )
        self._session = await self._session_ctx.__aenter__()
        self._receive_task = asyncio.create_task(self._receive_loop())
        _log.info("gemini-live driver session opened (model=%s)", self._config.model)

    async def stop(self) -> None:
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await asyncio.wait_for(self._receive_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            self._receive_task = None
        if self._session_ctx is not None:
            # Bounded close — websocket teardown can hang on a flaky network
            # and the daemon's Stop button must not wait on it.
            try:
                await asyncio.wait_for(
                    self._session_ctx.__aexit__(None, None, None), timeout=3.0,
                )
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                _log.warning("gemini-live __aexit__ did not complete in 3s")
            self._session_ctx = None
            self._session = None

    # --- input ----------------------------------------------------------

    async def push_audio(self, frame: AudioFrame) -> None:
        if self._session is None:
            return
        if self._mute_during_response and self._audio_cb is not None:
            grace_until = self._expected_speaker_finish_at + self._unmute_grace_seconds
            if time.time() < grace_until:
                return
        from google.genai import types as genai_types

        await self._session.send_realtime_input(
            audio=genai_types.Blob(
                data=frame.data,
                mime_type=f"audio/pcm;rate={frame.sample_rate}",
            ),
        )

    async def push_video(self, frame: VideoFrame) -> None:
        if self._session is None:
            return
        from google.genai import types as genai_types

        await self._session.send_realtime_input(
            video=genai_types.Blob(data=frame.data, mime_type=frame.mime_type),
        )

    # --- receive --------------------------------------------------------

    async def _receive_loop(self) -> None:
        # session.receive() is per-turn; re-enter after each turn. A
        # per-turn error (network blip, malformed frame) must not kill the
        # loop — log, tell the board, back off, re-enter.
        while True:
            try:
                async for response in self._session.receive():
                    await self._handle_response(response)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                _log.exception("gemini-live turn error — continuing")
                self._board.push("system", f"turn error: {e!r}")
                await asyncio.sleep(0.5)
                if self._session is None:
                    return

    async def _handle_response(self, response: Any) -> None:
        tool_call = getattr(response, "tool_call", None)
        if tool_call is not None:
            await self._handle_tool_call(tool_call)
            return

        sc = getattr(response, "server_content", None)
        if sc is None:
            return

        inp = getattr(sc, "input_transcription", None)
        if inp is not None and getattr(inp, "text", None):
            self._board.push("user", inp.text)

        out = getattr(sc, "output_transcription", None)
        if out is not None and getattr(out, "text", None):
            self._board.push("reachy", out.text)

        mt = getattr(sc, "model_turn", None)
        if mt is not None:
            for part in getattr(mt, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline is None:
                    continue
                audio_bytes = getattr(inline, "data", b"") or b""
                mime = getattr(inline, "mime_type", "") or ""
                if audio_bytes:
                    # 24kHz 16-bit mono = 48000 B/s. Track when the speaker
                    # should finish so the mic stays shut until then.
                    chunk_seconds = len(audio_bytes) / 48000.0
                    now = time.time()
                    self._expected_speaker_finish_at = (
                        max(self._expected_speaker_finish_at, now) + chunk_seconds
                    )
                if self._audio_cb is not None and audio_bytes:
                    try:
                        res = self._audio_cb(audio_bytes, mime)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception:  # noqa: BLE001
                        _log.exception("audio_output_callback raised")

    async def _handle_tool_call(self, tool_call: Any) -> None:
        """Launch drive/stop/state calls WITHOUT blocking the receive loop.

        The declarations are NON_BLOCKING, so the model is already talking
        while we work; each call runs as its own task (blocking chassis HTTP
        → executor) and answers with a `scheduling` hint: errors INTERRUPT
        (get spoken now), state lands WHEN_IDLE, successes stay SILENT.
        Handling calls concurrently also means a `stop` never queues behind
        a slow `drive` round-trip.
        """
        for fc in list(getattr(tool_call, "function_calls", None) or []):
            name = getattr(fc, "name", None) or "unknown"
            fc_id = getattr(fc, "id", None)
            args = getattr(fc, "args", None) or {}
            if not isinstance(args, dict):
                try:
                    args = dict(args)
                except Exception:  # noqa: BLE001
                    args = {}
            asyncio.create_task(self._run_tool(name, fc_id, args))

    async def _run_tool(self, name: str, fc_id: Any, args: dict) -> None:
        from google.genai import types as genai_types

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, dispatch_tool, self._wheels, name, args, self._limits,
            self._motion, self._follow,
        )
        self._board.push("tool", describe_tool_result(name, args, result))
        scheduling = response_scheduling(name, result)
        _log.info("tool %s(%s) -> %s [%s]", name, args, result, scheduling)
        if self._session is None:
            return
        try:
            await self._session.send_tool_response(function_responses=[
                genai_types.FunctionResponse(
                    name=name, id=fc_id, response=result,
                    scheduling=genai_types.FunctionResponseScheduling(scheduling),
                ),
            ])
        except Exception:  # noqa: BLE001
            _log.exception("send_tool_response failed")
