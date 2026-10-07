"""Env-backed configuration dataclass for the scanner app.

Variables (all optional, sensible defaults):

  DIMOS_SCANNER_BRIDGE_HOST     LAN IP / hostname of your station (the Mac or
                                DGX running the station-side bridge in
                                ``station/``). No default: the app idles
                                until you set it here or on the settings page.
  DIMOS_SCANNER_BRIDGE_PORT     default 9876
  DIMOS_SCANNER_JPEG_QUALITY    1..95, default 80
  DIMOS_SCANNER_FRAME_HZ        target send rate, default 5.0
  DIMOS_SCANNER_POSE_HZ         head-pose send rate, default 20.0
  DIMOS_SCANNER_IMU_HZ          IMU poll/send rate, default 50.0
  DIMOS_SCANNER_IMU_ENABLED     0/1, default 1 (wireless variant only)
  DIMOS_SCANNER_SETTINGS_URL    where the in-app settings page binds,
                                default http://0.0.0.0:8042
  DIMOS_SCANNER_CONFIG          persisted settings file, default
                                ~/.config/dimos_scanner/config.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# Where the settings page persists its config so it survives app restarts /
# redeploys (env defaults are only a fallback). Override with DIMOS_SCANNER_CONFIG.
CONFIG_PATH = Path(
    os.environ.get("DIMOS_SCANNER_CONFIG",
                   str(Path.home() / ".config" / "dimos_scanner" / "config.json"))
)

# Fields written to / read from the persisted config file (everything the
# settings page edits; settings_url is infra and stays env-driven).
_PERSIST_FIELDS = (
    "host", "port", "jpeg_quality", "frame_hz", "pose_hz", "imu_hz", "imu_enabled",
    "depth_model", "pose", "display_width", "max_fps", "device",
    "enable_clip_memory", "save_map", "no_detect", "launcher_url",
)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Depth model presets the Mac-side dimos pipeline knows how to spin up. The
# robot's settings page surfaces these as a dropdown. The Mac server translates
# them into dimos's underlying ``--depth`` + ``--da3-model`` flags.
DEPTH_MODEL_CHOICES = (
    "da3-small",   # Depth Anything 3 small — fastest, default
    "da3-base",    # Depth Anything 3 base — slower, more dynamic range
    "da3-large",   # Depth Anything 3 large — slowest, best quality (NOT metric)
    "da3metric-large",  # Depth Anything 3 metric large — true metric depth on MPS
    "depthpro",    # Apple DepthPro — sharp metric depth, ~1-3 s/frame on MPS
)


# "external" drives the Mac pipeline from the robot's streamed head pose
# (kinematics) — no drift and correct under in-place rotation, unlike "vo".
# Default for the scanner since the robot pans in place.
POSE_CHOICES = ("external", "vo", "identity")
DEVICE_CHOICES = ("mps", "cuda", "cpu")


@dataclass
class BridgeConfig:
    # LAN IP / hostname of your station (the Mac or DGX running the bridge in
    # ``station/``). Deliberately empty: there is no sensible default, so the
    # app idles until you set it via DIMOS_SCANNER_BRIDGE_HOST or the settings
    # page (which persists it across restarts and redeploys).
    host: str = ""
    port: int = 9876
    jpeg_quality: int = 80
    frame_hz: float = 5.0
    # Head-pose stream rate (dedicated WS task, decoupled from frames). Must
    # exceed frame_hz so the Mac can interpolate a pose for every frame's
    # capture timestamp; <= 0 reverts to one pose sent just before each frame.
    pose_hz: float = 20.0
    # IMU streaming (wireless variant only — Reachy Mini Lite has no IMU).
    # ``imu_enabled`` is toggleable live from the settings page without a
    # bridge reconnect; ``imu_hz`` requires a reconnect to take effect.
    imu_hz: float = 50.0
    imu_enabled: bool = True
    # Initial depth-model preference sent to the Mac server in the WS hello.
    # Changing this mid-run reconnects to the bridge; the Mac server needs to
    # be restarted to actually swap the loaded model. Default to the true-metric
    # model: metric depth needs no per-frame scale re-fit against the map (a
    # smear source), so the accumulated cloud stays geometrically stable.
    depth_model: str = "da3metric-large"
    settings_url: str | None = "http://0.0.0.0:8042"

    # ---- Mac-side dimos pipeline knobs ---------------------------------
    # These don't affect the on-robot WS client. They're persisted here so the
    # settings page has a single source of truth; mirror them in the flags you
    # start the station pipeline with. (The original full stack also POSTed
    # them to a station-side launcher daemon, which is not part of this repo.)
    pose: str = "external"
    display_width: int = 768
    max_fps: float = 2.0
    device: str = "mps"
    enable_clip_memory: bool = False
    save_map: bool = True
    no_detect: bool = False

    # Optional station-side launcher daemon (not shipped in this repo). Leave
    # empty to derive ``http://<host>:18765`` from the bridge host.
    launcher_url: str = ""

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        return cls(
            host=os.environ.get("DIMOS_SCANNER_BRIDGE_HOST", ""),
            port=_envi("DIMOS_SCANNER_BRIDGE_PORT", 9876),
            jpeg_quality=_envi("DIMOS_SCANNER_JPEG_QUALITY", 80),
            frame_hz=_envf("DIMOS_SCANNER_FRAME_HZ", 5.0),
            pose_hz=_envf("DIMOS_SCANNER_POSE_HZ", 20.0),
            imu_hz=_envf("DIMOS_SCANNER_IMU_HZ", 50.0),
            imu_enabled=os.environ.get("DIMOS_SCANNER_IMU_ENABLED", "1").lower() in ("1", "true", "yes"),
            depth_model=os.environ.get("DIMOS_SCANNER_DEPTH_MODEL", "da3metric-large"),
            settings_url=os.environ.get("DIMOS_SCANNER_SETTINGS_URL", "http://0.0.0.0:8042"),
            pose=os.environ.get("DIMOS_SCANNER_POSE", "external"),
            display_width=_envi("DIMOS_SCANNER_DISPLAY_WIDTH", 768),
            max_fps=_envf("DIMOS_SCANNER_MAX_FPS", 2.0),
            device=os.environ.get("DIMOS_SCANNER_DEVICE", "mps"),
            enable_clip_memory=os.environ.get("DIMOS_SCANNER_ENABLE_CLIP", "").lower() in ("1", "true", "yes"),
            save_map=os.environ.get("DIMOS_SCANNER_SAVE_MAP", "1").lower() in ("1", "true", "yes"),
            no_detect=os.environ.get("DIMOS_SCANNER_NO_DETECT", "").lower() in ("1", "true", "yes"),
            launcher_url=os.environ.get("DIMOS_SCANNER_LAUNCHER_URL", ""),
        )

    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def derived_launcher_url(self) -> str:
        """Default to the bridge host on port 18765 if no override is set."""
        if self.launcher_url:
            return self.launcher_url.rstrip("/")
        if not self.host:
            return ""
        return f"http://{self.host}:18765"

    def replace(self, **kwargs) -> "BridgeConfig":
        # Convenience used by the settings page to apply partial updates.
        return replace(self, **kwargs)


def load_config() -> BridgeConfig:
    """Env defaults, overlaid with the persisted settings file if present.

    Precedence: persisted file > DIMOS_SCANNER_* env > hardcoded defaults. This is
    what makes the bridge host (and every other setting) stick across app
    restarts and redeploys, instead of resetting every time.
    """
    cfg = BridgeConfig.from_env()
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            kw = {k: data[k] for k in _PERSIST_FIELDS
                  if k in data and data[k] is not None}
            if kw:
                cfg = cfg.replace(**kw)
    except Exception:  # noqa: BLE001 — a corrupt file must not brick the app
        pass
    return cfg


def persist_config(cfg: BridgeConfig) -> None:
    """Write the editable fields to CONFIG_PATH (best-effort, never raises)."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(cfg, k) for k in _PERSIST_FIELDS}
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception:  # noqa: BLE001
        pass
