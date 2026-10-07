"""Persisted app settings (chassis host, token, default speed).

A small JSON file under ~/.config so choices made in the web UI survive app
restarts and redeploys.

Environment variables (only used as first-run defaults; the web UI's
settings win once saved):

  WHEELS_HOST       LAN IP / hostname of the ESP32 chassis (no default —
                    set it here or in the UI's chassis panel)
  WHEELS_PORT       chassis HTTP port, default 80
  DEPTH_SERVER_URL  origin of the Pi depth streamer, e.g. http://<pi-ip>:8765
                    (read by sensors.py)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

STATE_PATH = Path(os.path.expanduser("~/.config/reachy_wheels_app/state.json"))

DEFAULTS = {
    # ESP32 chassis address. No hardcoded default: set WHEELS_HOST before
    # launching, or type it into the web UI (persisted below).
    "host": os.environ.get("WHEELS_HOST", ""),
    "port": int(os.environ.get("WHEELS_PORT", "80")),
    "token": "",
    # Board-side DEFAULT_SPEED is 0.8 (rotating in place scrubs the mecanum
    # rollers and needs the torque). Start the UI slider there too.
    "default_speed": 0.8,
    # --- wheel lab (manual per-wheel driving) ---------------------------
    # Runtime trims pushed to the board from the lab UI, kept so they can be
    # re-applied after the board resets (it forgets them; only the chassis's
    # own pins.py makes a trim permanent). Empty means "board defaults".
    "wheel_trim": {},        # name -> 0.3..1.5 output scale
    "wheel_invert": {},      # name -> bool polarity override
    # The last mix the lab drove, so a page reload does not throw away an
    # experiment mid-session.
    "wheel_lab_mix": {},     # name -> -1..1
    "wheel_lab_speed": 0.8,
    "wheel_lab_duration": 1.0,
    # --- voice (Gemini Live) -------------------------------------------
    "voice_enabled": True,
    "gemini_api_key": "",     # set from the web UI (POST /api/voice/config)
    "gemini_model": "gemini-3.1-flash-live-preview",
    "gemini_voice": "Aoede",
    "gemini_language": "en-US",  # BCP-47; pins Live speech output language
    "enable_video": True,     # head camera frames into the Live session
    "speak_responses": True,
    "max_drive_seconds": 4.0,  # per voice drive command (board caps at 30)
    # Open-loop motion calibration at speed 0.8 (the reference). Rough
    # estimates until measured: drive a taped metre / a marked 360° and set
    # these to what the chassis actually did.
    "cal_cm_per_s": 30.0,
    "cal_deg_per_s": 80.0,
    # --- visual following (tracking/) -----------------------------------
    # Detector backend: "onnx" runs on the robot over COCO's 80 classes;
    # "remote" posts frames to an open-vocabulary service on the LAN and
    # accepts any phrase. See the README's visual-following section.
    "track_detector": "onnx",
    "track_model_path": "",        # "" → first *.onnx under ~/.config/reachy_wheels_app/models
    # 256 measured on the robot at ~5 Hz with no accuracy cost versus 320
    # (person 0.82 vs 0.81); 320 runs at 2.9 Hz. Overridden anyway by the
    # size baked into a fixed-shape ONNX export.
    "track_imgsz": 256,
    "track_min_score": 0.35,
    "track_nms_iou": 0.45,
    "track_remote_url": "",        # e.g. http://<station-ip>:8055/detect
    "track_remote_timeout": 2.0,
    "track_algorithm": "sort",     # roboflow trackers: sort|bytetrack|ocsort|botsort
    "track_iou": 0.25,
    "track_rate_hz": 8.0,          # perception loop ceiling; detection is the real limit
    # Head-camera field of view — the one number turning pixels into angles,
    # so an over/under-turning follow loop is usually a wrong FOV here.
    "track_hfov_deg": 70.0,
    "track_vfov_deg": 55.0,
    # How the robot is bolted onto the chassis: the angle from the
    # CHASSIS's forward axis to the ROBOT's facing, degrees CCW.
    #   0    robot faces the same way the chassis drives forward
    #  -90   robot faces the chassis's RIGHT — so chassis `strafe_right`
    #        is what carries the robot forwards (the current mounting)
    # Everything upstream of the chassis thinks in the ROBOT's frame; this
    # is the single place that truth gets rotated into the board's frame.
    "track_mount_yaw_deg": -90.0,
    # How far behind the target to hold station, in metres. Used whenever
    # the target class has a height prior (person, dog, cup, …); the frame
    # fraction below is only the fallback for classes without one.
    "track_follow_distance_m": 2.0,
    "track_target_size": 0.55,
    "track_drive_speed": 0.5,      # translations stay gentle on this mount
    "track_rotate_speed": 0.7,
    "track_max_session_seconds": 240.0,
    # Ordered reacquisition: head up/down/left/right, shell reposition + the
    # same head scan, then stopped wheel sectors with a head scan after each.
    "track_search_seconds": 60.0,
    "track_acquire_seconds": 60.0,
    "track_search_waypoint_seconds": 5.0,
    "track_search_waypoint_hold_seconds": 0.35,
    "track_search_body_step_deg": 60.0,
    "track_search_body_seconds": 8.0,
    "track_search_wheel_step_deg": 45.0,
    # Open-loop full-revolution budget; actual omega remains capped at 0.18.
    "track_search_full_turn_seconds": 25.0,
    # Compatibility floor only. The state machine always completes the head
    # and shell stages before allowing wheel rotation.
    "track_search_wheels_after": 0.0,
    # Base spin rate at the search speed — same open-loop calibration the
    # voice drive tools use (cal_deg_per_s).
    "track_rotate_deg_per_s": 80.0,
    # --- mount detection (WiFi RSSI to the chassis beacon) --------------
    # The chassis raises an AP purely for ranging; how loudly the robot's Pi
    # hears it says whether the two are bolted together or metres apart.
    # Both baselines are LEARNED (POST /api/mount/calibrate) rather than
    # modelled — see mount.py for why a path-loss fit does not survive here.
    "mount_enabled": True,
    "mount_iface": "wlan0",
    "mount_ssid": "mecanum-beacon",
    "mount_interval_s": 2.5,   # scanning costs radio time the app shares
    "mount_rssi_on": None,     # dBm measured while bolted to the chassis
    "mount_rssi_off": None,    # dBm measured off the wheels
    "mount_min_separation": 4.0,
}

def resolve_gemini_key(state: dict) -> str:
    """The Gemini API key, taken only from the persisted settings.

    Paste it into the web UI's voice panel (POST /api/voice/config); it is
    stored in STATE_PATH on the robot and never read from anywhere else.
    """
    return str(state.get("gemini_api_key") or "").strip()


def load() -> dict:
    state = dict(DEFAULTS)
    try:
        state.update(json.loads(STATE_PATH.read_text()))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return state


def save(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(STATE_PATH)
    except OSError:
        pass
