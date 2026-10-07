#!/usr/bin/env bash
# One command, one terminal: live mono-RGB scan of a room into dimOS, viewed in Rerun.
#
# This script:
#   * runs the station server (station/dimos_bridge/server.py) in THIS terminal,
#     so arrow keys / WASD pan-tilt the robot's head right here;
#   * lets the dimOS pipeline spawn the Rerun viewer itself (--viz rerun) — you
#     do NOT start Rerun separately;
#   * auto-detects this machine's LAN IP and points the running dimos_scanner
#     app on the robot at it (POST /config), cold-starting the app through the
#     robot daemon if it isn't answering.
#
# Usage:
#   ./scripts/scan.sh                        # scan a room with DepthPro; SAVES a map on quit
#   ./scripts/scan.sh --no-save              # scan without saving
#   ./scripts/scan.sh --robot <robot-ip>     # override robot host (default reachy-mini.local)
#   ./scripts/scan.sh --station-ip <ip>      # skip LAN-IP autodetection
#   ./scripts/scan.sh --device cuda          # NVIDIA station (e.g. DGX Spark); default mps
#   ./scripts/scan.sh --depth da3 --da3-model da3metric-large   # Depth-Anything-3 instead
#                                            #   (needs the fork's vendored DA3 source)
#   ./scripts/scan.sh -- --max-fps 3         # anything after `--` goes to the pipeline
#
# Environment:
#   PYTHON             interpreter of the dimOS env (default: `python` on PATH)
#   DIMOS_DIR          path to the dimOS fork checkout (required; see station/README.md)
#   ROBOT_HOST         robot host (default reachy-mini.local; --robot overrides)
#   REACHY_RERUN_PORT  Rerun viewer gRPC port (default 9224, see below)
#   NO_KILL=1          error out instead of stopping a stale server on the bridge port
#
# Maps + keyframes are saved by the dimOS pipeline under ~/.dimos/sessions/:
#   ls -lt ~/.dimos/sessions/*.pkl
#
# Prereqs: dimos_scanner deployed to the robot (./scripts/deploy.sh robot
# robot/scanner_app) and launched from the Reachy dashboard; rerun-sdk installed
# in the dimOS env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python}"

ROBOT_HOST="${ROBOT_HOST:-reachy-mini.local}"
STATION_IP=""
WIRE=1
DEPTH="depthpro"
DA3_MODEL="da3metric-large"
DEVICE="${DIMOS_DEVICE:-mps}"
SAVE=1
MV_WINDOW=0        # DA3-only multi-view refinement window (0 = off)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot)       ROBOT_HOST="$2"; shift 2 ;;
    --station-ip|--mac-ip) STATION_IP="$2"; shift 2 ;;
    --save)        SAVE=1; shift ;;
    --no-save)     SAVE=0; shift ;;
    --no-wire)     WIRE=0; shift ;;
    --depth)       DEPTH="$2"; shift 2 ;;
    --da3-model)   DA3_MODEL="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    --mv-window)   MV_WINDOW="$2"; shift 2 ;;
    --)            shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "error: unknown flag '$1' (pipeline args go after '--')" >&2; exit 2 ;;
  esac
done

command -v "$PYTHON" >/dev/null 2>&1 || { echo "error: python not found: $PYTHON (set PYTHON=...)" >&2; exit 2; }
[[ -n "${DIMOS_DIR:-}" ]] || { echo "error: set DIMOS_DIR to your dimOS fork checkout (see station/README.md)" >&2; exit 2; }
[[ -d "$DIMOS_DIR" ]] || { echo "error: DIMOS_DIR=$DIMOS_DIR is not a directory" >&2; exit 2; }

# The bridge listens on 9879, NOT the protocol default 9876: Rerun's gRPC server
# (including viewers spawned by OTHER projects) also defaults to port 9876, so
# any open Rerun viewer would squat the bridge port and the robot would connect
# to Rerun instead of the station. 9879 is ours alone; the POST /config below
# tells the robot app to dial it.
#
# Free the bridge port if a previous pipeline is still holding it. Only this tool
# binds 9879, so the holder is always a stale scan.sh server — left running it
# makes the new bridge fail to bind and the pipeline wait for frames forever.
BRIDGE_PORT=9879
STALE="$(lsof -ti "tcp:$BRIDGE_PORT" 2>/dev/null || true)"
if [[ -n "$STALE" ]]; then
  if [[ "${NO_KILL:-0}" == 1 ]]; then
    echo "error: port $BRIDGE_PORT already in use (PID $(echo $STALE | tr '\n' ' ')) — stop the" >&2
    echo "       previous pipeline first (kill $STALE)" >&2
    exit 1
  fi
  echo "[scan] port $BRIDGE_PORT busy (PID $(echo $STALE | tr '\n' ' ')) — stopping the previous pipeline ..."
  kill $STALE 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    sleep 1
    lsof -ti "tcp:$BRIDGE_PORT" >/dev/null 2>&1 || break
  done
  if lsof -ti "tcp:$BRIDGE_PORT" >/dev/null 2>&1; then
    echo "error: port $BRIDGE_PORT still busy after kill — free it manually: lsof -i :$BRIDGE_PORT" >&2
    exit 1
  fi
fi

# And move the Rerun viewer itself off 9876 too, so a viewer spawned by this
# run never collides with anything else using Rerun's default (honoured by the
# fork's viz_backend.RerunViz).
export REACHY_RERUN_PORT="${REACHY_RERUN_PORT:-9224}"

# Auto-detect this machine's LAN IP: macOS first, then Linux.
if [[ -z "$STATION_IP" ]]; then
  if command -v ipconfig >/dev/null 2>&1; then
    for iface in en0 en1 en6; do
      STATION_IP="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
      [[ -n "$STATION_IP" ]] && break
    done
  fi
  if [[ -z "$STATION_IP" ]] && command -v hostname >/dev/null 2>&1; then
    STATION_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  fi
  [[ -n "$STATION_IP" ]] || { echo "error: couldn't auto-detect LAN IP; pass --station-ip <ip>" >&2; exit 2; }
fi

SERVER_ARGS=(--dimos-dir "$DIMOS_DIR" --depth "$DEPTH" --pose external
             --display-width 768 --max-fps 2.0 --device "$DEVICE"
             --ws-port "$BRIDGE_PORT")
ROBOT_DEPTH_MODEL="depthpro"
if [[ "$DEPTH" == "da3" ]]; then
  SERVER_ARGS+=(--da3-model "$DA3_MODEL")
  ROBOT_DEPTH_MODEL="$DA3_MODEL"
  # Multi-view windowed refinement (DA3 only): a background worker re-runs DA3
  # over sliding keyframe windows, publishing /accumulated_cloud_refined.
  [[ "$MV_WINDOW" -gt 0 ]] && SERVER_ARGS+=(--mv-window "$MV_WINDOW")
fi

# Save a reusable map (+ keyframes with depth). Absolute path: the server
# chdirs into the dimOS checkout before the pipeline resolves it.
SAVE_PATH=""
if [[ $SAVE == 1 ]]; then
  SAVE_PATH="$HOME/.dimos/sessions/reachy_$(date +%Y%m%d_%H%M%S).pkl"
fi

# Force Rerun-only (skips the LCM/Foxglove path). Everything after --extra goes
# to the pipeline, so --extra must be the last server flag.
SERVER_ARGS+=(--extra --viz rerun)
if [[ -n "$SAVE_PATH" ]]; then
  SERVER_ARGS+=(--save-map "$SAVE_PATH" --save-keyframes "${SAVE_PATH%.pkl}_kf" --keyframe-save-depth)
fi
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && SERVER_ARGS+=("${EXTRA_ARGS[@]}")

echo "==[ reachy scan -> dimOS -> Rerun ]==============================="
echo "station LAN ip: $STATION_IP   robot: $ROBOT_HOST   depth: $DEPTH   device: $DEVICE"
echo "MODE:           SCAN (move the head to cover the room)"
if [[ -n "$SAVE_PATH" ]]; then
  echo "SAVES TO:       $SAVE_PATH   (on quit — press q or Ctrl-C to stop & save)"
else
  echo "SAVES TO:       (not saving — pass --save to keep this run)"
fi
echo "Rerun viewer opens automatically (gRPC port $REACHY_RERUN_PORT). q or Ctrl-C = stop."
echo "=================================================================="

# Point the running robot app at this station (in the background — the server
# below must be listening first): try the live /config reconfigure, else
# cold-start the app via the robot daemon, then retry.
if [[ $WIRE == 1 ]]; then
  (
    body="$(printf '{"host":"%s","port":%d,"depth_model":"%s"}' "$STATION_IP" "$BRIDGE_PORT" "$ROBOT_DEPTH_MODEL")"
    post() { curl -sS --max-time 3 -X POST "http://$ROBOT_HOST:8042/config" \
               -H 'Content-Type: application/json' -d "$body" 2>/dev/null | grep -q '"ok":true'; }
    for _ in 1 2 3; do sleep 2; if post; then echo "[scan] pointed robot at $STATION_IP:$BRIDGE_PORT (live)"; exit 0; fi; done
    echo "[scan] app not answering on :8042 — cold-starting via the daemon ..."
    curl -sS --max-time 4 -X POST "http://$ROBOT_HOST:8000/api/apps/start-app/dimos_scanner" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8; do sleep 2; if post; then echo "[scan] started app + pointed at $STATION_IP:$BRIDGE_PORT"; exit 0; fi; done
    echo "[scan] WARNING: couldn't reach dimos_scanner at $ROBOT_HOST — launch it from the Reachy dashboard, or set host=$STATION_IP port=$BRIDGE_PORT at http://$ROBOT_HOST:8042/"
  ) &
fi

cd "$REPO"
exec "$PYTHON" -m station.dimos_bridge.server "${SERVER_ARGS[@]}"
