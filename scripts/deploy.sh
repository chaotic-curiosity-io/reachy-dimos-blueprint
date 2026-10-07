#!/usr/bin/env bash
# Install a Reachy Mini app from this repo onto a robot, a simulator or a local daemon.
#
# Three modes, picked by the first positional argument:
#
#   ./scripts/deploy.sh sim     <app-dir>
#       Install the app in editable mode into the same Python env where
#       reachy-mini-daemon runs ($PYTHON), then start the daemon in simulation
#       mode. Open http://127.0.0.1:8000/ and the app appears in the launcher.
#
#   ./scripts/deploy.sh local   <app-dir>
#       Same as sim but without launching the daemon — useful when the
#       daemon is already running. Just registers the entry point.
#
#   ./scripts/deploy.sh robot   <app-dir>   [host]
#       scp the app to the real Reachy Mini and pip-install it into the
#       robot's /venvs/apps_venv. Host defaults to $REACHY_HOST or
#       reachy-mini.local; the ssh user is $REACHY_USER (default: pollen).
#
# Apps in this repo:
#
#   ./scripts/deploy.sh robot robot/scanner_app      # mono RGB streamer -> station (dimos_scanner)
#   ./scripts/deploy.sh robot robot/wheels_app       # wheels driver, follow, sensors (reachy_wheels_app)
#   ./scripts/deploy.sh sim   robot/scanner_app
#
# Environment:
#   PYTHON       interpreter of the env holding reachy_mini + reachy-mini-daemon
#                (sim/local modes only; default: `python` on PATH)
#   REACHY_HOST  robot host for robot mode (default reachy-mini.local)
#   REACHY_USER  ssh user on the robot (default pollen, the stock Reachy user)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" >&2
  exit 1
}

[[ $# -lt 2 ]] && usage
MODE="$1"; APP_DIR="$2"; shift 2
HOST="${1:-${REACHY_HOST:-reachy-mini.local}}"
REACHY_USER="${REACHY_USER:-pollen}"
SSH_TARGET="$REACHY_USER@$HOST"

# Resolve to an absolute path so editable installs survive cwd changes.
# Try $REPO/$APP_DIR first (the common case — argv is relative to the repo
# root), then fall back to $APP_DIR (already absolute, or relative to cwd).
case "$APP_DIR" in
  /*) ;;
  *)
    if [[ -d "$REPO/$APP_DIR" ]]; then
      APP_DIR="$REPO/$APP_DIR"
    fi
    ;;
esac
if [[ ! -d "$APP_DIR" ]]; then
  echo "error: $APP_DIR not found" >&2
  exit 2
fi
APP_DIR="$(cd "$APP_DIR" && pwd)"
if [[ ! -f "$APP_DIR/pyproject.toml" ]]; then
  echo "error: $APP_DIR doesn't look like a Reachy Mini app (no pyproject.toml)" >&2
  exit 2
fi
APP_NAME="$(grep -E '^name *=' "$APP_DIR/pyproject.toml" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
echo "[deploy] app: $APP_NAME at $APP_DIR"

# The reachy_mini SDK + daemon usually live in their own env, separate from the
# station's dimOS env. Point PYTHON at that interpreter for sim/local modes.
PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Robot-mode helpers. The daemon's app state machine can wedge in "stopping":
# `stop-current-app` returns before the app is actually down, and a too-early
# `start-app` then fails with {"detail":"An app is already running"}. These
# poll/retry around that race so a normal deploy self-heals.
#
# !!! NEVER "fix" a stuck deploy by calling POST /cache/reset-apps — despite the
# name it runs shutil.rmtree("/venvs/apps_venv"), DELETING the shared venv and
# every installed app. /cache/reset-apps is destructive — never call it. If the
# daemon is genuinely wedged in "stopping", the safe recovery is:
#   ssh $REACHY_USER@<host> 'sudo systemctl restart reachy-mini-daemon'
# (clears the state; the daemon runs from its own /venvs/mini_daemon env).
# ---------------------------------------------------------------------------

_app_state() {  # echo the daemon's current app state ("" if nothing running)
  local body
  body="$(curl -sS -m 4 "http://$HOST:8000/api/apps/current-app-status" 2>/dev/null)"
  [[ -z "$body" || "$body" == "null" ]] && return 0
  printf '%s' "$body" \
    | grep -o '"state"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 \
    | sed -E 's/.*"([^"]*)"$/\1/'
}

wait_for_app_stopped() {  # poll until no app is running (or timeout); 0 = stopped
  local tries="${1:-30}" i state
  for ((i = 1; i <= tries; i++)); do
    state="$(_app_state)"
    case "$state" in
      "" | stopped | idle | none) return 0 ;;
    esac
    sleep 2
  done
  return 1
}

start_app_with_retry() {  # start $APP_NAME, retrying past the "stopping" race
  local tries="${1:-6}" i resp
  for ((i = 1; i <= tries; i++)); do
    resp="$(curl -sS -m 12 -X POST "http://$HOST:8000/api/apps/start-app/$APP_NAME" 2>&1)"
    if printf '%s' "$resp" | grep -q '"state"'; then
      echo "$resp"
      return 0
    fi
    if printf '%s' "$resp" | grep -qi "already running"; then
      echo "[deploy]   previous app still settling (attempt $i/$tries) — waiting ..."
      wait_for_app_stopped 12 || true
      sleep 2
      continue
    fi
    echo "[deploy]   start-app attempt $i/$tries returned: $resp"
    sleep 2
  done
  return 1
}

case "$MODE" in
  local)
    echo "[deploy] pip install -e $APP_DIR  (with $PYTHON)"
    "$PYTHON" -m pip install -e "$APP_DIR"
    echo "[deploy] OK — entry point registered. Start the daemon if it isn't already running:"
    echo "         reachy-mini-daemon         # Lite version"
    echo "         reachy-mini-daemon --sim   # simulator"
    ;;
  sim)
    echo "[deploy] pip install -e $APP_DIR  (with $PYTHON)"
    "$PYTHON" -m pip install -e "$APP_DIR"
    # The daemon console script sits next to the interpreter in the same env.
    BIN_DIR="$("$PYTHON" -c 'import os, sys; print(os.path.dirname(sys.executable))')"
    DAEMON="$BIN_DIR/reachy-mini-daemon"
    [[ -x "$DAEMON" ]] || DAEMON="$(command -v reachy-mini-daemon || true)"
    [[ -n "$DAEMON" ]] || { echo "error: reachy-mini-daemon not found next to $PYTHON or on PATH" >&2; exit 2; }
    echo "[deploy] starting $DAEMON --sim ..."
    exec "$DAEMON" --sim
    ;;
  robot)
    echo "[deploy] stopping any currently-running app on robot (so the new code loads) ..."
    curl -sS -X POST "http://$HOST:8000/api/apps/stop-current-app" >/dev/null \
      && echo "[deploy]   stop requested" \
      || echo "[deploy]   nothing was running (or daemon unreachable — ok)"
    # Wait for the app to ACTUALLY be down before touching the shared venv /
    # starting the new code — otherwise start-app races the old app's shutdown.
    if wait_for_app_stopped 30; then
      echo "[deploy]   confirmed stopped"
    else
      echo "[deploy]   warning: app still not 'stopped' after ~60s (daemon may be"
      echo "[deploy]   wedged). Continuing; if start-app fails, restart the daemon:"
      echo "[deploy]     ssh $SSH_TARGET 'sudo systemctl restart reachy-mini-daemon'"
    fi
    REMOTE_DIR="/tmp/$(basename "$APP_DIR")"
    echo "[deploy] cleaning $HOST:$REMOTE_DIR ..."
    ssh "$SSH_TARGET" "rm -rf $REMOTE_DIR"
    echo "[deploy] copying $APP_DIR to $HOST ..."
    scp -r "$APP_DIR" "$SSH_TARGET:$REMOTE_DIR"
    echo "[deploy] pip install on robot ..."
    # Uninstall a possible hyphen-vs-underscore twin so a rename (e.g.
    # dimos-scanner -> dimos_scanner) doesn't leave both distributions
    # registered side-by-side. pip is fine if neither variant is installed.
    UNAME_HYPHEN="${APP_NAME//_/-}"
    UNAME_UNDER="${APP_NAME//-/_}"
    ssh "$SSH_TARGET" "/venvs/apps_venv/bin/pip uninstall -y $UNAME_HYPHEN $UNAME_UNDER 2>/dev/null || true"
    # --force-reinstall: pip otherwise no-ops when the source pyproject.toml
    # version (e.g. 0.1.0) matches what's already installed, silently dropping
    # every edit you've made to existing .py files between deploys.
    # --no-deps: keeps the deploy fast — only the app's files are touched, not
    # the (often slow) reinstall of mediapipe/scipy/google-genai/etc. If you
    # ADD a new dependency to pyproject.toml, drop --no-deps for one deploy
    # (or ssh in and `pip install <new-dep>` once).
    ssh "$SSH_TARGET" "/venvs/apps_venv/bin/pip install --no-cache-dir --force-reinstall --no-deps $REMOTE_DIR"
    echo "[deploy] starting app on robot ..."
    if start_app_with_retry 6; then
      echo
      echo "[deploy] OK — '$APP_NAME' starting. Settings UI (if any) will come up shortly."
    else
      echo
      echo "[deploy] start-app did not succeed after retries."
      echo "[deploy]   • open the dashboard at http://$HOST:8000/ and launch '$APP_NAME' manually, OR"
      echo "[deploy]   • if the app is stuck 'stopping', restart the daemon (does NOT touch apps):"
      echo "[deploy]       ssh $SSH_TARGET 'sudo systemctl restart reachy-mini-daemon'"
      echo "[deploy]   Do NOT call /cache/reset-apps — it deletes the shared apps venv."
    fi
    ;;
  *)
    usage
    ;;
esac
