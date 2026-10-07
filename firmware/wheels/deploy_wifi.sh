#!/usr/bin/env bash
# Push the device/ package to the chassis over wifi and restart it.
#
#   HOST=<board-ip> ./deploy_wifi.sh
#   ./deploy_wifi.sh <board-ip>
#   HOST=192.0.2.10 TOKEN=secret ./deploy_wifi.sh   # when netcfg.API_TOKEN is set
#
# In AP mode, join the board's own "mecanum-bot" network and use the gateway
# address it prints on the serial console ("AP up: ... ip=..."); in STA mode
# use whatever address your router handed it ("STA up: ... ip=...", or see
# your router's client list).
#
# Needs the board already running server.py (netcfg.AUTOSTART = True).
set -euo pipefail

HOST="${1:-${HOST:-}}"
TOKEN="${TOKEN:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FILES="pins.py motors.py selftest.py netcfg.py wifi.py page.py server.py boot.py main.py"

if [ -z "$HOST" ]; then
    echo "error: no board address. Usage: HOST=<board-ip> ./deploy_wifi.sh" >&2
    exit 2
fi
if [ ! -f "$HERE/device/netcfg.py" ]; then
    echo "error: device/netcfg.py is missing." >&2
    echo "  cp device/netcfg.example.py device/netcfg.py   # then set your WiFi + token" >&2
    exit 1
fi

RESP="$(mktemp)"
trap 'rm -f "$RESP"' EXIT

# Wrapped in a function rather than an array: macOS still ships bash 3.2, where
# expanding an empty array under `set -u` is an error.
api() {
    if [ -n "$TOKEN" ]; then
        curl -H "X-Token: $TOKEN" "$@"
    else
        curl "$@"
    fi
}

echo "Deploying to http://$HOST"
for f in $FILES; do
    printf '  + %-14s' "$f"
    code=000
    for attempt in 1 2 3; do
        code=$(api -s -o "$RESP" -w '%{http_code}' --max-time 20 \
               -X POST --data-binary "@$HERE/device/$f" \
               "http://$HOST/put?path=$f" 2>/dev/null) || code=000
        [ "$code" = "200" ] && break
        sleep 1
    done
    if [ "$code" = "200" ]; then
        echo "ok"
    else
        echo "FAILED (http $code)"; cat "$RESP" 2>/dev/null; echo; exit 1
    fi
done

echo "Restarting..."
api -s --max-time 10 -X POST "http://$HOST/reset" >/dev/null 2>&1 || true

for attempt in 1 2 3 4 5 6 7 8 9 10; do
    sleep 2
    state=$(curl -s --max-time 5 "http://$HOST/state" 2>/dev/null) || state=""
    if [ -n "$state" ]; then
        echo "Back up after $((attempt * 2))s:"
        echo "  $state"
        exit 0
    fi
done
echo "Board did not answer within 20s -- check ./logs.sh or the USB console" >&2
exit 1
