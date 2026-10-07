#!/usr/bin/env bash
# Copy the device/ package onto the ESP32 over USB serial and soft-reset it.
#
#   ./deploy.sh                          # auto-detects a single USB-serial port
#   ./deploy.sh /dev/cu.usbserial-XXXX   # or name the port explicitly
#   PORT=/dev/ttyUSB0 ./deploy.sh        # same, via the environment
#
# Needs mpremote (pip install mpremote, or: uv tool install mpremote).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FILES="pins.py motors.py selftest.py netcfg.py wifi.py page.py server.py boot.py main.py"

if [ ! -f "$HERE/device/netcfg.py" ]; then
    echo "error: device/netcfg.py is missing." >&2
    echo "  cp device/netcfg.example.py device/netcfg.py   # then set your WiFi + token" >&2
    exit 1
fi

PORT="${1:-${PORT:-}}"
if [ -z "$PORT" ]; then
    # macOS names CP210x / CH340 adapters /dev/cu.usbserial-* or
    # /dev/cu.wchusbserial-*; Linux uses /dev/ttyUSB* or /dev/ttyACM*.
    candidates=$(ls /dev/cu.usbserial-* /dev/cu.wchusbserial-* /dev/cu.SLAB_USBtoUART* \
                    /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true)
    count=$(printf '%s\n' "$candidates" | grep -c . || true)
    if [ "$count" = "1" ]; then
        PORT="$candidates"
    elif [ "$count" = "0" ]; then
        echo "error: no USB-serial port found. Plug in the ESP32, or pass the port:" >&2
        echo "  ./deploy.sh /dev/cu.usbserial-XXXX" >&2
        exit 1
    else
        echo "error: several serial ports found; pass one explicitly:" >&2
        printf '  %s\n' $candidates >&2
        exit 1
    fi
fi

echo "Deploying to $PORT"
for f in $FILES; do
    echo "  + $f"
    mpremote connect "$PORT" fs cp "$HERE/device/$f" ":$f"
done

mpremote connect "$PORT" soft-reset
echo "Done. REPL: mpremote connect $PORT repl   (ctrl-] to exit)"
