#!/usr/bin/env bash
# Run teleop.py with its inline (PEP 723) dependencies via uv, falling back to
# plain python3 (then you need `pip install pyserial` for USB mode yourself).
HERE="$(cd "$(dirname "$0")" && pwd)"
if command -v uv >/dev/null 2>&1; then
    exec uv run --quiet "$HERE/teleop.py" "$@"
fi
exec python3 "$HERE/teleop.py" "$@"
