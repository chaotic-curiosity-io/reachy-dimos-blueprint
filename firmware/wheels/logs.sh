#!/usr/bin/env bash
# Fetch the board's in-memory log ring buffer.
#   HOST=<board-ip> ./logs.sh      |   ./logs.sh <board-ip>
#
# The board serves one connection at a time, so a request that lands while it is
# busy is simply refused. Retry rather than treating that as an error.
set -uo pipefail
HOST="${1:-${HOST:-}}"
if [ -z "$HOST" ]; then
    echo "error: no board address. Usage: HOST=<board-ip> ./logs.sh" >&2
    exit 2
fi

for attempt in 1 2 3 4 5; do
    body=$(curl -s --max-time 10 "http://$HOST/log" 2>/dev/null) || body=""
    if [ -n "$body" ]; then
        printf '%s' "$body" | python3 -c '
import json, sys
try:
    lines = json.load(sys.stdin)["lines"]
except Exception as exc:
    sys.exit("could not parse response: %s" % exc)
if not lines:
    print("(log empty)")
for l in lines:
    print(l)
' && exit 0
    fi
    sleep 1
done
echo "no response from $HOST after 5 attempts" >&2
exit 1
