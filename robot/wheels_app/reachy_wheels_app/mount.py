"""Is the robot sitting on the wheels, or off them?

Reachy has no switch that tells it whether it is bolted to the chassis. But
the chassis runs a WiFi beacon (SSID ``mecanum-beacon``, an AP the ESP32
raises purely for ranging), and how loudly the robot's Pi hears it is a
direct proxy for how close the two are. Mounted, the radios are centimetres
apart; off the wheels, metres. On this robot mounted measures about
**-39 dBm with a standard deviation under 0.5 dB** — a very steady signal.

The measurement runs in-process on the robot (`iw scan` on its WiFi
interface), so the app can answer "am I on the wheels?" without anything
external.

## Why it is calibrated rather than modelled

The obvious approach is a path-loss model — convert dBm to metres, call
anything under 0.3 m "mounted". Two things kill that. The fitted exponent
from the controlled trial came out at n=1.15, well outside the physically
plausible 2.0–3.5, so the model does not describe this environment. And
antenna *orientation* moves RSSI as much as a metre of distance does, so
re-seating the robot at a different angle shifts the curve underneath you.

What is reliable is that the two states are far apart and each is stable.
So we learn them: record a few seconds mounted, a few seconds off, and put
the threshold between them. That survives a channel change, a re-mount, or
a different room, none of which a fitted constant would.

## Practical traps, all of which bit during bring-up

* **The channel moves.** The beacon rides the channel the board's STA link
  uses, and the router reassigns that on a whim (it went 6 → 11 mid
  session). A channel-limited scan pointed at a stale number returns
  nothing and looks exactly like "out of range", so we ask the board.
* **`iw scan` returns the driver's whole BSS table**, including entries
  cached seconds ago by NetworkManager's own scans. A stale strong reading
  would silently stand in for a genuine miss, so anything older than
  ``max_age_ms`` is dropped. Every reading here was heard *now*.
* **Scanning costs radio time**, which the app shares with the chassis HTTP
  and a Gemini websocket, so the default interval is unhurried.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass

_log = logging.getLogger(__name__)

DEFAULT_IFACE = "wlan0"
DEFAULT_SSID = "mecanum-beacon"
# The chassis's own HTTP origin (asked for its beacon channel). Derived from
# WHEELS_HOST / WHEELS_PORT like the wheels client; empty when unset, in which
# case channel detection is skipped and the scan covers every channel. The app
# normally passes board_url explicitly from its persisted settings.
_WHEELS_HOST = os.environ.get("WHEELS_HOST", "")
DEFAULT_BOARD_URL = (
    f"http://{_WHEELS_HOST}:{os.environ.get('WHEELS_PORT', '80')}"
    if _WHEELS_HOST else "")
MAX_AGE_MS = 2500          # a BSS entry older than this is a cached echo
MEDIAN_WINDOW = 5

ON_WHEELS = "on_wheels"
OFF_WHEELS = "off_wheels"
NO_SIGNAL = "no_signal"
UNCALIBRATED = "uncalibrated"
UNKNOWN = "unknown"

# One scan, printed as `bssid|signal|age_ms|ssid`. Any offline proximity tool
# you calibrate with must use the same awk — the two must agree on what counts
# as a reading, because the calibration numbers move between them.
_AWK = (
    r'/^BSS /{ b=$2; sub(/\(.*/,"",b); s=""; a="" } '
    r'/last seen:/{ a=$3 } '
    r'/signal:/{ s=$2 } '
    r'/SSID: /{ printf "%s|%s|%s|%s\n", b, s, a, substr($0, index($0,"SSID: ")+6) }'
)


def channel_to_freq(channel: int) -> int:
    """2.4 GHz channel → MHz. 14 is the odd one out."""
    if channel == 14:
        return 2484
    if 1 <= channel <= 13:
        return 2407 + 5 * channel
    if 36 <= channel <= 165:
        return 5000 + 5 * channel
    return 0


def detect_freq(board_url: str, timeout: float = 4.0) -> int | None:
    """Ask the chassis which channel its beacon is on."""
    try:
        with urllib.request.urlopen(
                board_url.rstrip("/") + "/rssi", timeout=timeout) as fh:
            channel = json.loads(fh.read().decode()).get("channel")
    except (OSError, ValueError):
        return None
    return channel_to_freq(int(channel)) if channel else None


def parse_bss_line(line: str, ssid: str = DEFAULT_SSID,
                   max_age_ms: float = MAX_AGE_MS) -> float | None:
    """dBm from one `bssid|signal|age|ssid` line, or None if it doesn't count."""
    parts = line.split("|", 3)
    if len(parts) != 4:
        return None
    _bss, sig, age, name = parts
    if name.strip() != ssid:
        return None
    try:
        value = float(sig)
    except ValueError:
        return None
    try:
        if age and float(age) > max_age_ms:
            return None        # cached from an earlier scan, not heard now
    except ValueError:
        pass
    return value


def strongest(lines, ssid: str = DEFAULT_SSID,
              max_age_ms: float = MAX_AGE_MS) -> float | None:
    """Best fresh reading in one scan's output.

    A board with both interfaces up can appear as more than one BSS; taking
    the strongest tracks the radio rather than whichever entry came first.
    """
    best = None
    for line in lines:
        value = parse_bss_line(line, ssid, max_age_ms)
        if value is not None and (best is None or value > best):
            best = value
    return best


def scan_once(iface: str = DEFAULT_IFACE, freq: int = 0,
              ssid: str = DEFAULT_SSID, timeout: float = 15.0) -> float | None:
    """One `iw scan`, returning the beacon's dBm or None if not heard.

    ``timeout`` is not optional in practice: the driver can wedge a scan
    indefinitely when NetworkManager is scanning the same radio, and an
    unbounded call would hang this thread for good.
    """
    where = f"freq {int(freq)}" if freq else ""
    cmd = (f"timeout {int(timeout)} sudo -n /usr/sbin/iw dev {iface} scan "
           f"{where} 2>/dev/null | awk '{_AWK}'")
    try:
        out = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True,
                             text=True, timeout=timeout + 5)
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("rssi scan failed: %s", exc)
        return None
    return strongest(out.stdout.splitlines(), ssid)


@dataclass(frozen=True)
class MountThresholds:
    """Learned dBm for each state, and how far apart they must stay."""

    on_dbm: float | None = None
    off_dbm: float | None = None
    min_separation: float = 4.0

    @property
    def calibrated(self) -> bool:
        return (self.on_dbm is not None and self.off_dbm is not None
                and (self.on_dbm - self.off_dbm) >= self.min_separation)

    @property
    def midpoint(self) -> float | None:
        if not self.calibrated:
            return None
        return (self.on_dbm + self.off_dbm) / 2.0

    @property
    def hysteresis(self) -> float:
        """Dead band around the threshold, so a borderline signal doesn't
        flip the readout every scan. A fixed fraction of the real gap, so a
        cleanly separated pair gets a wide guard and a marginal one a
        narrow — never wider than the gap itself."""
        if not self.calibrated:
            return 0.0
        return max(1.0, 0.25 * (self.on_dbm - self.off_dbm))

    def classify(self, rssi: float | None, previous: str = UNKNOWN) -> str:
        if rssi is None:
            return NO_SIGNAL
        if not self.calibrated:
            return UNCALIBRATED
        mid, half = self.midpoint, self.hysteresis / 2.0
        if rssi >= mid + half:
            return ON_WHEELS
        if rssi <= mid - half:
            return OFF_WHEELS
        # Inside the guard band: keep whatever we last believed rather than
        # oscillating. An unknown start stays unknown until it clears.
        return previous if previous in (ON_WHEELS, OFF_WHEELS) else UNKNOWN

    def confidence(self, rssi: float | None) -> float | None:
        """0–1: how far past the threshold this reading sits, as a fraction
        of the distance to the matching baseline. 1.0 means "as clear as
        the calibration itself"."""
        if rssi is None or not self.calibrated:
            return None
        mid = self.midpoint
        span = (self.on_dbm - self.off_dbm) / 2.0
        return max(0.0, min(1.0, abs(rssi - mid) / span)) if span > 0 else None


def thresholds_from_state(state: dict) -> MountThresholds:
    def num(key):
        value = state.get(key)
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    return MountThresholds(on_dbm=num("mount_rssi_on"),
                           off_dbm=num("mount_rssi_off"),
                           min_separation=float(
                               state.get("mount_min_separation", 4.0)))


class MountMonitor(threading.Thread):
    """Samples the beacon in the background and publishes a mount state.

    Runs on its own thread because ``iw scan`` blocks for a second or more
    and must never sit in the follow loop or an HTTP handler. Everything a
    reader needs comes out of :meth:`snapshot` under a lock.
    """

    def __init__(self, state_provider, *, iface: str = DEFAULT_IFACE,
                 ssid: str = DEFAULT_SSID, board_url: str = DEFAULT_BOARD_URL,
                 interval: float = 2.5, scan=scan_once, freq_probe=detect_freq,
                 pause_when=None):
        super().__init__(name="reachy-wheels-mount", daemon=True)
        self._state = state_provider
        self._iface = iface
        self._ssid = ssid
        self._board = board_url
        self._interval = max(0.5, float(interval))
        self._scan = scan
        self._freq_probe = freq_probe
        self._pause_when = pause_when

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._freq = 0
        self._history: list[float] = []
        self._state_name = UNKNOWN
        self._last_rssi: float | None = None
        self._last_seen_at: float | None = None
        self._misses = 0
        self._scans = 0
        self._changed_at: float | None = None
        # Calibration is a request the sampling thread picks up, so it can
        # reuse the live scan loop instead of racing it with a second one.
        self._collecting: str | None = None
        self._collected: list[float] = []
        self._collect_target = 0
        self._collect_done = threading.Event()

    # --- lifecycle -------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        _log.info("mount monitor starting (iface=%s ssid=%s)", self._iface, self._ssid)
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                # `iw scan` temporarily occupies the same wlan0 radio used
                # for chassis HTTP. During a follow run that can turn a
                # safety `/stop` into a timeout, so retain the latest mount
                # result and resume sampling once motion is idle. Calibration
                # explicitly requested by the operator is never paused.
                with self._lock:
                    collecting = self._collecting is not None
                if not collecting and self._pause_when and self._pause_when():
                    pass
                else:
                    self._tick()
            except Exception:  # noqa: BLE001 — a scan fault must not end the thread
                _log.exception("mount monitor tick failed")
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))
        _log.info("mount monitor stopped")

    # --- sampling --------------------------------------------------------

    def _ensure_freq(self) -> None:
        """Learn (or re-learn) the beacon's channel.

        Re-probed after a run of misses because the router reassigns the
        board's channel without warning, and a stale channel-limited scan is
        indistinguishable from the beacon being gone.
        """
        if self._freq and self._misses < 4:
            return
        freq = self._freq_probe(self._board)
        if freq and freq != self._freq:
            _log.info("beacon is on %d MHz (was %s)", freq, self._freq or "unknown")
            with self._lock:
                self._freq = freq
        elif not self._freq and not freq:
            # No answer from the board: fall back to a full-band scan, which
            # is slower but does not depend on knowing the channel.
            with self._lock:
                self._freq = 0

    def _tick(self) -> None:
        self._ensure_freq()
        rssi = self._scan(self._iface, self._freq, self._ssid)
        now = time.time()

        with self._lock:
            self._scans += 1
            if rssi is None:
                self._misses += 1
            else:
                self._misses = 0
                self._last_rssi = rssi
                self._last_seen_at = now
                self._history.append(rssi)
                del self._history[:-MEDIAN_WINDOW]
                if self._collecting:
                    self._collected.append(rssi)
                    if len(self._collected) >= self._collect_target:
                        self._collect_done.set()

            smoothed = (statistics.median(self._history)
                        if self._history and self._misses < 3 else None)
            thresholds = thresholds_from_state(dict(self._state()))
            decided = thresholds.classify(smoothed, self._state_name)
            if decided != self._state_name:
                _log.info("mount state: %s → %s (%.1f dBm)", self._state_name,
                          decided, smoothed if smoothed is not None else float("nan"))
                self._state_name = decided
                self._changed_at = now

    # --- calibration -----------------------------------------------------

    def collect(self, samples: int = 8, timeout: float = 45.0) -> list[float]:
        """Gather readings from the live loop, for calibrating one state.

        Deliberately reuses the running sampler rather than starting a
        second scan: two `iw scan` loops on one radio interleave badly and
        each would see fewer, noisier results than one does alone.
        """
        with self._lock:
            self._collecting = "yes"
            self._collected = []
            self._collect_target = max(1, samples)
        self._collect_done.clear()
        self._collect_done.wait(timeout)
        with self._lock:
            got = list(self._collected)
            self._collecting = None
            self._collected = []
        return got

    # --- reporting -------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            history = list(self._history)
            rssi = self._last_rssi
            misses = self._misses
            state_name = self._state_name
            freq = self._freq
            scans = self._scans
            seen_at = self._last_seen_at
            changed_at = self._changed_at

        smoothed = statistics.median(history) if history and misses < 3 else None
        thresholds = thresholds_from_state(dict(self._state()))
        return {
            "state": state_name,
            "rssi_dbm": rssi,
            "smoothed_dbm": None if smoothed is None else round(smoothed, 1),
            "confidence": thresholds.confidence(smoothed),
            "calibrated": thresholds.calibrated,
            "on_dbm": thresholds.on_dbm,
            "off_dbm": thresholds.off_dbm,
            "threshold_dbm": thresholds.midpoint,
            "hysteresis_db": round(thresholds.hysteresis, 1),
            "freq_mhz": freq,
            "scans": scans,
            "misses": misses,
            "last_seen_ago": None if seen_at is None else round(time.time() - seen_at, 1),
            "changed_ago": None if changed_at is None else round(time.time() - changed_at, 1),
            "interval_s": self._interval,
        }
