"""WiFi-RSSI mount detection: is the robot on the wheels or off them?

The dBm figures here are the ones actually measured on this robot, so a
change that would have broken the real signal breaks these too.
"""

from __future__ import annotations

import threading
import time

import pytest

from reachy_wheels_app.mount import (
    MAX_AGE_MS,
    OFF_WHEELS,
    ON_WHEELS,
    NO_SIGNAL,
    UNCALIBRATED,
    UNKNOWN,
    MountMonitor,
    MountThresholds,
    channel_to_freq,
    parse_bss_line,
    strongest,
    thresholds_from_state,
)

# Measured on this robot, bolted to the chassis: -39.0 dBm, sd 0.46.
MOUNTED_DBM = -39.0


def bss(signal, age="4", ssid="mecanum-beacon", bssid="02:00:00:aa:bb:01"):
    return f"{bssid}|{signal}|{age}|{ssid}"


# --- parsing --------------------------------------------------------------

def test_reads_a_beacon_line():
    assert parse_bss_line(bss(-39.0)) == pytest.approx(-39.0)


def test_ignores_other_networks():
    assert parse_bss_line(bss(-30.0, ssid="HomeWiFi")) is None


def test_rejects_a_stale_cached_entry():
    """`iw scan` dumps the driver's whole BSS table, including entries
    NetworkManager cached seconds ago. A stale strong reading standing in
    for a genuine miss is the failure that matters — it would report the
    robot as mounted after it had been lifted off."""
    assert parse_bss_line(bss(-39.0, age=str(MAX_AGE_MS + 500))) is None
    assert parse_bss_line(bss(-39.0, age="4")) == pytest.approx(-39.0)


def test_survives_malformed_lines():
    for junk in ("", "garbage", "a|b|c", "a|notanumber|4|mecanum-beacon"):
        assert parse_bss_line(junk) is None


def test_takes_the_strongest_of_several_interfaces():
    # A board with both interfaces up shows more than one BSS.
    lines = [bss(-52.0, bssid="aa:aa"), bss(-39.0, bssid="bb:bb")]
    assert strongest(lines) == pytest.approx(-39.0)


def test_a_scan_that_heard_nothing_is_none():
    assert strongest([]) is None
    assert strongest([bss(-40.0, ssid="SomeoneElse")]) is None


def test_channel_to_frequency():
    assert channel_to_freq(11) == 2462     # what the board reported live
    assert channel_to_freq(6) == 2437
    assert channel_to_freq(14) == 2484     # the exception
    assert channel_to_freq(0) == 0


# --- deciding -------------------------------------------------------------

def calibrated(on=MOUNTED_DBM, off=-55.0):
    return MountThresholds(on_dbm=on, off_dbm=off)


def test_a_strong_signal_means_it_is_on_the_wheels():
    assert calibrated().classify(-38.0) == ON_WHEELS


def test_a_weak_signal_means_it_is_off_them():
    assert calibrated().classify(-56.0) == OFF_WHEELS


def test_no_beacon_is_reported_as_such_not_as_off():
    # The chassis being powered down is a different fact from the robot
    # having been lifted off it, and conflating them would mislead.
    assert calibrated().classify(None) == NO_SIGNAL


def test_without_both_baselines_it_refuses_to_guess():
    assert MountThresholds(on_dbm=MOUNTED_DBM).classify(-39.0) == UNCALIBRATED
    assert MountThresholds().classify(-39.0) == UNCALIBRATED


def test_baselines_too_close_together_are_not_calibrated():
    # If lifting the robot barely moves the signal, no threshold between
    # them means anything.
    assert not MountThresholds(on_dbm=-39.0, off_dbm=-41.0).calibrated
    assert MountThresholds(on_dbm=-39.0, off_dbm=-55.0).calibrated


def test_the_guard_band_holds_the_previous_state():
    """Without hysteresis a signal sitting near the threshold flips the
    readout on every scan."""
    t = calibrated()
    borderline = t.midpoint
    assert t.classify(borderline, previous=ON_WHEELS) == ON_WHEELS
    assert t.classify(borderline, previous=OFF_WHEELS) == OFF_WHEELS
    assert t.classify(borderline, previous=UNKNOWN) == UNKNOWN


def test_the_guard_band_scales_with_how_separable_the_states_are():
    wide = MountThresholds(on_dbm=-39.0, off_dbm=-69.0)
    narrow = MountThresholds(on_dbm=-39.0, off_dbm=-45.0)
    assert wide.hysteresis > narrow.hysteresis
    assert wide.hysteresis < (wide.on_dbm - wide.off_dbm)


def test_confidence_is_highest_at_the_measured_baselines():
    t = calibrated()
    assert t.confidence(t.on_dbm) == pytest.approx(1.0)
    assert t.confidence(t.midpoint) == pytest.approx(0.0)
    assert t.confidence(None) is None
    assert MountThresholds().confidence(-39.0) is None


def test_thresholds_read_from_app_settings():
    t = thresholds_from_state({"mount_rssi_on": -39.0, "mount_rssi_off": -55.0})
    assert t.calibrated and t.midpoint == pytest.approx(-47.0)


def test_junk_settings_do_not_crash_the_reader():
    assert not thresholds_from_state({"mount_rssi_on": "soon", "mount_rssi_off": None}).calibrated


# --- the monitor thread ---------------------------------------------------

class FakeRadio:
    """A scriptable stand-in for `iw scan`."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, iface, freq, ssid, timeout=15.0):
        with self.lock:
            self.calls += 1
            return self.values[min(self.calls - 1, len(self.values) - 1)]


def monitor(values, state, **kw):
    return MountMonitor(lambda: state, interval=0.05, scan=FakeRadio(values),
                        freq_probe=lambda url, timeout=4.0: 2462, **kw)


def wait_until(predicate, timeout=4.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


CALIBRATED_STATE = {"mount_rssi_on": -39.0, "mount_rssi_off": -60.0}


def test_the_monitor_reports_mounted_from_a_strong_beacon():
    m = monitor([-39.0], CALIBRATED_STATE)
    m.start()
    try:
        assert wait_until(lambda: m.snapshot()["state"] == ON_WHEELS)
        snap = m.snapshot()
        assert snap["smoothed_dbm"] == pytest.approx(-39.0)
        assert snap["confidence"] > 0.9
        assert snap["freq_mhz"] == 2462
    finally:
        m.stop()


def test_the_monitor_notices_the_robot_being_lifted_off():
    radio = FakeRadio([-39.0])
    m = MountMonitor(lambda: CALIBRATED_STATE, interval=0.05, scan=radio,
                     freq_probe=lambda url, timeout=4.0: 2462)
    m.start()
    try:
        assert wait_until(lambda: m.snapshot()["state"] == ON_WHEELS)
        with radio.lock:                      # robot picked up
            radio.values = [-62.0]
            radio.calls = 0
        assert wait_until(lambda: m.snapshot()["state"] == OFF_WHEELS), \
            "never noticed the robot leaving the chassis"
    finally:
        m.stop()


def test_a_silent_beacon_becomes_no_signal():
    m = monitor([None], CALIBRATED_STATE)
    m.start()
    try:
        assert wait_until(lambda: m.snapshot()["state"] == NO_SIGNAL)
    finally:
        m.stop()


def test_an_uncalibrated_monitor_still_reports_the_live_signal():
    m = monitor([-39.0], {})
    m.start()
    try:
        assert wait_until(lambda: m.snapshot()["rssi_dbm"] == -39.0)
        snap = m.snapshot()
        assert snap["state"] == UNCALIBRATED
        assert snap["calibrated"] is False
    finally:
        m.stop()


def test_a_scan_that_raises_does_not_kill_the_monitor():
    class Exploding(FakeRadio):
        def __call__(self, *a, **kw):
            with self.lock:
                self.calls += 1
            raise OSError("radio on fire")

    m = MountMonitor(lambda: CALIBRATED_STATE, interval=0.05, scan=Exploding([]),
                     freq_probe=lambda url, timeout=4.0: 2462)
    m.start()
    try:
        assert wait_until(lambda: m.snapshot()["scans"] >= 0 and m.is_alive())
        time.sleep(0.3)
        assert m.is_alive(), "monitor thread died on a scan error"
    finally:
        m.stop()


def test_radio_scans_pause_while_following_and_resume_afterward():
    active = True
    radio = FakeRadio([-39.0])
    m = MountMonitor(lambda: CALIBRATED_STATE, interval=0.05, scan=radio,
                     freq_probe=lambda url, timeout=4.0: 2462,
                     pause_when=lambda: active)
    m.start()
    try:
        time.sleep(0.2)
        assert radio.calls == 0
        active = False
        assert wait_until(lambda: radio.calls > 0)
    finally:
        m.stop()


def test_calibration_collects_from_the_running_loop():
    # One radio, one scan loop: a second sampler would interleave with the
    # first and both would see less.
    m = monitor([-39.0], {})
    m.start()
    try:
        got = m.collect(samples=3, timeout=5.0)
        assert len(got) >= 3
        assert all(v == -39.0 for v in got)
    finally:
        m.stop()


def test_calibration_gives_up_rather_than_hanging_when_nothing_is_heard():
    m = monitor([None], {})
    m.start()
    try:
        assert m.collect(samples=3, timeout=0.4) == []
    finally:
        m.stop()


def test_the_channel_is_reprobed_after_a_run_of_misses():
    """The router reassigns the board's channel without warning; a stale
    channel-limited scan looks exactly like the beacon being gone."""
    probes = []

    def probe(url, timeout=4.0):
        probes.append(url)
        return 2462 if len(probes) < 3 else 2437

    m = MountMonitor(lambda: CALIBRATED_STATE, interval=0.05,
                     scan=FakeRadio([None]), freq_probe=probe)
    m.start()
    try:
        assert wait_until(lambda: m.snapshot()["freq_mhz"] == 2437, timeout=5.0)
    finally:
        m.stop()
