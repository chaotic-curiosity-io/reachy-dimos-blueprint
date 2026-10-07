"""Bring up the network interface in whichever mode netcfg selects."""

import network
import time
import netcfg

_sta = None             # kept so the server can report link RSSI
_beacon = None          # AP_IF when the ranging beacon is running


def start(timeout=15):
    """Return the IP address the server should be reached on.

    A failed STA join falls back to hosting the AP rather than raising. Without
    that, a router reboot or a moved robot would leave the board with no network
    at all and no way in except a USB cable.
    """
    if netcfg.MODE == "ap":
        return _start_ap()
    try:
        ip = _start_sta(timeout)
    except OSError as exc:
        print("wifi: STA join failed (%s) -- falling back to AP" % exc)
        return _start_ap()
    # Only once the real link is up, and never at its expense.
    if getattr(netcfg, "RANGING_BEACON", False):
        _start_beacon()
    return ip


def _configure_ap(ap, essid, password):
    """Configure an AP across MicroPython releases.

    The keyword naming the security mode was renamed (`authmode` -> `security`)
    and the module-level `AUTH_*` constants moved onto `WLAN`, at different
    times on different ports. Rather than pin one spelling, try them in order of
    preference and keep the first that takes. The last resort is an open AP,
    so the caller must treat an open beacon as a real outcome rather than
    assuming the password stuck.
    """
    attempts = (
        ("security=WLAN.SEC_WPA_WPA2_PSK",
         lambda: ap.config(essid=essid, password=password,
                           security=network.WLAN.SEC_WPA_WPA2_PSK)),
        ("security=WLAN.SEC_WPA2_PSK",
         lambda: ap.config(essid=essid, password=password,
                           security=network.WLAN.SEC_WPA2_PSK)),
        ("authmode=AUTH_WPA_WPA2_PSK",
         lambda: ap.config(essid=essid, password=password,
                           authmode=network.AUTH_WPA_WPA2_PSK)),
        ("password only",
         lambda: ap.config(essid=essid, password=password)),
        ("open",
         lambda: ap.config(essid=essid)),
    )
    for label, attempt in attempts:
        try:
            attempt()
            print("wifi: AP configured (%s)" % label)
            return label
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            print("wifi: AP config %s rejected (%s)" % (label, exc))
    return None


def _start_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    _configure_ap(ap, netcfg.AP_SSID, netcfg.AP_PASSWORD)
    while not ap.active():
        time.sleep_ms(100)
    ip = ap.ifconfig()[0]
    print("AP up: SSID=%r  ip=%s" % (netcfg.AP_SSID, ip))
    return ip


def _start_beacon():
    """Raise a second, beaconing SSID while staying joined as a station.

    A station never transmits anything another station can passively measure --
    its frames go to the AP and nowhere else. An AP beacons on a fixed interval
    to anyone listening, so bringing AP_IF up alongside STA_IF is what makes
    this board's signal strength readable by the robot's `iw scan` without
    either device leaving its own network. The radio is single-channel, so the
    AP lands on whatever channel the STA link is already using.

    Failure here is logged and swallowed: a missing beacon costs an experiment,
    a raised exception costs the drive server.
    """
    global _beacon
    ssid = getattr(netcfg, "RANGING_SSID", "mecanum-beacon")
    try:
        ap = network.WLAN(network.AP_IF)
        ap.active(True)
        _configure_ap(ap, ssid, netcfg.AP_PASSWORD)
        try:
            ap.config(max_clients=1)
        except (AttributeError, ValueError, OSError):
            pass
        deadline = time.ticks_add(time.ticks_ms(), 5000)
        while not ap.active():
            if time.ticks_diff(deadline, time.ticks_ms()) <= 0:
                print("wifi: ranging beacon did not come active")
                return
            time.sleep_ms(100)
        _beacon = ap
        print("wifi: ranging beacon up: SSID=%r channel=%s" % (ssid, channel()))
    except Exception as exc:
        print("wifi: ranging beacon failed (%s) -- continuing without it" % exc)


def channel():
    """Channel the radio is actually on, or None. Both interfaces share it."""
    for iface in (_beacon, _sta):
        if iface is not None:
            try:
                return iface.config("channel")
            except (AttributeError, ValueError, OSError):
                pass
    return None


def beacon_ssid():
    """SSID the ranging beacon is advertising, or None if it is not running."""
    if _beacon is None:
        return None
    try:
        ssid = _beacon.config("essid")
    except (AttributeError, ValueError, OSError):
        return None
    return ssid.decode() if isinstance(ssid, bytes) else ssid


def rssi():
    """Signal strength of our own link to the router, in dBm, or None.

    This is the board's distance to the *access point*, not to the robot. It is
    worth reporting anyway: when both devices move around one shared anchor,
    comparing the two link strengths is a weak second opinion on whether they
    are in the same part of the house.
    """
    if _sta is None:
        return None
    try:
        return _sta.status("rssi")
    except (AttributeError, ValueError, OSError):
        return None


def _start_sta(timeout):
    global _sta

    if not netcfg.STA_SSID:
        raise OSError("netcfg.STA_SSID is empty; set it or use MODE='ap'")

    sta = network.WLAN(network.STA_IF)
    sta.active(True)

    # Disable wifi power save. The default duty-cycles the radio between beacons,
    # which pushed round-trip latency to ~850ms average and 1.7s peak on this
    # board -- fine for telemetry, useless for driving. Costs a little current.
    try:
        sta.config(pm=network.WLAN.PM_NONE)
    except (AttributeError, ValueError, OSError) as exc:
        print("wifi: could not disable power save (%s)" % exc)

    if not sta.isconnected():
        sta.connect(netcfg.STA_SSID, netcfg.STA_PASSWORD)
        deadline = time.ticks_add(time.ticks_ms(), int(timeout * 1000))
        while not sta.isconnected():
            if time.ticks_diff(deadline, time.ticks_ms()) <= 0:
                sta.active(False)
                raise OSError("wifi: could not join %r in %ss" % (netcfg.STA_SSID, timeout))
            time.sleep_ms(200)

    _sta = sta
    ip = sta.ifconfig()[0]
    print("STA up: joined %r  ip=%s" % (netcfg.STA_SSID, ip))
    return ip
