# Wheels — ESP32 mecanum base firmware

The open-source omnidirectional base the Reachy Mini rides on. An ESP32 running
MicroPython drives four TT gearmotors on mecanum wheels through two L298N
H-bridge boards, and exposes everything over a tiny JSON HTTP API with a
hardware-independent deadman: if the controller goes away, the wheels stop.

It is deliberately hobby-grade — about the cheapest parts that work — and this
README is written so you can build one from scratch.

```
device/              everything that lives on the ESP32
  boot.py            pin lockdown, runs first on every boot
  main.py            builds `bot`, starts the server if AUTOSTART
  pins.py            SOURCE OF TRUTH: pin map, wheel map, invert, trim, speeds
  motors.py          Motor (one L298N channel) + MecanumDrive (kinematics)
  server.py          HTTP API + deadman watchdog + log ring buffer
  page.py            browser control page served at GET /
  wifi.py            AP / STA bring-up, AP fallback, ranging beacon
  selftest.py        bench routines: run(), primitives(), demo()
  netcfg.example.py  template for netcfg.py (WiFi, token, timeouts)
deploy.sh            push device/ over USB serial
deploy_wifi.sh       push device/ over HTTP, then soft-reset
logs.sh              fetch the board's in-memory log
teleop.py / .sh      keyboard driving, per-wheel diagnostics, calibration
```

## Hardware

| Part | Notes |
| --- | --- |
| ESP32 dev board | ESP32-D0WD-V3 class (classic dual-core ESP32, 4 MB flash), CP210x or CH340 USB-serial |
| 2× L298N dual H-bridge | One board per side; `ENA`/`ENB` jumpers **left on** |
| 4× TT gearmotors | The yellow 3–6 V hobby gearmotors |
| 4× mecanum wheels | Mounted in the standard **X configuration** (rollers form an X seen from above) |
| Battery pack | Dedicated to the base, separate from the Reachy's own power |

See [`../../hardware/BOM.md`](../../hardware/BOM.md) for the full parts list.

### How the motors are driven

Each L298N channel is driven **sign-magnitude on its IN pins**: both `IN`
pins are PWM outputs, direction comes from which pin carries the duty cycle and
speed from the duty itself. That is why `ENA`/`ENB` stay jumpered (always
enabled) and no extra enable wires are needed — 8 GPIOs drive 4 motors.

- `IN_A` PWM, `IN_B` low → one direction
- both low → coast (wheel free-spins)
- both high → brake (motor shorted, stops hard)

PWM runs at 1 kHz. TT motors buzz without turning below about 35% duty, so
`motors.py` maps a commanded speed `0..1` onto the duty band
`MIN_DUTY..1.0` (`MIN_DUTY = 0.35`): any non-zero speed actually moves the
wheel instead of sitting in a dead zone.

### Pin map

From [`device/pins.py`](device/pins.py) — that file is the single source of
truth; edit it, not this table, when you move a wire.

| Channel name | L298N inputs | GPIO (IN_A, IN_B) | Corner in shipped map | invert |
| --- | --- | --- | --- | --- |
| `DRIVER1_A` | driver 1, IN1 / IN2 | 32, 33 | `rear_left` | False |
| `DRIVER1_B` | driver 1, IN3 / IN4 | 18, 13 | `front_left` | False |
| `DRIVER2_A` | driver 2, IN1 / IN2 | 25, 26 | `rear_right` | True |
| `DRIVER2_B` | driver 2, IN3 / IN4 | 27, 19 | `front_right` | True |

The *corner* and *invert* columns are the `WHEELS` block in `pins.py`, which
`./teleop.sh --calibrate` regenerates for your build — which channel reaches
which corner, and which motors have swapped leads, depends on how you wired it
and cannot be known from a schematic. Run the calibration (below) rather than
trusting these values.

**Do not use GPIO 12 or GPIO 14 for motor inputs.** GPIO12 (MTDI) is a
strapping pin that selects flash voltage at reset: if a motor driver holds it
high while booting, the ESP32 picks 1.8 V for a 3.3 V flash chip and does not
boot. GPIO14 emits a pulse train during boot, which shows up as the wheels
"ghost driving" for a moment. The map above already avoids both.

### Power wiring

- Battery → both L298N `+12V` (motor supply) terminals and `GND`.
- L298N 5 V regulator jumper **on**; the drivers make their own logic 5 V.
- **ESP32 `GND` ↔ driver `GND` must always be connected.** It is the signal
  reference for the IN pins, not a supply.
- Bench (USB attached): power the ESP32 from USB and leave the driver's
  `+5V` → ESP32 `VIN` wire **disconnected**.
- Untethered: connect driver `+5V` → ESP32 `VIN` and unplug USB. **Never have
  both** USB and `VIN` powered at once.

Pack choice is up to you; remember the L298N drops roughly 1.5–2.5 V, so size
the pack for what your TT motors should see after that drop.

## Flashing MicroPython

The firmware targets **MicroPython 1.29.x, `ESP32_GENERIC` build**, from
<https://micropython.org/download/ESP32_GENERIC/>. You need
[esptool](https://github.com/espressif/esptool) and
[mpremote](https://docs.micropython.org/en/latest/reference/mpremote.html):

```bash
pip install esptool mpremote            # or: uv tool install esptool mpremote
```

Find the board's serial port (macOS: `ls /dev/cu.usbserial-*` or
`/dev/cu.wchusbserial-*`; Linux: `ls /dev/ttyUSB*`), then:

```bash
PORT=/dev/cu.usbserial-XXXX             # yours
esptool --port $PORT erase-flash
esptool --port $PORT --baud 460800 write-flash -z 0x1000 ESP32_GENERIC-<version>.bin
```

(Older esptool releases spell these `erase_flash` / `write_flash`, and may be
invoked as `esptool.py`.) The classic ESP32 image goes at offset `0x1000`.

## Configure and deploy

```bash
cp device/netcfg.example.py device/netcfg.py
$EDITOR device/netcfg.py                # WiFi mode, credentials, API token
./deploy.sh                             # USB; auto-detects a single serial port
./deploy.sh /dev/cu.usbserial-XXXX      # ...or name it
```

`device/netcfg.py` is git-ignored because it holds your WiFi password; both
deploy scripts refuse to run until it exists. `main.py`, `wifi.py` and
`server.py` import it as `netcfg`, so on the board it must be named exactly
`netcfg.py`.

Once the board is on the network and running the server (`AUTOSTART = True`),
later deploys can go over WiFi in about five seconds:

```bash
HOST=<board-ip> ./deploy_wifi.sh                # files via POST /put, then /reset
HOST=<board-ip> TOKEN=<your-token> ./deploy_wifi.sh
HOST=<board-ip> ./logs.sh                       # read the log ring buffer
```

Connecting over USB with `mpremote` (or `teleop.py` in USB mode) sends ctrl-C,
which **interrupts the running HTTP server** until the next reset. That is the
intended escape hatch — the board can never lock you out — but don't poke a
board over serial while you are also testing it over WiFi.

### Network modes

Set `MODE` in `netcfg.py`:

- **`"ap"`** (the example default) — the board hosts its own WPA2 network,
  `AP_SSID` / `AP_PASSWORD` (`mecanum-bot` / change the password!). Join it
  from your laptop and use the address the board prints on the serial console
  (`AP up: SSID='mecanum-bot'  ip=...`). Needs no infrastructure: ideal on the
  bench.
- **`"sta"`** — the board joins your existing network (`STA_SSID` /
  `STA_PASSWORD`) so the robot, the base and the station share one LAN. The
  ESP32 is **2.4 GHz only**; a 5/6 GHz-only network is invisible to it. If the
  join fails, the board **falls back to AP mode** rather than leaving you with
  no way in except a USB cable.

WiFi power save is disabled in STA mode (`pm=PM_NONE`): with it on, round-trip
latency was ~850 ms average and 1.7 s peak — fine for telemetry, useless for
driving. Off, it averages ~300 ms with ~10 ms best case, at some cost in
current.

### The `mecanum-beacon` ranging AP

With `RANGING_BEACON = True` in STA mode, the board additionally raises a
second, beaconing SSID (`RANGING_SSID`, default `mecanum-beacon`) while staying
joined to your router. It exists so **the robot can tell whether it is
physically mounted on the base**: the Reachy runs a passive WiFi scan, reads
this beacon's RSSI, and treats a very strong signal (centimetres away) as
"mounted". The mounted/unmounted thresholds are learned per installation, not
modelled — the absolute dBm depends on your enclosure and antenna placement.

Why a beacon at all: a WiFi *station* is invisible to a passive scan (it only
ever transmits to its access point), whereas an AP beacons on a fixed interval
to anyone listening. The ESP32 radio is single-channel, so the beacon sits on
whatever channel the STA link uses — scan that channel. This ESP32 has no
802.11mc FTM support, so true time-of-flight ranging isn't available; RSSI is
what there is. The beacon is started only after the STA join succeeds, uses
`AP_PASSWORD` as its key, accepts at most one client, and any failure is logged
and swallowed — a missing beacon must never take the drive server down.

`GET /rssi` reports the board's link strength **to the router**, which is a
different and much weaker signal about where the base is.

## HTTP API

Default port 80. All responses are JSON unless noted, with
`Access-Control-Allow-Origin: *` so browser front-ends can call it directly.
The board handles one connection at a time and refuses requests that arrive
while busy — **clients should retry**, which the shipped scripts do.

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Browser control page (press-and-hold D-pad, WASD/QE keys, speed slider) |
| `/state` | GET | Wheel speeds, last command, per-wheel tuning, deadman remaining |
| `/log` | GET | `{"lines": [...]}` — in-memory log ring buffer (last 200 lines, `"<ticks_ms> <msg>"`) |
| `/rssi` | GET | `{"rssi", "channel", "beacon_ssid", "t_ms"}` — router-link dBm, radio channel, beacon SSID (or null) |
| `/ls` | GET | `{"files": [{"name", "size"}]}` — files on the board |
| `/cmd` | POST | Drive command (below) |
| `/stop` | any | Stop immediately (coast) and disarm the deadman |
| `/reset` | POST | Soft-reset the board (applies files written with `/put`). Token-protected |
| `/put?path=NAME` | POST | Write a file; body = file contents, flat names only, max 64 KiB. Token-protected |
| `/tune` | POST | Change per-wheel invert/trim at runtime (lost on reset) |

### `GET /state`

```json
{"last_command": "forward",
 "wheels": {"front_left": 0.8, "front_right": 0.8, "rear_left": 0.8, "rear_right": 0.8},
 "moving": true,
 "tuning": {"front_left": {"invert": false, "trim": 1.0}, "...": {}},
 "stops_in": 1.4}
```

`stops_in` is seconds until the deadman fires, or `null` when idle. Wheel
values are the commanded speed (before trim and inversion).

### `POST /cmd`

Body is a JSON object with a `command` and optional `speed` (0..1, default
`pins.DEFAULT_SPEED = 0.8`) and `duration` (seconds, clamped to 0..30).
With a `duration`, the chassis stops at the end of it; without one it stops
after `COMMAND_TIMEOUT` (2 s) unless another command arrives first. Response:
`{"ok": true, "command": "...", "state": {...}}`, or HTTP 400 with
`{"ok": false, "error": "...", "state": {...}}`.

| `command` | Extra fields | Effect |
| --- | --- | --- |
| `forward`, `reverse` | `speed` | Drive straight |
| `strafe_left`, `strafe_right` | `speed` | Slide sideways (mecanum) |
| `rotate_ccw`, `rotate_cw` | `speed` | Spin in place |
| `diagonal_fl`, `diagonal_fr`, `diagonal_rl`, `diagonal_rr` | `speed` | 45° travel with two wheels idle |
| `move` | `vx`, `vy`, `omega`, `speed` | Any blend of translation and rotation (kinematics below) |
| `wheels` | `speeds: {corner: -1..1}`, `speed` | Explicit per-wheel mix, scaled by `speed`, **not** renormalised; unnamed wheels coast |
| `wheel` | `wheel: corner`, `speed` (signed) | One wheel alone — for identifying corners and polarity |
| `channel` | `a`, `b` (GPIO pair), `speed` | One L298N channel by its pins, bypassing the wheel map (calibration) |
| `stop` | — | Coast all wheels, disarm the deadman |
| `brake` | — | Short all motors (hard stop), disarm the deadman |

Corner names are `front_left`, `front_right`, `rear_left`, `rear_right`.

```bash
curl -X POST http://<board-ip>/cmd -d '{"command":"strafe_left","speed":0.6,"duration":1.5}'
curl -X POST http://<board-ip>/cmd -d '{"command":"move","vx":1,"vy":0.5,"omega":0,"speed":0.7}'
curl -X POST http://<board-ip>/cmd -d '{"command":"wheels","speeds":{"front_left":1,"rear_right":-0.6}}'
curl -X POST http://<board-ip>/stop
```

**Kinematics** (`MecanumDrive.move`, X configuration). Body frame: `+vx`
forward, `+vy` left, `+omega` counter-clockwise.

```
front_left  = vx - vy - omega        front_right = vx + vy + omega
rear_left   = vx + vy - omega        rear_right  = vx - vy + omega
```

If any wheel exceeds 1 the four are divided by the peak, so the fastest wheel
sits at `speed` and the rest stay proportional.

### `POST /tune`

`{"wheel": corner, ...}` plus any of `"flip": true` (toggle invert),
`"invert": bool`, `"trim": 0.3..1.5` (per-wheel output scale — evens out TT
motors that turn at different rates on the same duty), `"unbias": true`
(clear every wheel's invert flag, used by calibration). Returns the wheel's new
`invert`/`trim`. Runtime only: persist values by editing `pins.py` (teleop does
this for you) and redeploying.

### Authentication

`/put` and `/reset` honour `netcfg.API_TOKEN` via an `X-Token` header.
**An empty token (the example default) means `/put` and `/reset` are
unauthenticated**: anyone who can reach the board can overwrite any file on it
and reboot it. That is tolerable on the board's own private AP; on a shared LAN,
**set a token** and pass it as `TOKEN=...` to `deploy_wifi.sh` and `--token` to
`teleop.py`. The drive endpoints (`/cmd`, `/stop`, `/tune`) are never
authenticated — keep the base on a network you trust.

## Safety model

The rule: **no single failure — crash, dropped link, killed script, pulled
cable — leaves the wheels driving.**

1. **Boot lockdown.** `boot.py` drives all 8 motor inputs LOW before anything
   else runs, killing the twitch a floating H-bridge input produces at power-up.
   It is tiny and exception-wrapped on purpose: an error there would be a reset
   loop with the REPL hard to reach.
2. **HTTP deadman (2 s).** Every `/cmd` arms a deadline; a watchdog task checks
   it every 50 ms and coasts the wheels when it passes. Continuous control means
   re-sending the command faster than the timeout (the browser page re-sends
   every 300 ms while a button is held, and sends `stop` on release).
   Any request that raises inside the handler also stops the chassis.
3. **Teleop hardware deadman (900 ms).** Over USB, `teleop.py` arms an ESP32
   hardware `Timer` one-shot on every command whose interrupt zeroes all PWM
   outputs, so the wheels stop even if the host script dies or the cable is
   pulled. Over WiFi, teleop sends every command with a 0.7 s `duration`.
4. **Fail-safe `timed()`.** `MecanumDrive.timed()` stops in a `finally:`, so an
   exception mid-move can't leave the chassis driving off the bench.
5. **`main.py` never blocks** unless `AUTOSTART` is set, so the REPL stays
   reachable; with `AUTOSTART`, ctrl-C at the REPL stops the server and halts
   the wheels.
6. **Minimum duty.** Below `MIN_DUTY = 0.35` TT motors buzz and heat without
   turning; the speed rescale means the firmware never parks a motor there.

## Bench bring-up

Do this with the chassis **propped up so the wheels spin free**.

**1. Per-wheel check on the REPL** (`mpremote connect $PORT repl`):

```python
>>> import selftest
>>> selftest.run()          # each wheel forward then reverse, one at a time
>>> selftest.primitives()   # on the floor: every primitive once
>>> selftest.demo()         # on the floor: rotate, translate, announced
```

**2. Calibrate the wheel map with teleop** (needs `pyserial` for USB; the
`teleop.sh` wrapper uses [uv](https://docs.astral.sh/uv/) to fetch it
automatically, or `pip install pyserial` and run `python3 teleop.py`):

```bash
./teleop.sh --calibrate                 # guided: which corner did each channel spin?
./teleop.sh                             # keyboard driving + per-wheel keys
./teleop.sh --host <board-ip>           # same over WiFi, no cable
./teleop.sh --flip-front                # make the other end the front (180° reframe)
```

The guided calibration spins each L298N channel and asks which corner moved and
whether **the top of the wheel travelled toward the front or the back** —
answer that way, because "forward/backward" flips depending on which side of
the robot you stand on. It rewrites the `WHEELS` block in `pins.py` and offers
to deploy it.

Teleop keys: `1 2 3 4` drive one wheel forward (front-left, front-right,
rear-left, rear-right), shift+number backward; `f` flips the last wheel spun;
`[` / `]` trim it; `t` runs a narrated sweep; arrows / `w a s d` drive and
rotate; `z` / `c` strafe; `space` stops; `+` / `-` change speed; `q` quits and
saves any flips/trims to `pins.py`.

**3. Strafe is the test that matters.** Forward, reverse and both rotations
all look correct even when front and rear corners are swapped in the map. Only
strafing exposes it: check that `z` / `c` slide the chassis square. Also judge
by chassis motion, not by staring at the wheels — mecanum rollers spin against
the hub and a correctly driven wheel often *looks* like it is turning
backwards.

### Rotating in place under load

`rotate_cw` / `rotate_ccw` drive all four wheels at equal magnitude: the
textbook mix, and the worst case for grip, because every roller scrubs sideways
at once. With a top-heavy load like the Reachy Mini, whichever corner carries
least weight slips first and the pivot walks instead of spinning. The
`wheels` command exists to experiment with asymmetric mixes; the right numbers
are a property of your build's weight distribution and have to be measured.
`DEFAULT_SPEED = 0.8` was settled on the bench as the point where turns pivot
cleanly.
