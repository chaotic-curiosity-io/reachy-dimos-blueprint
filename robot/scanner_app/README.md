---
title: dimos_scanner
emoji: 🎥
sdk: docker
pinned: false
short_description: Stream Reachy Mini camera frames to a dimOS station bridge.
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# dimos_scanner

A Reachy Mini app that turns the robot into a remote camera for the
[dimOS](https://github.com/dimensionalOS/dimos) spatial-perception pipeline.
The robot streams JPEG-encoded camera frames (plus head pose, IMU and camera
intrinsics) over WebSocket to the bridge running on your *station* — the Mac
or DGX in [`../../station/`](../../station/) — and applies the head/body motion
commands the operator sends back via arrow keys.

This app passes `reachy-mini-app-assistant check` (all 9 SDK conformance
gates) and uses the standard `src/`-layout: one entry-point class in
`main.py`, with the substantive logic broken into `core/` + `io/`.

## Layout

```
robot/scanner_app/
├── pyproject.toml               src/-layout, optional [robot] [settings] [dev] [all]
├── README.md                    (this file — YAML frontmatter consumed by the SDK)
├── .env.example
├── src/dimos_scanner/
│   ├── main.py                  class DimosScanner(ReachyMiniApp) — SDK entry point
│   ├── cli.py                   `dimos-scanner` command (info / check / run)
│   ├── config.py                env-backed BridgeConfig dataclass
│   ├── core/
│   │   ├── scan_state.py        head/body pose tracker with body-yaw bleed
│   │   └── motion_loop.py       30 Hz set_target pump (background thread)
│   ├── io/
│   │   ├── protocol.py          wire codec (JSON control + binary frame messages)
│   │   ├── frame_encoder.py     BGR ndarray → JPEG bytes
│   │   └── bridge_client.py     async WS client (send frames, recv control)
│   └── static/index.html        settings page served at custom_app_url
├── examples/stub_run.py         rehearse the bridge handshake (no robot)
└── tests/                       pytest — runs without the Reachy SDK
```

`io/protocol.py` and `core/scan_state.py` are vendored copies of the
station-side modules in `../../station/`, so the robot never needs the station
code installed. Keep them in sync when you change the wire format.

## What makes this a Reachy Mini app

Three things, all in `pyproject.toml` and `main.py`:

```toml
keywords = ["reachy-mini-app"]

[project.entry-points."reachy_mini_apps"]
"dimos_scanner" = "dimos_scanner.main:DimosScanner"
```

The distribution name and the entry-point key must both use the underscore
form `dimos_scanner` (matching the package directory) — the Reachy daemon
looks up `custom_app_url` by that literal name, and a hyphenated name silently
breaks the embedded settings page. See the comments in `pyproject.toml`.

And:

```python
# src/dimos_scanner/main.py
class DimosScanner(ReachyMiniApp):
    custom_app_url = "http://0.0.0.0:8042"   # in-app settings UI

    def run(self, reachy_mini, stop_event):
        scan = ScanState()
        threading.Thread(
            target=run_motion_loop, args=(reachy_mini, scan, stop_event)
        ).start()
        client = BridgeClient(
            host=self.config.host,
            port=self.config.port,
            frame_producer=lambda: reachy_mini.media.get_frame(),
            scan=scan,
        )
        asyncio.run(client.run(stop_event))
```

The Reachy Mini daemon discovers this app via Python's
`importlib.metadata.entry_points(group="reachy_mini_apps")` after the
package is `pip install`-ed.

## Deploy to a Reachy Mini

From the **root of this repository**, using the blueprint's
[`scripts/deploy.sh`](../../scripts/):

```bash
# A. Local simulator (no physical robot). pip-installs into the env where
#    reachy_mini lives and starts `reachy-mini-daemon --sim`.
#    Open http://127.0.0.1:8000/.
./scripts/deploy.sh sim robot/scanner_app

# B. Real robot on the LAN. scp + pip into /venvs/apps_venv, then POSTs to
#    /api/apps/start-app/dimos_scanner on the robot's daemon.
#    reachy-mini.local is the stock hostname; pass an IP if mDNS is flaky.
./scripts/deploy.sh robot robot/scanner_app reachy-mini.local

# C. Publish to the Hugging Face community app store.
hf auth login                                           # one-time
reachy-mini-app-assistant publish robot/scanner_app
# Then on the robot dashboard → "Install community app" → paste the HF URL.
```

Under the hood `deploy.sh sim` is just:

```bash
pip install -e robot/scanner_app        # same env as reachy-mini-daemon
reachy-mini-daemon --sim
```

…and `deploy.sh robot` is (with `ROBOT=reachy-mini.local` or the robot's IP;
`pollen` is the stock robot user):

```bash
scp -r robot/scanner_app pollen@$ROBOT:/tmp/dimos_scanner
ssh pollen@$ROBOT "/venvs/apps_venv/bin/pip install /tmp/dimos_scanner"
curl -X POST http://$ROBOT:8000/api/apps/start-app/dimos_scanner
```

This is the same flow the SDK documents for any community app — see
[docs/SDK/apps.md upstream](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/SDK/apps.md).

> **Never** call the daemon's `/cache/reset-apps` endpoint to recover from a
> stuck app — it deletes the shared `/venvs/apps_venv` and wipes every
> installed app. If an app is stuck in "stopping", restart the daemon instead:
> `ssh pollen@$ROBOT 'sudo systemctl restart reachy-mini-daemon'`.

## Configure the bridge endpoint

The app needs to know your station's LAN IP. There is deliberately **no
default** — until you set it, the app idles and logs a reminder. Either set
`DIMOS_SCANNER_BRIDGE_HOST` (copy `.env.example` to `.env` for local runs), or
type it into the in-app settings page at `http://<robot>:8042/`. Values saved
on the settings page persist to `~/.config/dimos_scanner/config.json` on the
robot (override with `DIMOS_SCANNER_CONFIG`) and win over env defaults, so they
survive restarts and redeploys.

| env var | default | meaning |
|--|--|--|
| `DIMOS_SCANNER_BRIDGE_HOST` | `""` (idles until set) | LAN IP / hostname of your station (the machine running the bridge in `station/`) |
| `DIMOS_SCANNER_BRIDGE_PORT` | `9876` | bridge WS port |
| `DIMOS_SCANNER_JPEG_QUALITY` | `80` | 10..95 |
| `DIMOS_SCANNER_FRAME_HZ` | `5.0` | target send rate |
| `DIMOS_SCANNER_POSE_HZ` | `20.0` | head-pose stream rate |
| `DIMOS_SCANNER_IMU_HZ` / `DIMOS_SCANNER_IMU_ENABLED` | `50.0` / `1` | IMU stream (wireless Reachy Mini only) |
| `DIMOS_SCANNER_DEPTH_MODEL` | `da3metric-large` (`.env.example` sets `da3-small`) | depth-model preference sent to the station in the WS hello |
| `DIMOS_SCANNER_SETTINGS_URL` | `http://0.0.0.0:8042` | in-app settings page bind URL (`""` disables it) |
| `DIMOS_SCANNER_CONFIG` | `~/.config/dimos_scanner/config.json` | persisted settings file |

Find the station's IP with `ipconfig getifaddr en0` (macOS) or `hostname -I`
(Linux).

The settings page also stores station-side pipeline knobs (pose source,
device, display width, max FPS, CLIP memory, save-map, detection) as a single
source of truth; start the station pipeline with matching flags.

## Local development (no Reachy SDK)

```bash
cd robot/scanner_app
pip install -e '.[dev]'
pytest                                                  # unit tests, no robot
dimos-scanner info                                      # print resolved config
DIMOS_SCANNER_BRIDGE_HOST=localhost python examples/stub_run.py
```

`examples/stub_run.py` mocks the robot entirely — useful for debugging the
bridge handshake on a laptop with no robot online.

## Run end-to-end with dimOS

See [`../../station/`](../../station/) for the station-side workflow: start
the bridge + dimOS pipeline on the station, launch this app on the robot, set
the station host on the settings page, and drive with arrow keys while dimOS
builds the map in Rerun.

## Using this app as a template

1. `cp -r robot/scanner_app robot/<your_app>` (or
   `reachy-mini-app-assistant create <your_app>` for the SDK's blank template).
2. Rename the `src/<your_app>/` package, the `pyproject.toml` `name` +
   `[project.entry-points."reachy_mini_apps"]` key, the class in `main.py`,
   and the README's `title`.
3. Replace `BridgeClient` + `ScanState` with your own behaviour. The
   `core/` + `io/` split keeps testable pieces out of the SDK glue layer.
4. `./scripts/deploy.sh sim robot/<your_app>` to install it on a local sim.

## Tests

```bash
pip install -e '.[dev]'
pytest
```
