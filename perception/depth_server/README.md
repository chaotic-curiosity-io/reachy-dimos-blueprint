# Depth server — Intel RealSense L515 on a Raspberry Pi

The depth add-on's software. An Intel RealSense **L515** LiDAR camera is
USB-attached to a Raspberry Pi riding on the Reachy; this small Python service
runs on the Pi, captures depth (and optionally RGB) there, and serves it
**pull-based over HTTP**: clients ask for the latest frame when they want one,
nothing is pushed. The station fetches point clouds and calibration bundles from
it; a built-in WebGL page lets you eyeball the point cloud from any browser on
the LAN, with no CDN or internet access needed.

```
realsense_viewer/
  server.py          capture thread + HTTP server (numpy + pyrealsense2 only)
  static/            the browser point-cloud viewer (index.html, app.js, styles.css)
tests/               offline unit tests (no camera needed)
examples/
  l515-smoke.cpp     30-frame C++ smoke test against librealsense
deploy/
  realsense-viewer.service.template   systemd unit
  calibration.conf.template           drop-in that enables --color
pyproject.toml       installable as `realsense-browser-viewer`
```

## Try it with zero hardware

`--synthetic` replaces the camera with an animated test surface that exercises
the whole HTTP and browser path. Only `numpy` is needed:

```bash
pip install numpy
python -m realsense_viewer.server --synthetic --host 127.0.0.1
# open http://127.0.0.1:8765/
```

Run the tests (also hardware-free):

```bash
pip install numpy pytest
python -m pytest perception/depth_server/tests -q
```

## Run it

```bash
python -m realsense_viewer.server [flags]
# or, after `pip install .` in this folder:
realsense-browser-viewer [flags]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Interface to bind |
| `--port` | `8765` | HTTP port |
| `--stride` | `2` | Subsample the depth image by this step (1–8) before building points |
| `--max-fps` | `8` | Cap on published point-cloud frames per second (capture still runs at 30) |
| `--min-depth` | `0.15` | Discard points nearer than this (m) |
| `--max-depth` | `4.0` | Discard points farther than this (m) |
| `--color` | off | Also capture RGB (640×480 RGB8 @ 6 Hz) and enable the calibration endpoints |
| `--synthetic` | off | No camera: serve a generated test surface |
| `--verbose` | off | Log every HTTP request |

Depth is captured at 640×480 Z16 @ 30 Hz. (On firmware 1.5.2 the L515
advertises a 320×240 mode that, on a Pi 5, never delivers frames; the native
640×480 profile is stable.) If the camera unplugs or errors, the capture thread
reports `reconnecting` and retries every 2 s.

## HTTP API

| Route | Returns |
| --- | --- |
| `GET /` | WebGL point-cloud viewer (drag to orbit, scroll to zoom, point size, pause, auto-rotate, live FPS/latency) |
| `GET /healthz` | `{"ok", "state"}` — 200 while streaming with a frame newer than 3 s, else 503 |
| `GET /api/status` | Camera and stream status JSON (below) |
| `GET /api/points` | Latest point cloud, RSPC binary (below); 503 before the first frame |
| `GET /api/color.bmp` | Latest RGB frame as an uncompressed 24-bit BMP (`--color` only) |
| `GET /api/calibration-frame` | ZIP: `color.bmp` + `depth.u16` + `metadata.json` from one frameset (`--color` only) |
| `GET /api/mapping-frame` | Same ZIP with color downsampled 4× (`color_pixel_stride: 4`); depth stays full resolution |
| `GET /api/rgbd-frame` | ZIP with `depth.u16` + `metadata.json` only (no color image) |

The color/calibration endpoints answer **503** when `--color` is off or the
newest depth+color pair is older than **two seconds** — you never get a stale
pair silently. All API responses carry `Cache-Control: no-store`.

### `/api/status`

```json
{"state": "streaming", "camera": "Intel RealSense L515", "serial": "<camera serial>",
 "firmware": "1.5.2.0", "resolution": "640x480@30", "depth_scale_m": 0.00025,
 "point_count": 41234, "sequence": 1803, "capture_fps": 30.0, "published_fps": 8.0,
 "last_frame_age_ms": 42.1, "error": null}
```

`state` is one of `starting`, `warming_up`, `streaming`, `reconnecting`.
(The values above are illustrative.)

### `/api/points` — the RSPC binary format

All little-endian. A 16-byte header followed by the points:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `char[4]` | magic `RSPC` |
| 4 | `uint32` | format version (`1`) |
| 8 | `uint32` | frame sequence number (wraps at 2³²) |
| 12 | `uint32` | point count `N` |
| 16 | `float32[N][3]` | `x, y, z` per point, **in metres** |

Points are in the depth camera's optical frame (`+x` right, `+y` down, `+z`
forward), deprojected with the factory depth intrinsics from the subsampled
depth image; invalid and out-of-range pixels are dropped, so `N` varies per
frame. Decoding in Python:

```python
import struct, urllib.request
import numpy as np

buf = urllib.request.urlopen("http://<pi-ip>:8765/api/points").read()
magic, version, seq, n = struct.unpack_from("<4sIII", buf)
assert magic == b"RSPC" and version == 1
xyz = np.frombuffer(buf, dtype="<f4", offset=16).reshape(n, 3)   # metres
```

### Calibration bundle (`/api/calibration-frame`)

An uncompressed (`ZIP_STORED`, so the Pi doesn't spend time on DEFLATE) ZIP
whose files all come from **one captured frameset**:

- **`color.bmp`** — 640×480 24-bit BMP.
- **`depth.u16`** — the raw, full-resolution depth image: `height × width`
  little-endian `uint16`, row-major, no header. Multiply by `depth_scale_m` for
  metres; 0 means no return.
- **`metadata.json`** — everything needed to put the two images in one frame:

| Key | Content |
| --- | --- |
| `depth_intrinsics`, `color_intrinsics` | Factory intrinsics: `width`, `height`, `fx`, `fy`, `ppx`, `ppy`, distortion `model`, `coeffs` |
| `depth_to_color` | Factory extrinsics: `rotation_column_major` (9 floats), `translation_m` (3 floats, metres) |
| `depth_scale_m` | Metres per depth unit |
| `serial`, `session_id` | Camera serial; UUID of this capture session (changes on reconnect) |
| `color_frame_number`, `depth_frame_number` | SDK frame counters |
| `color_timestamp_ms`, `depth_timestamp_ms`, `*_timestamp_domain` | SDK timestamps and their clock domain |
| `received_at_unix` | Pi wall-clock time the frameset arrived |
| `color_format`, `depth_format` | `rgb8`, `uint16_le` |
| `synchronized_with_reachy` | Always `false` — see below |
| `server_frame_age_ms` | Age of the pair when it was served |

The bundle is internally consistent (color and depth from the same L515
frameset) but makes **no claim of synchronisation with the Reachy's head
camera**. Pairing an L515 frame with a head-camera frame, and solving the
head-camera ↔ L515 extrinsic, is done downstream on the station; this service
only supplies the L515's own factory calibration and honest timestamps.

## On the Pi: librealsense compatibility

This is the part that bites. **The L500 family (L515) needs librealsense
2.48.0.** Intel discontinued the L515 and later librealsense releases removed
the L500-family code. Some intermediate 2.5x builds still enumerate the camera,
but results vary by build and firmware — one 2.54.x build we tried rejected the
camera's 1.5.2.0 firmware during initialisation. 2.48.0 supports both the L515
and that firmware, so pin it.

- **librealsense:** v2.48.0 from upstream,
  <https://github.com/IntelRealSense/librealsense/tree/v2.48.0> — build it
  yourself; it is not vendored here.
- **Backend:** build with the **RSUSB** (userspace libusb) backend,
  `-DFORCE_RSUSB_BACKEND=ON`, which avoids kernel patches on Raspberry Pi OS.
- **Python:** build the bindings (`-DBUILD_PYTHON_BINDINGS=ON
  -DPYTHON_EXECUTABLE=<venv>/bin/python`) and make `pyrealsense2` importable
  from the venv that runs the server (copy or symlink the built
  `pyrealsense2*.so` into its `site-packages`). PyPI's prebuilt `pyrealsense2`
  wheels don't cover the Pi's ARM build, so build them alongside the library.
- **udev rules:** install librealsense's `config/99-realsense-libusb.rules`
  (or run `scripts/setup_udev_rules.sh`) so a normal user can open the camera.
- **Camera firmware: 1.5.2.0 is known-good** with this stack. The RealSense
  Viewer may nag you to update (it recommends 1.5.8.1); we never did. Don't
  accept a firmware flash casually — it needs stable power and the exact L515
  image, and changes which SDK versions accept the camera.
- **USB:** use a USB 3 port (the blue ones on a Pi 4/5). The `--color` profile
  (640×480 @ 6 Hz) is the one the L515 also supports over USB 2.

A typical source build, abbreviated:

```bash
git clone --branch v2.48.0 --depth 1 https://github.com/IntelRealSense/librealsense.git
cd librealsense && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON \
         -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$HOME/.venvs/realsense-l515-rsusb/bin/python \
         -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
         -DCMAKE_INSTALL_PREFIX=$HOME/.local/realsense-l515
make -j"$(nproc)" && make install
```

A 2021 release on a current compiler/Python can need small fixes; check the
upstream issue tracker for your OS version. If you install into a private
prefix like the one above, point `LD_LIBRARY_PATH` at its `lib/` (the service
template does).

### Smoke-testing the SDK

[`examples/l515-smoke.cpp`](examples/l515-smoke.cpp) opens the first device,
prints name/serial/firmware, captures 30 depth frames and prints depth
statistics — the quickest way to prove the SDK and camera talk before involving
Python:

```bash
c++ -std=c++14 -O2 examples/l515-smoke.cpp -o l515-smoke $(pkg-config --cflags --libs realsense2)
./l515-smoke
```

(With a private install prefix, add its `lib/pkgconfig` to `PKG_CONFIG_PATH`.)

## Run it as a service

Copy this folder to the Pi (the template assumes `~/apps/realsense-viewer`),
then fill the `{{USER}}` / `{{HOME}}` placeholders and install the unit:

```bash
cd ~/apps/realsense-viewer
sed -e "s|{{USER}}|$USER|g" -e "s|{{HOME}}|$HOME|g" \
    deploy/realsense-viewer.service.template \
  | sudo tee /etc/systemd/system/realsense-viewer.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now realsense-viewer.service
curl -s http://localhost:8765/healthz
```

Edit the venv path or drop the `LD_LIBRARY_PATH` line in the template if your
setup differs.

### Calibration mode: the systemd drop-in pattern

Paired depth+RGB capture costs USB bandwidth and Pi CPU, so it is off by
default. Rather than editing the base unit, turn it on with a **drop-in** that
replaces `ExecStart` with the same command plus `--color`:

```bash
sudo mkdir -p /etc/systemd/system/realsense-viewer.service.d
sed -e "s|{{HOME}}|$HOME|g" deploy/calibration.conf.template \
  | sudo tee /etc/systemd/system/realsense-viewer.service.d/calibration.conf >/dev/null
sudo systemctl daemon-reload && sudo systemctl restart realsense-viewer.service
```

The empty `ExecStart=` line in the drop-in is required — it clears the base
command before setting the new one. With it active, depth and RGB publish
together at roughly 6 Hz. To go back to depth-only, delete that one
`calibration.conf`, `daemon-reload`, and restart; the base unit is untouched.
