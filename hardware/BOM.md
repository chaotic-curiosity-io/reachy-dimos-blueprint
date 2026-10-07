# Bill of materials

Every part is commercial off-the-shelf. The links are the exact listings we
bought from; equivalents work. Amazon listings change, so check that a
substitute matches the spec in the Part column.

## The robot

| Part | Role | Qty | Link |
| --- | --- | ---: | --- |
| Reachy Mini | The robot: head camera, neck, antennas, speaker/mic | 1 | [Pollen Robotics](https://www.pollen-robotics.com/) |

## The wheel base

| Part | Role | Qty | Link |
| --- | --- | ---: | --- |
| LewanSoul mecanum chassis kit — aluminum frame, 4× TT gearmotors, 4× 66 mm mecanum wheels (unassembled) | The whole mechanical base: frame, motors and wheels in one kit. Mount the wheels in the X configuration | 1 kit | [Amazon](https://www.amazon.com/dp/B093WDD9N5) |
| ESP32 dev board — ESP-WROOM-32 (classic dual-core ESP32), 4 MB flash, USB-serial onboard | Wireless motor control (HTTP API + deadman), runs MicroPython 1.29.x `ESP32_GENERIC` | 1 (sold as a 3-pack) | [Amazon](https://www.amazon.com/dp/B08D5ZD528) |
| L298N dual H-bridge driver board | One board per side, two motors each; `ENA`/`ENB` jumpers left on | 2 (sold as a 4-pack) | [Amazon](https://www.amazon.com/dp/B0C5JCF5RS) |
| 2S LiPo battery, 7.4 V 1200 mAh, JST plug, USB charger included | Dedicated base power: feeds both L298N motor-supply terminals; the drivers' 5 V regulator powers the ESP32 when untethered | 1 | [Amazon](https://www.amazon.com/dp/B0FB93HWMJ) |
| JST-RCY 2-pin silicone pigtails, 20 AWG (10 pairs) | Battery → driver power harness, and a quick-disconnect for the pack | 1 pack | [Amazon](https://www.amazon.com/dp/B071XN7C43) |
| Dupont jumpers, standoffs, fasteners | ESP32 GPIO → L298N IN pins, mounting | — | any |

## The depth add-on

| Part | Role | Qty | Link |
| --- | --- | ---: | --- |
| Intel RealSense L515 LiDAR camera — **discontinued**: buy used (common on the second-hand market); any L500-family camera works with librealsense ≤ v2.54.2 (L500 support was removed in v2.55.1) | Measured depth / point clouds | 1 | used market |
| Raspberry Pi 3 Model B+ | Reads the RealSense over USB, serves RGB-D to the station over WiFi | 1 | any |
| USB cable + mount | Attach the sensor to the Reachy | — | any |

## The station

Any reasonably capable machine that can run the depth models: we used both an
Apple-silicon Mac and an NVIDIA DGX Spark. Nothing here is required to be
NVIDIA-specific — the monocular pipeline runs on the Mac alone.

## Notes

- The base has its **own battery**, separate from the Reachy's power — the
  robot and the wheels only meet mechanically (the mounting plate) and over
  WiFi.
- **Battery voltage.** The L298N's motor terminal is silkscreened `+12V`, but
  that's its maximum, not a requirement: the 2S pack (7.4 V nominal, 8.4 V
  full) is what we run. The L298N drops roughly 2 V across its bridge, so the
  TT motors see about 5–6 V — inside their 3–6 V rating. As the pack sags
  toward 6.6 V, the drivers' onboard 5 V regulator stops regulating and the
  ESP32 can brown out; recharge before that, or power the ESP32 from USB
  while benching (and never connect USB *and* the 5 V jumper wire at once —
  see the [wiring blueprint](../docs/diagrams/wiring-blueprint.svg)).
- **Raspberry Pi model.** We run the depth server on a Pi 3 Model B+. Its
  ports are USB 2.0, which is enough for the server's default profile
  (depth at 640×480, published at up to 8 fps with `--stride 2`). The code
  has also run on a Pi 5; a Pi 4 or 5 on a USB 3 port gives more bandwidth
  headroom if you want higher frame rates or the colour stream alongside
  depth.
- Firmware, pin map and wiring for the base:
  [firmware/wheels/README.md](../firmware/wheels/README.md). Depth server and
  librealsense setup: [perception/depth_server/README.md](../perception/depth_server/README.md).
- Assembly order, wiring diagram, and the camera↔depth calibration procedure
  live in [ASSEMBLY.md](ASSEMBLY.md).
