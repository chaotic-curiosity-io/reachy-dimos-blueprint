# Bill of materials

Every part is commercial off-the-shelf. Links and prices to be filled in —
the table below is the working inventory; `TBD` cells need the exact SKU we
bought.

## The robot

| Part | Role | Qty | Link | ~Price |
| --- | --- | ---: | --- | --- |
| Reachy Mini | The robot: head camera, neck, antennas, speaker/mic | 1 | [Pollen Robotics](https://www.pollen-robotics.com/) | TBD |

## The wheel base

| Part | Role | Qty | Link | ~Price |
| --- | --- | ---: | --- | --- |
| ESP32 dev board — ESP32-D0WD-V3 (classic dual-core ESP32), 4 MB flash, USB-serial onboard | Wireless motor control (HTTP API + deadman), runs MicroPython 1.29.x `ESP32_GENERIC` | 1 | TBD | TBD |
| L298N dual H-bridge driver board | H-bridge drive for the gearmotors — one board per side, two motors each; `ENA`/`ENB` jumpers left on | 2 | TBD | TBD |
| TT gearmotor (yellow 3–6 V hobby gearmotor) | One per wheel | 4 | TBD | TBD |
| Mecanum wheel (TT-shaft compatible), 2 left-hand + 2 right-hand | Omnidirectional motion, mounted in the X configuration | 4 | TBD | TBD |
| Chassis / mounting plate | Carries the base and the Reachy | 1 | TBD | TBD |
| Battery pack (base), dedicated | Powers the L298N motor supply; the drivers' 5 V regulator feeds the ESP32 when untethered | 1 | TBD | TBD |
| Wiring, standoffs, fasteners | Assembly | — | TBD | TBD |

## The depth add-on

| Part | Role | Qty | Link | ~Price |
| --- | --- | ---: | --- | --- |
| Intel RealSense L515 LiDAR camera — **discontinued**: buy used (common on the second-hand market); any L500-family camera that works with librealsense 2.48 will do | Measured depth / point clouds | 1 | TBD | TBD |
| Raspberry Pi 4 or Raspberry Pi 5 (we've deployed both) | Reads the RealSense over USB 3, serves RGB-D to the station | 1 | TBD | TBD |
| USB cable + mount | Attach the sensor to the Reachy | — | TBD | TBD |

## The station

Any reasonably capable machine that can run the depth models: we used both an
Apple-silicon Mac and an NVIDIA DGX Spark. Nothing here is required to be
NVIDIA-specific — the monocular pipeline runs on the Mac alone.

## Notes

- The base has its **own battery**, separate from the Reachy's power — the
  robot and the wheels only meet mechanically (the mounting plate) and over
  WiFi.
- Firmware, pin map and wiring for the base:
  [firmware/wheels/README.md](../firmware/wheels/README.md). Depth server and
  librealsense setup: [perception/depth_server/README.md](../perception/depth_server/README.md).
- Assembly order, wiring diagram, and the camera↔depth calibration procedure
  live in [ASSEMBLY.md](ASSEMBLY.md).
