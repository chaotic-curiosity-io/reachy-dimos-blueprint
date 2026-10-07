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
| ESP32 dev board | Wireless motor control (HTTP API + deadman) | 1 | TBD | TBD |
| Motor driver boards | H-bridge drive for the gearmotors | TBD | TBD | TBD |
| DC gearmotors | One per wheel | 4 | TBD | TBD |
| Mecanum wheels | Omnidirectional motion | 4 | TBD | TBD |
| Chassis / mounting plate | Carries the base and the Reachy | 1 | TBD | TBD |
| Battery pack (base) | Dedicated power for motors + ESP32 | 1 | TBD | TBD |
| Wiring, standoffs, fasteners | Assembly | — | TBD | TBD |

## The depth add-on

| Part | Role | Qty | Link | ~Price |
| --- | --- | ---: | --- | --- |
| Intel RealSense LiDAR camera | Measured depth / point clouds | 1 | TBD | TBD |
| Raspberry Pi | Reads the RealSense, streams RGB-D to the station | 1 | TBD | TBD |
| USB cable + mount | Attach the sensor to the Reachy | — | TBD | TBD |

## The station

Any reasonably capable machine that can run the depth models: we used both an
Apple-silicon Mac and an NVIDIA DGX Spark. Nothing here is required to be
NVIDIA-specific — the monocular pipeline runs on the Mac alone.

## Notes

- The base has its **own battery**, separate from the Reachy's power — the
  robot and the wheels only meet mechanically (the mounting plate) and over
  WiFi.
- Assembly order, wiring diagram, and the camera↔depth calibration procedure
  live in [ASSEMBLY.md](ASSEMBLY.md).
