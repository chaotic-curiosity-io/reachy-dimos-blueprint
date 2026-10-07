# Official Reachy Mini MJCF (Pollen Robotics)

`reachy_mini.xml` is the official Reachy Mini MuJoCo model, copyright 2025
Pollen Robotics, redistributed unmodified under the Apache License 2.0 (see
`LICENSE` in this folder, copied verbatim from upstream).

| | |
| --- | --- |
| Upstream | <https://github.com/pollen-robotics/reachy_mini> |
| Commit | `bae7844f36cd98358a35d965aa0e19e8419cb680` |
| Path upstream | `src/reachy_mini/descriptions/reachy_mini/mjcf/reachy_mini.xml` |
| SHA-256 (`reachy_mini.xml`) | `efd7e49d4288e5ef53945771a1f116584aa2c8b89721b725d5d77da9f0fcbf46` |
| SHA-256 (`LICENSE`) | `1cd66cd7ef50bb247f43d247f7c2a3259a600d951170e778d75c9278e5680f03` |

## What uses it

Only `station/l515/articulated_rgbd.py::model_head_T_camera()`, called from
`station/l515/reachy_rgbd.py`. It parses the XML with `ElementTree` and reads
two sites on the head link, `head` and `camera_optical`, to get the
head -> optical-camera transform (about 39.5 mm forward, 52.5 mm up, plus the
optical-axis rotation). The SHA-256 of the file is recorded in every RGB-D
observation as the geometry revision.

Override the location with `--mjcf` or `REACHY_MJCF=/path/to/reachy_mini.xml`.

## Meshes are not included

The XML references ~90 STL meshes (`meshdir="assets"`, about 20 MB in total).
The site lookup above never opens them, so they aren't vendored here. To load
the model in MuJoCo (simulation, visual checks), fetch the matching meshes from
the same upstream commit:

```sh
git clone https://github.com/pollen-robotics/reachy_mini.git /tmp/reachy_mini
git -C /tmp/reachy_mini checkout bae7844f36cd98358a35d965aa0e19e8419cb680
cp -r /tmp/reachy_mini/src/reachy_mini/descriptions/reachy_mini/mjcf/assets \
      station/assets/official_reachy/assets
```

Without the meshes, `mujoco.MjModel.from_xml_path()` fails. Parsing the head and
camera sites still works.
