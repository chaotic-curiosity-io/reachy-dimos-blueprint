"""Head-camera frame constants for the Reachy Mini (stdlib + numpy leaf module).

Conventions:

* **optical** — OpenCV camera axes: X-right, Y-down, Z-forward.
* **body / FLU** — robot body: X-forward, Y-left, Z-up.

``T_HEAD_CAM`` here is the extrinsic the *mono* path (``server.py``) composes
onto the streamed head pose. The L515 path (``station/l515``) instead derives
head->camera from the official MJCF sites (~39.5 mm forward / 52.5 mm up); see
``station/l515/ARTICULATED_RGBD.md`` for why the two differ.
"""

from __future__ import annotations

import numpy as np

# Camera-optical -> head-body FLU axis map (rotation only): optical +Z (view)
# is body +X (forward), optical +X (right) is body -Y, optical +Y (down) is
# body -Z.
OPT_TO_BODY = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Reachy Mini head->camera extrinsic: the OPT_TO_BODY rotation plus the camera
# lever arm — the optical center sits 43.7 mm forward / 51.2 mm above the head
# frame origin. Without the rotation a head yaw (about body Z) is applied as a
# ROLL about the optical axis and the cloud fans into overlapping copies;
# without the translation every rotation swings the real camera on an arc the
# math ignores and nearby objects smear by centimetres per viewpoint.
T_HEAD_CAM = OPT_TO_BODY.copy()
T_HEAD_CAM[:3, 3] = [0.0437, 0.0, 0.0512]
