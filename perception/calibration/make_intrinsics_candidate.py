"""Turn ``rgb-fit-report.json`` (from fit_rgb.py) into the intrinsics file fit_pair.py reads.

Picks the model with the lowest mean leave-one-view-out error (the same rule
fit_rgb.py prints) and writes ``reachy-rgb-intrinsics-candidate.json`` next to
the report (``$CALIBRATION_DIR`` or this folder).
"""
import json
import os
from pathlib import Path

import numpy as np

root = Path(os.environ.get('CALIBRATION_DIR') or Path(__file__).parent)
report = json.loads((root / 'rgb-fit-report.json').read_text())
models = report['models']
best = min(models, key=lambda n: np.mean(models[n]['leave_one_view_out_rms_pixels']))
m = models[best]
candidate = dict(
    camera='reachy_head_camera_optical', image_size=report['image_size'],
    model='opencv_brown_conrady', distortion_order=['k1', 'k2', 'p1', 'p2', 'k3'],
    K=m['K'], D=m['D'], views=report['views'], rms_pixels=m['rms_pixels'],
    mean_leave_one_view_out_rms_pixels=float(np.mean(m['leave_one_view_out_rms_pixels'])),
    square_length_m=report['square_length_m'], fit_model=best,
    scale_basis='provisional estimate from a photo of the displayed board',
    status='candidate_not_deployed',
    scope='RGB intrinsics only; no RGB-depth extrinsics, head kinematics, or metric validation')
(root / 'reachy-rgb-intrinsics-candidate.json').write_text(json.dumps(candidate, indent=2) + '\n')
print(f'{best}: rms {m["rms_pixels"]:.3f} px -> reachy-rgb-intrinsics-candidate.json')
