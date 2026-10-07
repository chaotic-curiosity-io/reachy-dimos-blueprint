"""Capture one ChArUco view from the Reachy head camera for the intrinsics fit.

Writes ``<directory>/<UTC stamp>.jpg`` and a matching ``.json`` holding the
detected corners, in exactly the schema ``fit_rgb.py`` reads (it globs
``rgb-ipad/*.json`` next to itself, or under ``$CALIBRATION_DIR``). Only reads
the camera; it sends no motion commands.

Hold the board still, run once per view, and vary position (centre, all four
corners and edges of the image), distance and tilt between views. About 13
views gave a stable fit for us.

    python perception/calibration/capture_rgb_view.py --label top_left
"""
import argparse
import datetime
import json
import os
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

SQUARES = (5, 7)
SQUARE_M = .0235
MARKER_M = .01175


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reachy', default=os.environ.get('REACHY_URL', 'http://reachy-mini.local:8042'),
                   help='wheels app base URL serving /api/camera (env REACHY_URL)')
    p.add_argument('--directory', type=Path,
                   default=Path(os.environ.get('CALIBRATION_DIR') or Path(__file__).parent) / 'rgb-ipad')
    p.add_argument('--label', default='view', help='free-text pose label stored with the view')
    p.add_argument('--min-corners', type=int, default=12)
    args = p.parse_args()
    with urlopen(args.reachy.rstrip('/') + '/api/camera', timeout=8) as r:
        data = r.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise SystemExit('Oversized camera response')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit('Camera returned an invalid image')
    board = cv2.aruco.CharucoBoard(SQUARES, SQUARE_M, MARKER_M,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if ids is None or len(ids) < args.min_corners:
        raise SystemExit(f'Only {0 if ids is None else len(ids)} ChArUco corners; need {args.min_corners}')
    args.directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (args.directory / f'{stamp}.jpg').write_bytes(data)
    record = dict(image_size=[image.shape[1], image.shape[0]], board_squares=list(SQUARES),
                  dictionary='DICT_5X5_100', square_length_m=SQUARE_M, marker_length_m=MARKER_M,
                  scale_basis='estimated from a photo of the displayed board, not measured',
                  corner_ids=ids.flatten().tolist(), corners=corners.reshape(-1, 2).tolist(),
                  pose_label=args.label, capture_timestamp_known=False)
    (args.directory / f'{stamp}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(saved=str(args.directory / stamp), corners=len(ids))))


if __name__ == '__main__':
    main()
