"""One explicit, low-speed wheel pulse while observing experimental L515 pose.

For a supervised clear-floor test only. This is not a navigation controller.
The --execute flag is required; motion is never retried and expires on-board.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
from scipy.spatial.transform import Rotation
from reachy_wheels_app.wheels_client import WheelsClient, WheelsConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--command', choices=['forward', 'reverse', 'strafe_left', 'strafe_right',
                                       'rotate_cw', 'rotate_ccw'], required=True)
    p.add_argument('--speed', type=float, default=0.2)
    p.add_argument('--duration', type=float, default=0.15)
    p.add_argument('--mapping-status', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reachy-url', default=os.environ.get('REACHY_URL', 'http://reachy-mini.local:8042'),
                   help='wheels app base URL on the robot (env REACHY_URL)')
    args = p.parse_args()
    if not args.execute:
        p.error('Use --execute only for an authorized supervised clear-floor test')
    if not (math.isfinite(args.speed) and 0 < args.speed <= 0.25
            and math.isfinite(args.duration) and 0.05 <= args.duration <= 0.25):
        p.error('Probe limits: speed (0,0.25], duration [0.05,0.25] seconds')
    def get(path):
        with urlopen(args.reachy_url + path, timeout=2) as r:
            return json.load(r)
    def mapping():
        value=json.loads(args.mapping_status.read_text())
        if (not value.get('running') or value['state'] != 'tracking'
                or not 0 <= time.time()-value['reported_at_unix'] < 1.5
                or value['source_age_s'] > 1.0):
            raise RuntimeError('L515 mapper is not tracking fresh scans')
        return value
    if get('/api/track/status')['active'] or get('/api/voice/status')['voice_enabled']:
        raise RuntimeError('Stop following and disable voice before a supervised wheel probe')
    state = get('/api/status')
    if not state.get('connected') or state['state']['moving']:
        raise RuntimeError('Chassis must be connected and initially stopped')
    before = mapping()
    client = WheelsClient(WheelsConfig(host=state['host'], port=state['port'],
                                       timeout=1.0, retries=1))
    report = {'command': args.command, 'speed': args.speed, 'duration_s': args.duration,
              'started_at_unix': time.time(), 'before': before, 'experimental': True}
    try:
        # One motion request only. An uncertain response is not retried.
        report['command_reply'] = client.command(args.command, speed=args.speed, duration=args.duration)
        time.sleep(args.duration + 0.15)
    except Exception as exc:
        report['command_error'] = str(exc)
    finally:
        try:
            report['stop_reply'] = client.stop()
        except Exception as exc:
            report['stop_error'] = str(exc)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    time.sleep(1.5)
    try:
        after = mapping()
        if before.get('map_frame') != after.get('map_frame'):
            raise RuntimeError('Map segment changed; before/after poses cannot be compared')
        report['after'] = after
        delta = np.linalg.inv(np.asarray(before['sensor_pose_matrix'])) @ np.asarray(after['sensor_pose_matrix'])
        report['estimated_sensor_translation_m'] = delta[:3, 3].tolist()
        report['estimated_sensor_distance_m'] = float(np.linalg.norm(delta[:3, 3]))
        report['estimated_sensor_rotation_deg'] = math.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())
        report['final_chassis'] = client.state()
    except Exception as exc:
        report['verification_error'] = str(exc)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'before','after'}}, indent=2))
    if (report.get('command_error') or report.get('stop_error') or report.get('verification_error')
            or report.get('final_chassis', {}).get('moving', True)):
        raise SystemExit('Probe incomplete: inspect report before any further movement')


if __name__ == '__main__':
    main()
