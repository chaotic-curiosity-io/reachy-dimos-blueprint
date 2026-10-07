"""Bounded RGB-follow trial with a frontal depth stop, NOT obstacle navigation.

The rotating sensor does not cover all wheel travel directions. A nearby human
operator is required. Mapping registration is deliberately not a motion gate.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from urllib.request import Request, urlopen

import numpy as np
from reachy_wheels_app.sensors import DepthSource, HEADER

# Defaults from the environment; override with --reachy-url / --wheels-host.
APP = os.environ.get('REACHY_URL', 'http://reachy-mini.local:8042')
BOARD = 'http://' + os.environ['WHEELS_HOST'] if os.environ.get('WHEELS_HOST') else ''


def request(url, data=None, timeout=3):
    payload = None if data is None else json.dumps(data).encode()
    with urlopen(Request(url, data=payload, headers={'Content-Type': 'application/json'}),
                 timeout=timeout) as response:
        return json.load(response)


def frontal_clearance(payload):
    xyz = np.frombuffer(payload, dtype='<f4', offset=HEADER.size).reshape(-1, 3)
    front = xyz[(abs(xyz[:, 0]) < .32) & (abs(xyz[:, 1]) < .25)]
    if len(front) < 100:
        raise ValueError('insufficient frontal depth coverage')
    # A handful of close returns must not disappear behind a percentile.
    if np.count_nonzero(front[:, 2] < .65) >= 3:
        return float(np.min(front[:, 2]))
    return float(np.quantile(front[:, 2], .05))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reachy-url', default=APP, help='wheels app base URL (env REACHY_URL)')
    parser.add_argument('--wheels-host', default=os.environ.get('WHEELS_HOST', ''),
                        help='ESP32 wheel base host (env WHEELS_HOST), e.g. <wheels-ip>')
    parser.add_argument('--depth-url', default=os.environ.get('DEPTH_SERVER_URL', ''),
                        help='Pi depth streamer base URL (env DEPTH_SERVER_URL)')
    parser.add_argument('--viewer-url', default='http://127.0.0.1:8777',
                        help='local l515_viewer / navigation HTTP service')
    args = parser.parse_args()
    if not 5 <= args.seconds <= 60:
        parser.error('seconds must be 5–60')
    if not args.wheels_host or not args.depth_url:
        parser.error('--wheels-host/WHEELS_HOST and --depth-url/DEPTH_SERVER_URL are required')
    app = args.reachy_url.rstrip('/')
    board = 'http://' + args.wheels_host
    changes = dict(track_max_session_seconds=args.seconds, track_drive_speed=.2,
                   track_rotate_speed=.3, track_search_seconds=args.seconds,
                   track_acquire_seconds=args.seconds, track_search_wheels_after=6,
                   track_body_assist_deg=60)
    config = request(app+'/api/config')
    previous = {k: config.get(k, 12) for k in changes}
    if request(app+'/api/track/status').get('active'):
        raise RuntimeError('another follow session is active')
    if request(app+'/api/voice/status')['voice_enabled']:
        raise RuntimeError('voice motion owner is enabled')
    nav = request(args.viewer_url.rstrip('/')+'/api/navigation')
    if nav.get('execution_enabled') or nav.get('goal') or nav.get('exploring'):
        raise RuntimeError('DimOS motion owner is active')
    if request(board+'/state')['moving'] is not False:
        raise RuntimeError('wheels must start stopped')
    source = DepthSource(url=args.depth_url, timeout=.7, max_age_ms=800)
    payload, meta = source.frame()
    clearance = frontal_clearance(payload)
    if clearance < .65:
        raise RuntimeError(f'frontal proximity preflight: {clearance:.3f}m')
    if not args.execute:
        print(json.dumps(dict(preflight='passed', clearance_m=clearance, depth=meta)))
        return
    report = dict(started=time.time(), requested_seconds=args.seconds, samples=[])
    halt = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: halt.__setitem__(0, True))
    try:
        report['config'] = request(app+'/api/track/config', changes)
        report['start'] = request(app+'/api/track/start',
                                  dict(target='person', distance_cm=200), timeout=5)
        if report['start'].get('status') != 'ok':
            raise RuntimeError(str(report['start']))
        print('Following person for up to 60 seconds', flush=True)
        deadline = time.monotonic()+args.seconds
        last_status = 0
        while time.monotonic() < deadline and not halt[0]:
            payload, meta = source.frame()
            clearance = frontal_clearance(payload)
            if clearance < .65:
                raise RuntimeError(f'frontal proximity stop: {clearance:.3f}m')
            if time.monotonic()-last_status >= 1:
                status = request(app+'/api/track/status', timeout=1)
                sample = dict(time=time.time(), clearance_m=clearance,
                              depth_age_ms=meta['age_upper_bound_ms'], status=status)
                report['samples'].append(sample)
                print(json.dumps({k:status.get(k) for k in
                      ('active','phase','moving','seen','detail')}), flush=True)
                last_status = time.monotonic()
                if not status.get('active'):
                    report['reason'] = 'follower ended: '+status.get('detail', '')
                    break
            time.sleep(.05)
        report.setdefault('reason', 'interrupted' if halt[0] else 'time limit')
    except Exception as exc:
        report['reason'] = str(exc)
    finally:
        for name, url in [('follow_stop', app+'/api/track/stop'),
                          ('board_stop', board+'/cmd')]:
            try:
                report[name] = request(url, {'command':'stop'} if name=='board_stop' else {}, timeout=6)
            except Exception as exc:
                report[name+'_error'] = str(exc)
        try:
            report['final_follow'] = request(app+'/api/track/status')
            report['final_wheels'] = request(board+'/state')
            if not report['final_follow'].get('active'):
                report['restored'] = request(app+'/api/track/config', previous)
        except Exception as exc:
            report['verification_error'] = str(exc)
        report['elapsed_s'] = time.time()-report['started']
        path = args.directory/f'rgb-follow-trial-{time.time_ns()}.json'
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('samples','final_follow')}), flush=True)
        print(path, flush=True)


if __name__ == '__main__':
    main()
