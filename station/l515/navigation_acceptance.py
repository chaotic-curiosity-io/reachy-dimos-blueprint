"""Live service acceptance test. Never enables motors; refuses an armed service.

Exercises the real HTTP command path and records whether mapped exploration can
produce a path. A safe rejection is not reported as successful navigation.
"""
import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def request(url, body=None):
    req = Request(url, data=None if body is None else json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=3) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--viewer-url', default='http://127.0.0.1:8777',
                        help='local l515_viewer / navigation HTTP service')
    parser.add_argument('--pi', default=os.environ.get('DEPTH_SERVER_URL', ''),
                        help='Pi depth streamer base URL (env DEPTH_SERVER_URL), e.g. http://<pi-ip>:8765')
    parser.add_argument('--wheels-host', default=os.environ.get('WHEELS_HOST', ''),
                        help='ESP32 wheel base host (env WHEELS_HOST), e.g. <wheels-ip>')
    args = parser.parse_args()
    if not args.pi or not args.wheels_host:
        parser.error('--pi/DEPTH_SERVER_URL and --wheels-host/WHEELS_HOST are required')
    base = args.viewer_url.rstrip('/')
    nav = lambda: request(base + '/api/navigation')
    command = lambda body: request(base + '/api/navigation/command', body)
    initial = nav()
    if initial.get('execution_enabled') is not False:
        raise SystemExit('Refusing acceptance test against an armed/unknown service')
    if initial.get('goal') or initial.get('exploring'):
        raise SystemExit('Refusing to replace an existing operator goal')
    report = {'started_at_unix': time.time(), 'initial_navigation': initial,
              'motor_execution_requested': False, 'observations': []}
    try:
        report['depth'] = request(args.pi.rstrip('/') + '/api/status')
        report['localization_before'] = request(base + '/api/localization')
        report['command_reply'] = command({'action': 'explore'})
        for _ in range(12):
            time.sleep(0.5)
            state = nav()
            report['observations'].append(state)
            if state.get('execution_enabled') is not False:
                raise RuntimeError('Execution mode changed during test')
        report['localization_after'] = request(base + '/api/localization')
        report['path_generated'] = any(bool(s.get('path')) for s in report['observations'])
        before, after = report['localization_before'], report['localization_after']
        report['mapping_advanced'] = (after.get('state') == 'tracking' and
            after.get('accepted', 0) > before.get('accepted', 0) and
            after.get('segment') == before.get('segment'))
        report['autonomous_navigation_passed'] = False
        report['physical_obstacle_test_passed'] = False
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        try:
            command({'action': 'stop'})
            for _ in range(10):
                time.sleep(0.2)
                report['final_navigation'] = nav()
                if report['final_navigation'].get('last_action') == 'operator stop':
                    break
            board = request(f'http://{args.wheels_host}/state')
            report['final_wheels'] = board
            report['stopped_verified'] = (board.get('moving') is False and
                bool(board.get('wheels')) and all(v == 0 for v in board['wheels'].values()))
        except Exception as exc:
            report['stop_verification_error'] = str(exc)
        output = args.directory / f'navigation-acceptance-{time.time_ns()}.json'
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k in (
            'error', 'mapping_advanced', 'path_generated', 'stopped_verified',
            'autonomous_navigation_passed', 'physical_obstacle_test_passed')}))
        print(output)


if __name__ == '__main__':
    main()
