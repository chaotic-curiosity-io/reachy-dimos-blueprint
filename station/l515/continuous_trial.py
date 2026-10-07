"""Supervised continuous forward / obstacle-stop qualification, not a route planner."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from urllib.request import urlopen
import numpy as np
from station.l515.continuous_drive import ContinuousDrive, DriveObservation
from station.l515.persistent_rgbd import depth_points, transform_points
from reachy_wheels_app.wheels_client import WheelsClient, WheelsConfig


def get(url):
    with urlopen(url, timeout=1) as response:
        return json.load(response)


def corridor_clear(points, floor_z, base_xy, yaw, known_floor=None):
    """Current scan only; retain all obstacles and require floor across the lane.

    The near-body floor is operator-verified for this supervised qualification;
    it is not inserted into the autonomous costmap as sensed free space.
    """
    xy = points[:, :2] - base_xy
    c, s = np.cos(yaw), np.sin(yaw)
    xy = xy @ np.array([[c, -s], [s, c]])
    obstacles = (points[:, 2] >= floor_z + .04) & (points[:, 2] <= floor_z + 1.2)
    if np.any(obstacles & (xy[:, 0] > 0) & (xy[:, 0] < .85) & (abs(xy[:, 1]) < .32)):
        return False
    floor = xy[abs(points[:, 2] - floor_z) < .025]
    if known_floor is not None and len(known_floor):
        floor = np.vstack((floor, (known_floor - base_xy) @ np.array([[c, -s], [s, c]])))
    if len(floor) < 100:
        return False
    for x in np.linspace(.45, .95, 6):
        for y in np.linspace(-.23, .23, 5):
            if np.min(np.sum((floor - [x, y])**2, axis=1)) > .06**2:
                return False
    return True


def initial_floor_patch(base, yaw):
    """Operator-confirmed 80 cm lane, fixed in map coordinates once at start."""
    x, y = np.meshgrid(np.arange(0, .801, .04), np.arange(-.32, .321, .04))
    local = np.column_stack((x.ravel(), y.ravel()))
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--initial-floor-confirmed', action='store_true',
                        help='Operator has verified the initial 80 cm lane is clear level floor')
    parser.add_argument('--wheels-host', default=os.environ.get('WHEELS_HOST', ''),
                        help='ESP32 wheel base host (env WHEELS_HOST), e.g. <wheels-ip>')
    parser.add_argument('--reachy-url', default=os.environ.get('REACHY_URL', 'http://reachy-mini.local:8042'),
                        help='wheels app base URL on the robot (env REACHY_URL)')
    parser.add_argument('--viewer-url', default='http://127.0.0.1:8777',
                        help='local l515_viewer / navigation HTTP service')
    args = parser.parse_args()
    if not args.wheels_host:
        parser.error('--wheels-host or WHEELS_HOST is required')
    if not 0 < args.seconds <= 20:
        parser.error('seconds must be between 0 and 20')
    root = args.directory
    frame = json.loads((root/'navigation-frame.json').read_text())
    segment = frame['segment']
    level = np.array(frame['level_rotation'])
    basis = level @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    board = WheelsClient(WheelsConfig(host=args.wheels_host, timeout=.6, retries=1))
    app = args.reachy_url.rstrip('/')
    if get(app+'/api/track/status')['active'] or get(app+'/api/voice/status')['voice_enabled']:
        raise RuntimeError('another controller owns motion')
    if abs(get(app+'/api/motion')['body_yaw']) > 3:
        raise RuntimeError('body is not centered')
    state = board.state()
    if state.get('moving') is not False:
        raise RuntimeError('wheels must start stopped')
    expected = dict(front_left=False, rear_left=False, front_right=True, rear_right=True)
    if any(state['tuning'][key]['invert'] is not val for key, val in expected.items()):
        raise RuntimeError('wheel polarity changed')
    nav = get(args.viewer_url.rstrip('/')+'/api/navigation')
    if nav.get('execution_enabled') or nav.get('goal') or nav.get('exploring'):
        raise RuntimeError('navigation must be idle for qualification')
    command_file = root/'navigation-command.json'
    initial_command = command_file.stat().st_mtime_ns if command_file.exists() else 0
    drive = ContinuousDrive(board, segment, seconds=args.seconds)
    report = dict(execute=args.execute, started=time.time(), observations=[])
    floor_memory = {}
    start_base = None
    halt = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: halt.__setitem__(0, True))
    try:
        while time.monotonic() < drive.deadline and not halt[0]:
            if (command_file.stat().st_mtime_ns if command_file.exists() else 0) != initial_command:
                raise RuntimeError('operator command interrupted trial')
            mapping = json.loads((root/'continuous-mapping.json').read_text())
            with np.load(root/'mapping-frame.npz', allow_pickle=False) as data:
                handoff = json.loads(str(data['handoff']))
                if handoff['segment'] != segment or handoff['map_pose'] is None:
                    raise RuntimeError('accepted scan pose unavailable or changed segment')
                transform = np.array(handoff['map_pose'])
                points = transform_points(depth_points(data['depth'], handoff['meta']), transform) @ basis.T
                capture_time = time.time() - (time.monotonic() - handoff['depth_time'])
            rotation = basis @ transform[:3, :3] @ basis.T
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            base = (basis @ transform[:3, 3])[:2] - .151 * np.array([np.cos(yaw), np.sin(yaw)])
            if start_base is None:
                start_base = base.copy()
                if args.initial_floor_confirmed:
                    patch = initial_floor_patch(base, yaw)
                    for point in patch:
                        floor_memory[tuple(np.floor(point/.04).astype(int))] = point
                    report['operator_confirmed_floor'] = dict(base=base.tolist(), yaw=float(yaw),
                        forward_m=.8, half_width_m=.32, frame=segment)
            if np.linalg.norm(base-start_base) > 1.5:
                raise RuntimeError('1.5 m supervised travel limit')
            # Keep recently observed floor behind the sensor's near cutoff as
            # the robot advances. Current obstacle points always override it.
            floor = points[abs(points[:, 2]-frame['floor_z_m']) < .025, :2]
            keys = np.floor(floor/.04).astype(int)
            _, ids = np.unique(keys, axis=0, return_index=True)
            for i in ids:
                floor_memory[tuple(keys[i])] = floor[i]
            clear = corridor_clear(points, frame['floor_z_m'], base, yaw,
                                   np.array(list(floor_memory.values())).reshape(-1, 2))
            obs = DriveObservation(capture_time, mapping['segment'],
                mapping.get('state') == 'tracking' and mapping.get('running', False),
                float(mapping.get('front_clearance_m') or 0), clear)
            report['observations'].append(dict(time=time.time(), clear=clear,
                source_age=time.time()-capture_time, clearance=obs.clearance_m,
                sensor_pose=transform.tolist()))
            if args.execute:
                if not drive.tick(obs): break
            elif not clear:
                report['reason'] = 'preflight: floor corridor or obstacle clearance not satisfied'
                break
            time.sleep(.04)
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        if args.execute:
            drive.stop('trial ended')
            # Read-only verification may retry; movement commands never do.
            reader = WheelsClient(WheelsConfig(host=args.wheels_host, timeout=1, retries=3))
            try:
                report['final_wheels'] = reader.state()
            except Exception as exc:
                report['stop_verification_error'] = str(exc)
        report.update(reason=drive.reason or report.get('reason'), commands=drive.commands,
                      elapsed_s=time.time()-report['started'])
        out = root/f'continuous-trial-{time.time_ns()}.json'
        out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'observations'}), flush=True)
        print(out, flush=True)


if __name__ == '__main__':
    main()
