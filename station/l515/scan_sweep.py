"""Bounded, stop-and-observe wheel sweep for an authorized clear-floor test."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from urllib.request import urlopen
import numpy as np
from scipy.spatial.transform import Rotation
from reachy_wheels_app.wheels_client import WheelsClient,WheelsConfig
from station.l515.reachy_perception import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--command',choices=['rotate_cw','rotate_ccw','forward'],default='rotate_cw')
    p.add_argument('--speed',type=float,default=.15)
    p.add_argument('--duration',type=float,default=.06)
    p.add_argument('--trial-seconds',type=float,default=None,
                   help='Optional supervised command window, at most 10 seconds')
    p.add_argument('--max-travel',type=float,default=.12)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--wheels-host',default=os.environ.get('WHEELS_HOST',''),
                   help='ESP32 wheel base host (env WHEELS_HOST), e.g. <wheels-ip>')
    p.add_argument('--reachy-url',default=os.environ.get('REACHY_URL','http://reachy-mini.local:8042'),
                   help='wheels app base URL on the robot (env REACHY_URL)')
    a=p.parse_args()
    if not a.wheels_host:p.error('--wheels-host or WHEELS_HOST is required')
    app=a.reachy_url.rstrip('/')
    if not a.execute or not 1<=a.steps<=30:p.error('Explicit execution and 1–30 bounded steps required')
    if not (math.isfinite(a.speed) and 0<a.speed<=.25 and math.isfinite(a.duration) and .05<=a.duration<=.15):
        p.error('Speed must be <=0.25; duration 0.05–0.15 seconds')
    if not math.isfinite(a.max_travel) or not .03<=a.max_travel<=.35:
        p.error('Travel envelope must be between 3 and 35 cm')
    if a.trial_seconds is not None and not (math.isfinite(a.trial_seconds) and 0<a.trial_seconds<=10):
        p.error('Trial window must be positive and at most 10 seconds')
    out=a.directory/f'sweep-{time.time_ns()}.json'
    client=WheelsClient(WheelsConfig(host=a.wheels_host,timeout=5,retries=1))
    reader=WheelsClient(WheelsConfig(host=a.wheels_host,timeout=2,retries=3))
    def stop_verified():
        try:client.stop()
        except Exception:
            # A timed pulse may have expired despite a lost stop response.
            # Confirm stopped before allowing any further command.
            if reader.state().get('moving',True):raise RuntimeError('Stop not confirmed')
        if reader.state().get('moving',True):raise RuntimeError('Wheels did not stop')
        return {'verified_stopped':True}
    report=dict(command=a.command,speed=a.speed,duration=a.duration,steps=[],started=time.time())
    def read():
        s=json.loads((a.directory/'continuous-mapping.json').read_text())
        if (s.get('state')!='tracking' or not s.get('running') or
            not 0<=time.time()-s.get('source_received_at_unix',0)<=1.3):
            raise RuntimeError('Fresh continuous tracking required: '+str(s.get('reason')))
        if s.get('front_clearance_m') is None or s['front_clearance_m']<(.65 if a.command=='forward' else .35):
            raise RuntimeError('Insufficient observed clearance')
        return s
    try:
        with urlopen(app+'/api/track/status',timeout=2) as r:
            if json.load(r)['active']:raise RuntimeError('Following is active')
        with urlopen(app+'/api/voice/status',timeout=2) as r:
            if json.load(r)['voice_enabled']:raise RuntimeError('Voice control is active')
        first=read();report['before']=first
        trial_start=time.monotonic()
        trial_end=trial_start+a.trial_seconds if a.trial_seconds is not None else float('inf')
        report.update(trial_seconds=a.trial_seconds,max_travel_m=a.max_travel)
        for step in range(a.steps):
            if time.monotonic()+a.duration+.3>=trial_end:
                report['end_reason']='trial window complete';break
            before=read()
            if before['segment']!=first['segment']:raise RuntimeError('Map origin changed')
            travel=np.linalg.norm(np.array(before['sensor_pose_matrix'])[:3,3]-np.array(first['sensor_pose_matrix'])[:3,3])
            if travel+.03>a.max_travel:
                report['end_reason']='travel envelope reached';break
            if reader.state().get('moving',True):raise RuntimeError('Wheels already moving')
            before=read()
            if time.monotonic()+a.duration+.3>=trial_end:
                report['end_reason']='trial window complete';break
            event=dict(index=step,before=before);report['steps'].append(event)
            atomic_json(out,report)
            try:
                event['reply']=client.command(a.command,speed=a.speed,duration=a.duration)
                time.sleep(.25)
            finally:stop_verified()
            completed_at=time.time()
            deadline=time.monotonic()+12
            # Require three further accepted scans before the next pulse.
            while True:
                try:after=read()
                except RuntimeError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.2);continue
                scans,settle=(1,.15) if a.trial_seconds is not None else (3,.8)
                if after['accepted']>=before['accepted']+scans and after['source_received_at_unix']>completed_at+settle:break
                if time.monotonic()>deadline:raise RuntimeError('Map did not accept new scans')
                time.sleep(.2)
            if after['segment']!=first['segment']:raise RuntimeError('Map origin changed')
            event['after']=after
            delta=np.linalg.inv(np.array(before['sensor_pose_matrix']))@np.array(after['sensor_pose_matrix'])
            event['rotation_deg']=float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude()))
            event['translation_m']=float(np.linalg.norm(delta[:3,3]))
            atomic_json(out,report)
            print(json.dumps(dict(step=step,rotation_deg=event['rotation_deg'],translation_m=event['translation_m'],voxels=after['voxels'])),flush=True)
        report['completed']=True
        report['elapsed_with_observation_s']=time.monotonic()-trial_start
    except Exception as exc:
        report['error']=str(exc);print('Stopped: '+str(exc),flush=True)
    finally:
        try:report['stop']=stop_verified();report['final_wheels']=reader.state()
        except Exception as exc:report['stop_error']=str(exc)
        atomic_json(out,report);print(str(out),flush=True)

if __name__=='__main__':main()
