"""Supervise mapping, perception, DimOS navigation and visualization.

Navigation is a planner-only dry run unless ``--execute-navigation`` is given.
Exited services restart with bounded backoff; SIGTERM stops children.

Run from the repository root (children are launched as ``python -m
station.l515.<module>`` with PYTHONPATH set to the repo root plus
``robot/wheels_app``, so the wheels-app client/sensor modules import)::

    DEPTH_SERVER_URL=http://<pi-ip>:8765 WHEELS_HOST=<wheels-ip> \
    python -m station.l515.l515_stack --directory ./l515-output

Children read their device addresses from the environment
(``DEPTH_SERVER_URL``, ``REACHY_URL``, ``REACHY_DAEMON_URL``, ``WHEELS_HOST``),
which this supervisor passes through unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
from station.l515.reachy_perception import atomic_json


def stalled_publisher(directory, now, grace=30):
    """HTTP failure alone is not evidence that a live publisher should be killed."""
    try:
        heartbeat=json.loads((directory/'rerun-heartbeat.json').read_text())
        return now-heartbeat['reported_at_unix']>grace
    except (OSError,ValueError,KeyError):
        return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=None,
                   help='WebGL viewer static dir (default: perception/depth_server/realsense_viewer/static)')
    p.add_argument('--native',action='store_true')
    p.add_argument('--execute-navigation',action='store_true',
                   help='allow the navigation child to issue bounded wheel pulses')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]  # repository root
    if args.assets is None:
        args.assets=root/'perception'/'depth_server'/'realsense_viewer'/'static'
    out=args.directory.resolve();out.mkdir(parents=True,exist_ok=True)
    pythonpath=[str(root),str(root/'robot'/'wheels_app')]
    if os.environ.get('PYTHONPATH'):pythonpath.append(os.environ['PYTHONPATH'])
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(pythonpath),
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    commands={
        'continuous':['continuous_mapping','--directory',str(out)],
        'perception':['reachy_rgbd','--directory',str(out)],
        'rerun':['l515_rerun','--directory',str(out)]+(['--native'] if args.native else []),
        'viewer':['l515_viewer','--directory',str(out),'--assets',str(args.assets.resolve())],
        'navigation':['reachy_navigation','--directory',str(out)]+
                     (['--execute'] if args.execute_navigation else []),
    }
    halt=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:halt.set())
    signal.signal(signal.SIGINT,lambda *_:halt.set())
    launched={}
    children={};attempts={k:0 for k in commands};due={k:0 for k in commands};failures=0
    rerun_failures=0
    health={}
    try:
        while not halt.is_set():
            for name,command in commands.items():
                child=children.get(name)
                if child is not None and child.poll() is None: continue
                if child is not None:
                    attempts[name]+=1
                    due[name]=time.monotonic()+min(30,2**min(attempts[name],5))
                    children[name]=None
                if time.monotonic()<due[name]: continue
                # Each restarted mapper establishes its own new origin. Keep
                # the previous map as a separately named archive.
                if name=='mapping' and (out/'live-map.ply').exists():
                    import shutil
                    shutil.copy2(out/'live-map.ply',out/f'archive-before-start-{time.time_ns()}.ply')
                with (out/f'{name}.log').open('ab') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','station.l515.'+command[0],*command[1:]],
                        cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                children[name]=child
                launched[name]=time.monotonic()
                (out/f'{name}.pid').write_text(str(child.pid)+'\n')
                print(f'{name}: started {child.pid}',flush=True)
            try:
                with urlopen('http://127.0.0.1:8777/',timeout=2) as response:
                    response.read(32)
                failures=0
                health['viewer']='ready'
            except OSError:
                failures+=1
                health['viewer']='HTTP unreachable; process retained'
            # The SDK's web server can disappear while its Python/gRPC parent
            # remains alive. A live PID alone is not a healthy Rerun viewer.
            child=children.get('rerun')
            if child is not None and child.poll() is None and time.monotonic()-launched.get('rerun',0)>30:
                try:
                    with urlopen('http://127.0.0.1:8778/',timeout=2) as response:
                        if response.status != 200: raise OSError('Rerun web server is unhealthy')
                    rerun_failures=0
                    health['rerun']='ready'
                except OSError:
                    rerun_failures+=1
                    health['rerun']='HTTP unreachable; publisher still running'
                    if rerun_failures>=6 and stalled_publisher(out,time.time()):
                        print('rerun: HTTP unavailable and publisher heartbeat stale; restarting',flush=True)
                        child.terminate();rerun_failures=0
            status={name:{'pid':c.pid if c else None,'running':c is not None and c.poll() is None,'restarts':attempts[name]} for name,c in children.items()}
            for name,value in health.items():status[name]['health']=value
            atomic_json(out/'stack-status.json',status)
            halt.wait(2)
    finally:
        for child in children.values():
            if child is not None and child.poll() is None: child.terminate()
        for child in children.values():
            if child is not None:
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill()


if __name__=='__main__': main()
