"""Capture stationary paired camera observations; no motion commands.

Grabs three bracketed samples from the Reachy head camera (wheels app
``/api/camera``) and the L515 colour stream (Pi ``/api/calibration-frame``),
detects the ChArUco board in both, and saves the middle pair only if both views
share >=12 corners, corners moved <1.5 px across the samples, and the head/body
pose read from the daemon did not change. Output: ``<directory>/<UTC stamp>/``
with ``reachy.jpg``, ``l515.zip`` and ``observation.json`` (what fit_pair.py reads).

Addresses come from flags or the environment:
  --reachy / REACHY_URL         default http://reachy-mini.local:8042
  --daemon / REACHY_DAEMON_URL  default http://reachy-mini.local:8000
  --pi     / DEPTH_SERVER_URL   required, e.g. http://<pi-ip>:8765
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import datetime
import io
import json
from pathlib import Path
import time
from urllib.request import urlopen
import zipfile
import cv2
import numpy as np


def fetch(url):
    started=time.time()
    with urlopen(url,timeout=8) as r:
        body=r.read(4*1024*1024+1)
    if len(body)>4*1024*1024: raise ValueError('Oversized response')
    return body,dict(request_started_unix=started,response_received_unix=time.time())


def detect(data):
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Invalid image')
    board=cv2.aruco.CharucoBoard((5,7),.0235,.01175,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    corners,ids,_,_=cv2.aruco.CharucoDetector(board).detectBoard(image)
    if ids is None or len(ids)<12: raise ValueError('Fewer than 12 detected corners')
    return dict(image_size=[image.shape[1],image.shape[0]],ids=ids.flatten().tolist(),corners=corners.reshape(-1,2).tolist())


def stability(a,b):
    x=dict(zip(a['ids'],a['corners']));y=dict(zip(b['ids'],b['corners']))
    ids=sorted(x.keys()&y.keys())
    if len(ids)<12: raise ValueError('Fewer than 12 stable common corners')
    return float(np.max(np.linalg.norm(np.array([x[i] for i in ids])-np.array([y[i] for i in ids]),axis=1)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--reachy',default=os.environ.get('REACHY_URL','http://reachy-mini.local:8042'))
    p.add_argument('--daemon',default=os.environ.get('REACHY_DAEMON_URL','http://reachy-mini.local:8000'))
    p.add_argument('--pi',default=os.environ.get('DEPTH_SERVER_URL',''),
                   help='Pi depth streamer base URL, e.g. http://<pi-ip>:8765 (env DEPTH_SERVER_URL)')
    args=p.parse_args()
    if not args.pi:p.error('--pi or DEPTH_SERVER_URL is required (e.g. http://<pi-ip>:8765)')
    args.reachy,args.daemon,args.pi=(u.rstrip('/') for u in (args.reachy,args.daemon,args.pi))
    pose_before=json.loads(fetch(args.daemon+'/api/state/present_head_pose')[0])
    body_before=json.loads(fetch(args.daemon+'/api/state/present_body_yaw')[0])
    samples=[]
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(fetch,args.reachy+'/api/camera')
            b=pool.submit(fetch,args.pi+'/api/calibration-frame')
            rgb,rt=a.result();bundle,pt=b.result()
        with zipfile.ZipFile(io.BytesIO(bundle)) as z:
            color=z.read('color.bmp');meta=json.loads(z.read('metadata.json'))
        rd,pd=detect(rgb),detect(color)
        common=sorted(set(rd['ids'])&set(pd['ids']))
        if len(common)<12: raise ValueError('Too few common corners between cameras')
        samples.append(dict(rgb=rgb,bundle=bundle,reachy=rd,l515=pd,metadata=meta,
                            receipt_times=dict(reachy=rt,l515=pt),common_ids=common))
    if len({s['metadata']['session_id'] for s in samples})!=1:raise ValueError('Pi restarted during capture')
    if len({s['metadata']['color_frame_number'] for s in samples})!=3:raise ValueError('Repeated Pi frame')
    shifts={k:max(stability(samples[0][k],s[k]) for s in samples[1:]) for k in ['reachy','l515']}
    if max(shifts.values())>1.5:raise ValueError(f'Board/camera moved during capture: {shifts}')
    pose_after=json.loads(fetch(args.daemon+'/api/state/present_head_pose')[0])
    body_after=json.loads(fetch(args.daemon+'/api/state/present_body_yaw')[0])
    if any(abs(pose_after[k]-pose_before[k])>.002 for k in ['x','y','z']) or any(abs(pose_after[k]-pose_before[k])>.02 for k in ['roll','pitch','yaw']) or abs(body_after-body_before)>.02:
        raise ValueError('Head/body pose changed during capture')
    sample=samples[1]
    sample['head_pose_before']=pose_before
    sample['head_pose_after']=pose_after
    sample['body_yaw_before']=body_before
    sample['body_yaw_after']=body_after
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.directory/stamp;out.mkdir(parents=True,exist_ok=False)
    (out/'reachy.jpg').write_bytes(sample.pop('rgb'))
    (out/'l515.zip').write_bytes(sample.pop('bundle'))
    sample.update(stationarity_max_shift_pixels=shifts,square_length_m=.0235,scale_basis='estimated from iPad photo',
        synchronized=False,timing_note='Stationary bracketing check, not synchronized hardware capture. Reachy capture freshness unavailable.',
        board_squares=[5,7],dictionary='DICT_5X5_100')
    (out/'observation.json').write_text(json.dumps(sample,indent=2)+'\n')
    print(json.dumps(dict(saved=str(out),common_corners=len(sample['common_ids']),shifts=shifts)))

if __name__=='__main__': main()
