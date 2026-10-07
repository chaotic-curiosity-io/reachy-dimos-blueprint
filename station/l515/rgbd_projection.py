"""Fixed-head optical projection and conservative visible-surface association."""
import io
import json
import os
from pathlib import Path
import zipfile
import cv2
import numpy as np

FRAME='reachy_head_camera_optical'

REPO_ROOT=Path(__file__).resolve().parents[2]
CALIBRATION_NAME='paired-extrinsics-report.json'

# Device addresses come from the environment so no LAN address is baked in.
# DEPTH_SERVER_URL is the Raspberry Pi depth streamer (perception/depth_server),
# REACHY_URL the wheels app on the robot, REACHY_DAEMON_URL the stock daemon.
DEFAULT_DEPTH_URL=os.environ.get('DEPTH_SERVER_URL','')
DEFAULT_REACHY_URL=os.environ.get('REACHY_URL','http://reachy-mini.local:8042')
DEFAULT_DAEMON_URL=os.environ.get('REACHY_DAEMON_URL','http://reachy-mini.local:8000')


def require_url(value, name, env):
    """Fail early with a clear message when a device address is unset."""
    if not value:
        raise SystemExit(f'{name} is not set: pass {name} or export {env} '
                         f'(e.g. {env}=http://<pi-ip>:8765)')
    return value.rstrip('/')


def calibration_path(directory, explicit=None):
    """Locate the paired camera/L515 extrinsics report.

    Precedence: an explicit ``--calibration`` path, then ``$CALIBRATION_REPORT``,
    then ``<directory>/calibration/`` (a copy kept next to the run output), then
    ``perception/calibration/`` where ``fit_pair.py`` writes it by default.
    """
    candidates=[explicit,os.environ.get('CALIBRATION_REPORT'),
                Path(directory)/'calibration'/CALIBRATION_NAME,
                REPO_ROOT/'perception'/'calibration'/CALIBRATION_NAME]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser()
    raise SystemExit('No '+CALIBRATION_NAME+' found. Run the procedure in '
                     'perception/calibration/README.md or pass --calibration.')


def check_pose(calibration, pose, body_yaw):
    reference=next(p for p in calibration['head_poses'] if p['before'] is not None)
    a=np.array([pose[k] for k in ('x','y','z','roll','pitch','yaw')],float)
    b=np.array([reference['before'][k] for k in ('x','y','z','roll','pitch','yaw')],float)
    if not np.isfinite(a).all() or not np.isfinite(body_yaw):
        raise ValueError('Nonfinite head/body pose')
    if np.linalg.norm(a[:3]-b[:3])>.003 or np.max(np.abs(a[3:]-b[3:]))>.02 or abs(body_yaw-reference['body_yaw'])>.02:
        raise ValueError('Head/body is outside the calibrated fixed pose')


def unpack_bundle(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for info in z.infolist():
            if info.file_size>2_000_000:raise ValueError('Oversize calibration member')
        m=json.loads(z.read('metadata.json'))
        i=m['depth_intrinsics']
        if (i['width'],i['height'])!=(320,240):raise ValueError('Unexpected depth mode')
        raw=z.read('depth.u16')
        if len(raw)!=320*240*2:raise ValueError('Invalid depth length')
        depth=np.frombuffer(raw,dtype='<u2').reshape(240,320)
    return depth,m


def project(depth, metadata, calibration, image, transform=None):
    i=metadata['depth_intrinsics']
    if i['model']!='distortion.none':raise ValueError('Unsupported depth distortion')
    if image.shape[:2]!=(720,1280):raise ValueError('Reachy image resolution changed')
    if metadata['color_intrinsics']!=calibration['l515_color_intrinsics']:
        raise ValueError('L515 calibration/mode changed')
    scale=float(metadata['depth_scale_m'])
    if not np.isfinite(scale) or not 0<scale<.01:raise ValueError('Invalid depth scale')
    v,u=np.indices(depth.shape)
    z=depth.astype(float)*scale
    valid=(z>=.15)&(z<=4)
    xyz=np.column_stack(((u[valid]-i['ppx'])*z[valid]/i['fx'],(v[valid]-i['ppy'])*z[valid]/i['fy'],z[valid]))
    T=np.array(calibration['l515_depth_to_reachy_optical'] if transform is None else transform)
    xyz=xyz@T[:3,:3].T+T[:3,3]
    xyz=xyz[np.isfinite(xyz).all(axis=1)&(xyz[:,2]>.05)]
    if not len(xyz):return xyz,np.empty((0,2),int),np.empty((0,3),np.uint8)
    intr=calibration['reachy_intrinsics']
    uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),np.array(intr['K']),np.array(intr['D']))
    uv=np.rint(uv.reshape(-1,2)).astype(int)
    keep=(uv[:,0]>=0)&(uv[:,0]<1280)&(uv[:,1]>=0)&(uv[:,1]<720)
    xyz,uv=xyz[keep],uv[keep]
    # Keep nearest projected depth sample at each RGB pixel.
    order=np.argsort(xyz[:,2]);keys=uv[order,1]*1280+uv[order,0]
    _,first=np.unique(keys,return_index=True);keep=order[first]
    xyz,uv=xyz[keep],uv[keep]
    colors=image[uv[:,1],uv[:,0],::-1]
    return xyz,uv,colors


def surface(mask, xyz, uv):
    """Return bounds of supported visible surface, not full physical object extent."""
    mask=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8))
    selected=xyz[mask[uv[:,1],uv[:,0]]>0]
    if len(selected)<30:return None
    # Reject mixed foreground/background depth rather than invent an association.
    lo,hi=np.quantile(selected[:,2],[.1,.9])
    if hi-lo>max(.2,.25*float(np.median(selected[:,2]))):return None
    low,high=np.quantile(selected,[.05,.95],axis=0)
    return dict(center_m=((low+high)/2).tolist(),size_m=(high-low).tolist(),
                support_points=len(selected),bounds_kind='visible_surface_percentile_bounds',
                frame_id=FRAME)
