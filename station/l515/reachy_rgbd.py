"""Experimental articulated RGB-D semantic preview. No navigation or actuation."""
import argparse
import io
import zipfile
import hashlib
import json
from pathlib import Path
import signal
import shutil
import socket
from urllib.parse import urlsplit, urlunsplit
import threading
import time
import uuid
from urllib.request import urlopen
from urllib.error import HTTPError

import cv2
import numpy as np
from station.l515.rgbd_projection import (FRAME, DEFAULT_DAEMON_URL, DEFAULT_DEPTH_URL, DEFAULT_REACHY_URL,
                                         REPO_ROOT, calibration_path, project, require_url, surface, unpack_bundle)
import os
from station.l515.articulated_rgbd import model_head_T_camera, posed_transform
from station.l515.reachy_perception import open_memory, remember, atomic_json


def fetch(url):
    begin=time.monotonic()
    with urlopen(url,timeout=3) as response:data=response.read(4_000_001)
    if len(data)>4_000_000:raise ValueError('Oversize response')
    return data,begin,time.monotonic()


def matching_map_pose(early, mapped):
    """Never attach a later/earlier scan's world pose to an alignment pair."""
    a,b=early['meta'],mapped['meta']
    if (a['session_id'],a['depth_frame_number']) != (b['session_id'],b['depth_frame_number']):
        return None
    return np.array(mapped['map_pose']) if mapped.get('map_pose') is not None else None



def main():
    from ultralytics import YOLO
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
    from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
    from dimos.msgs.std_msgs.Header import Header
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
    from dimos.perception.detection.type.detection3d.bbox import Detection3DBBox
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--reachy',default=DEFAULT_REACHY_URL,help='wheels app base URL (env REACHY_URL)')
    p.add_argument('--daemon',default=DEFAULT_DAEMON_URL,help='Reachy daemon base URL (env REACHY_DAEMON_URL)')
    p.add_argument('--pi',default=DEFAULT_DEPTH_URL,help='Pi depth streamer base URL (env DEPTH_SERVER_URL)')
    p.add_argument('--calibration',type=Path,default=None,help='paired-extrinsics-report.json override')
    p.add_argument('--mjcf',type=Path,
                   default=Path(os.environ.get('REACHY_MJCF',REPO_ROOT/'station/assets/official_reachy/reachy_mini.xml')),
                   help='official Reachy Mini MJCF (env REACHY_MJCF); only its head/camera sites are read')
    args=p.parse_args();out=args.directory
    args.pi=require_url(args.pi,'--pi','DEPTH_SERVER_URL')
    path=calibration_path(out,args.calibration)
    calibration=json.loads(path.read_text());revision=hashlib.sha256(path.read_bytes()).hexdigest()
    model_path=args.mjcf
    head_T_camera=model_head_T_camera(model_path)
    geometry_revision=hashlib.sha256(model_path.read_bytes()).hexdigest()
    model=YOLO(str(out/'models/yolo11n-seg.pt'))
    db=open_memory(out/'observations.sqlite3')
    map_pose=None
    raw=LCMTransport('/reachy/color_image',Image)
    annotated=LCMTransport('/reachy/perception/annotated_image',Image)
    det2=LCMTransport('/reachy/perception/detections2d',Detection2DArray)
    det3=LCMTransport('/reachy/experimental/detections3d',Detection3DArray)
    cloud=LCMTransport('/reachy/experimental/rgbd_cloud',PointCloud2)
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    last_frame=None;last_memory=0
    resolved_at=0;endpoints={}
    def emit_empty(ts):
        det3.publish(Detection3DArray(header=Header(ts,FRAME),detections=[],detections_length=0))
        cloud.publish(PointCloud2.from_numpy(np.empty((0,3),np.float32),frame_id=FRAME,timestamp=ts))
    print('RGB-D semantic preview ready',flush=True)
    try:
        while not stop.wait(.1):
            try:
                if time.monotonic()-resolved_at>60:
                    # Resolve mDNS outside the synchronized acquisition window.
                    # macOS resolution can otherwise add seconds to every GET.
                    addresses={}
                    for key,base in [('reachy',args.reachy),('daemon',args.daemon),('pi',args.pi)]:
                        parts=urlsplit(base)
                        host=parts.hostname
                        if host not in addresses:addresses[host]=socket.gethostbyname(host)
                        netloc=addresses[host]+(':'+str(parts.port) if parts.port else '')
                        endpoints[key]=urlunsplit((parts.scheme,netloc,parts.path,parts.query,parts.fragment))
                    resolved_at=time.monotonic()
                start=time.monotonic()
                gate_error=None;timing={};transform=None
                desired_age_ms=0
                try:
                    acquisition=out/'alignment-frame.npz'
                    if not acquisition.exists():acquisition=out/'mapping-frame.npz'
                    with np.load(acquisition,allow_pickle=False) as data:
                        depth=data['depth'].copy();handoff=json.loads(str(data['handoff']))
                    if time.monotonic()-handoff['created_monotonic']>.5:
                        raise ValueError('No fresh mapping frame')
                    meta=handoff['meta'];pb=handoff['begin'];pe=handoff['end']
                    depth_time=handoff['depth_time']
                    desired_age_ms=max(0,min(500,(time.monotonic()-depth_time)*1000))
                except Exception as exc:
                    gate_error='Depth unavailable: '+str(exc)
                try:
                    rgb,rb,re=fetch(endpoints['reachy']+f'/api/camera/posed-frame?age_ms={desired_age_ms:.2f}')
                    with zipfile.ZipFile(io.BytesIO(rgb)) as archive:
                        rgb_meta=json.loads(archive.read('metadata.json'))
                        rgb=archive.read('color.jpg')
                except OSError as exc:
                    # Missing historical pose must block 3D projection, not
                    # fresh RGB classification. Keep the two statuses separate.
                    detail=exc.read(4096).decode('utf-8',errors='replace') if isinstance(exc,HTTPError) else str(exc)
                    gate_error='Posed RGB unavailable: '+detail
                    rgb,rb,re=fetch(endpoints['reachy']+'/api/camera')
                    rgb_meta=dict(server_frame_age_ms=0,sequence=None,session_id=None,
                        timestamp_basis='HTTP receipt estimate; pose unavailable',
                        source_capture_timestamp_available=False)
                try:
                    if gate_error:raise ValueError(gate_error)
                    transform,timing=posed_transform(calibration,head_T_camera,rgb_meta,meta,(rb,re),(pb,pe))
                    frame=(meta['session_id'],meta['depth_frame_number'])
                    if frame==last_frame:raise ValueError('Repeated Pi frame')
                    last_frame=frame
                    expected_serial=calibration.get('l515_serial')
                    if expected_serial and meta['serial']!=expected_serial:raise ValueError('L515 serial changed')
                except Exception as exc:gate_error=str(exc)
                map_pose=np.array(handoff['map_pose']) if gate_error is None and handoff['map_pose'] is not None else None
                image=cv2.imdecode(np.frombuffer(rgb,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('Invalid RGB image')
                ts=time.time()-(time.monotonic()-((rb+re)/2-rgb_meta['server_frame_age_ms']/1000))
                results=model.predict(image,device='cpu',conf=.5,imgsz=640,retina_masks=True,verbose=False)
                native=Image.from_opencv(image,ts=ts,frame_id=FRAME)
                raw.publish(native)
                converted=ImageDetections2D.from_ultralytics_result(native,results)
                message=converted.to_ros_detection2d_array();message.header.frame_id=FRAME
                for d in message.detections:d.header.frame_id=FRAME
                det2.publish(message)
                preview=results[0].plot()
                annotated.publish(Image.from_opencv(preview,ts=ts,frame_id=FRAME))
                xyz=np.empty((0,3));uv=np.empty((0,2),int);colors=np.empty((0,3),np.uint8)
                objects=[];messages=[];overlay=image.copy()
                if gate_error is None:
                    try:
                        if time.monotonic()-start>3:raise ValueError('Observation exceeded 3-second processing budget')
                        xyz,uv,colors=project(depth,meta,calibration,image,transform=transform)
                    except Exception as exc:gate_error=str(exc)
                result=results[0]
                for index,box in enumerate(result.boxes):
                    cls=int(box.cls.item());confidence=float(box.conf.item());bbox=box.xyxy[0].cpu().tolist()
                    obj=dict(name=result.names[cls],class_id=cls,confidence=confidence,bbox_xyxy=bbox,
                             position_3d=None,surface_3d=None)
                    if gate_error is None and result.masks is not None:
                        mask=result.masks.data[index].cpu().numpy()>.5
                        if mask.shape!=image.shape[:2]:raise ValueError('Segmentation mask/image mismatch')
                        support=surface(mask,xyz,uv)
                        if support:
                            obj.update(position_3d=support['center_m'],surface_3d=support)
                            d=Detection3DBBox(bbox=tuple(bbox),track_id=-1,class_id=cls,confidence=confidence,name=obj['name'],ts=ts,image=native,center=Vector3(*support['center_m']),size=Vector3(*support['size_m']),frame_id=FRAME)
                            messages.append(d.to_detection3d_msg())
                    objects.append(obj)
                if gate_error is None:
                    for (u,v),point in zip(uv[::3],xyz[::3]):
                        cv2.circle(overlay,(int(u),int(v)),2,(0,int(255*max(0,1-point[2]/4)),255),-1)
                    cloud.publish(PointCloud2.from_numpy(xyz.astype(np.float32),frame_id=FRAME,timestamp=ts))
                if gate_error is not None:emit_empty(ts)
                det3.publish(Detection3DArray(header=Header(ts,FRAME),detections=messages,detections_length=len(messages)))
                report=dict(observation_id=str(uuid.uuid4()),state='running',reported_at_unix=time.time(),source_received_at_unix=ts,
                    model='yolo11n-seg.pt',frame_id=FRAME,coordinate_space='image_pixels_with_optical_surface_estimates',position_3d=None,
                    objects=objects,rgbd_state='preview' if gate_error is None else 'blocked',rgbd_reason=gate_error,
                    projected_points=len(xyz),calibration_sha256=revision,scale_basis='estimated iPad size; not metrically validated',
                    synchronized=False,timestamp_basis=rgb_meta['timestamp_basis']+'; SDK read-time pose interpolation; bounded HTTP clock relation',
                    source_capture_timestamp_available=rgb_meta['source_capture_timestamp_available'],
                    articulation='body-mounted L515; independent head camera',timing=timing,
                    camera_transform_source='official model lever arm plus fitted stereo reference',
                    geometry_sha256=geometry_revision,
                    rgb_sequence=rgb_meta['sequence'],rgb_session=rgb_meta['session_id'],
                    l515_depth_to_reachy_optical=transform.tolist() if transform is not None else None,
                    scene_assumption='moving preview; inter-frame chassis displacement and moving objects are not compensated',
                    inference_ms=round((time.monotonic()-start)*1000),actuation_enabled=False)
                report['map_segment']=handoff['segment'] if gate_error is None else None
                # ICP may have finished while segmentation ran. Use its pose
                # only if it belongs to exactly this sensor session/frame.
                if gate_error is None and map_pose is None:
                    try:
                        with np.load(out/'mapping-frame.npz',allow_pickle=False) as data:
                            mapped=json.loads(str(data['handoff']))
                        map_pose=matching_map_pose(handoff,mapped)
                        if map_pose is not None:report['map_segment']=mapped['segment']
                    except (OSError,ValueError,KeyError):pass
                report['map_T_camera']=(map_pose @ np.linalg.inv(transform)).tolist() if gate_error is None and map_pose is not None else None
                atomic_json(out/'perception-live.json',report)
                if gate_error is None:
                    # Queue accepted detections even when ICP has not finished.
                    # The mapper joins by exact frame, not by latest-file timing.
                    report['depth_frame_key']=list(frame)
                    queue=out/'semantic-pending';queue.mkdir(exist_ok=True)
                    atomic_json(queue/(report['observation_id']+'.json'),report)
                    for old in queue.glob('*.json'):
                        if time.time()-old.stat().st_mtime>30:old.unlink(missing_ok=True)
                temp=out/'rgbd-live.tmp'
                with temp.open('wb') as f:np.savez_compressed(f,points=xyz[::2],colors=colors[::2],overlay=overlay,report=json.dumps(report))
                temp.replace(out/'rgbd-live.npz')
                if gate_error is None:
                    accepted=out/'rgbd-last-accepted.tmp'
                    shutil.copyfile(out/'rgbd-live.npz',accepted)
                    accepted.replace(out/'rgbd-last-accepted.npz')
                if time.monotonic()-last_memory>5:
                    remember(db,report);last_memory=time.monotonic()
            except Exception as exc:
                emit_empty(time.time())
                atomic_json(out/'perception-live.json',dict(state='unavailable',reported_at_unix=time.time(),rgbd_state='blocked',rgbd_reason=str(exc),objects=[]))
                print('Waiting:',exc,flush=True)
                stop.wait(1)
    finally:
        emit_empty(time.time())
        for t in (raw,annotated,det2,det3,cloud):t.stop()
        db.close()
        atomic_json(out/'perception-live.json',dict(state='stopped',reported_at_unix=time.time(),objects=[]))

if __name__=='__main__':main()
