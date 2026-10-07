"""Single camera reader with GStreamer timestamps and actual-pose history.

Uses the local SDK IPC appsink. No motor commands. Poses are sampled SDK
telemetry; their original hardware timestamp is unavailable and is reported so.
"""
import io
import json
import threading
import time
import uuid
import zipfile
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_pose(samples, stamp, max_gap=.12):
    for a, b in zip(samples, samples[1:]):
        if a[0] <= stamp <= b[0] and 0 < b[0]-a[0] <= max_gap:
            fraction=(stamp-a[0])/(b[0]-a[0])
            T=np.eye(4)
            T[:3,3]=(1-fraction)*a[1][:3,3]+fraction*b[1][:3,3]
            T[:3,:3]=Slerp([0,1],Rotation.from_matrix([a[1][:3,:3],b[1][:3,:3]]))([fraction]).as_matrix()[0]
            yaw=a[2]+fraction*np.arctan2(np.sin(b[2]-a[2]),np.cos(b[2]-a[2]))
            return T,float(yaw),b[0]-a[0]
    raise ValueError('No recent actual-pose bracket for image timestamp')


class PosedCamera:
    def __init__(self, mini):
        self.mini=mini
        self.lock=threading.Lock()
        self.poses=deque(maxlen=150)
        self.latest=None
        self.frames=deque(maxlen=20)
        self.error='Camera warming up'
        self.stop_event=threading.Event()
        self.session=str(uuid.uuid4())
        self.sequence=0
        self.original_read=mini.media.camera.read
        mini.media.camera.read=self.frame
        self.threads=[threading.Thread(target=self._poses,daemon=True),threading.Thread(target=self._frames,daemon=True)]
        for thread in self.threads:thread.start()

    def _poses(self):
        last_head=None;last_joints=None
        while not self.stop_event.is_set():
            try:
                message=self.mini.client._last_head_pose
                joints=self.mini.client._last_joint_positions
                if message is None or joints is None or message is last_head or joints is last_joints:
                    self.stop_event.wait(.01)
                    continue
                last_head,last_joints=message,joints
                begin=time.monotonic()
                head=self.mini.get_current_head_pose().copy()
                yaw=float(self.mini.get_current_joint_positions()[0][0])
                end=time.monotonic()
                if end-begin<.05 and np.isfinite(head).all() and np.isfinite(yaw):
                    with self.lock:self.poses.append(((begin+end)/2,head,yaw))
            except Exception as exc:self.error=str(exc)
            self.stop_event.wait(.02)

    def _frames(self):
        from gi.repository import Gst
        camera=self.mini.media.camera
        while not self.stop_event.is_set():
            try:
                sample=camera._appsink_video.emit('try-pull-sample',100_000_000)
                if sample is None:continue
                buffer=sample.get_buffer()
                # IPC PTS is zero initially and later can use an unrelated
                # clock origin on this SDK. Never promote it to capture time.
                stamp=time.monotonic()
                basis='IPC frame arrival; source capture timestamp unavailable'
                width,height=camera.resolution
                raw=buffer.extract_dup(0,buffer.get_size())
                frame=np.frombuffer(raw,np.uint8).reshape(height,width,3).copy()
                self.sequence+=1
                with self.lock:
                    self.latest=(frame,stamp,self.sequence,int(buffer.pts),basis)
                    self.frames.append(self.latest)
                self.error=None
            except Exception as exc:
                self.error=str(exc)
                self.stop_event.wait(.05)

    def frame(self):
        with self.lock:latest=self.latest
        if latest is None or time.monotonic()-latest[1]>.5:return None
        return latest[0].copy()

    def bundle(self, age_ms=0):
        if not np.isfinite(age_ms) or not 0<=age_ms<=500:raise ValueError("Requested frame age must be 0–500 ms")
        target=time.monotonic()-age_ms/1000
        with self.lock:
            latest=min(self.frames,key=lambda f:abs(f[1]-target)) if self.frames else None
            poses=list(self.poses)
        if latest is None:raise ValueError(self.error or 'No camera frame')
        image,stamp,sequence,pts,basis=latest
        # The newest frame can arrive just after the most recent pose sample.
        if poses and poses[-1][0]<stamp:
            self.stop_event.wait(.03)
            with self.lock:poses=list(self.poses)
        try:
            head,yaw,gap=interpolate_pose(poses,stamp)
        except ValueError as exc:
            raise ValueError(f'{exc}; samples={len(poses)}, image_age={time.monotonic()-stamp:.3f}, '
                             f'pose_age={time.monotonic()-poses[-1][0] if poses else None}, reader={self.error}') from exc
        age=time.monotonic()-stamp
        if age>.5:raise ValueError('RGB frame is stale')
        ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise ValueError('JPEG encoding failed')
        report_time=time.monotonic()
        metadata=dict(session_id=self.session,sequence=sequence,base_T_head=head.tolist(),body_yaw=yaw,
            server_frame_age_ms=(report_time-stamp)*1000,gstreamer_pts_ns=pts,
            pose_bracket_ms=gap*1000,timestamp_basis=basis,source_capture_timestamp_available=False,
            pose_timestamp_basis='SDK telemetry read time; hardware sample age unavailable',
            pose_history=[dict(age_ms=(report_time-t)*1000,base_T_head=T.tolist(),body_yaw=y) for t,T,y in poses if report_time-t<1],
            head_pose_includes_body_yaw=True,width=image.shape[1],height=image.shape[0])
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('metadata.json',json.dumps(metadata))
            archive.writestr('color.jpg',jpeg.tobytes())
        return output.getvalue()

    def close(self):
        self.stop_event.set()
        for thread in self.threads:thread.join(timeout=1)
        self.mini.media.camera.read=self.original_read
