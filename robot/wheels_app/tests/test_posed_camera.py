import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from reachy_wheels_app.posed_camera import interpolate_pose


def test_interpolates_pose_and_wraps_body_yaw():
    a=np.eye(4);b=np.eye(4)
    b[:3,3]=[.02,0,0]
    b[:3,:3]=Rotation.from_euler('z',.2).as_matrix()
    pose,yaw,gap=interpolate_pose([(1,a,3.13),(1.1,b,-3.13)],1.05)
    np.testing.assert_allclose(pose[:3,3],[.01,0,0])
    assert abs(abs(yaw)-np.pi)<1e-6
    assert abs(Rotation.from_matrix(pose[:3,:3]).as_rotvec()[2]-.1)<1e-6
    assert gap<.12


def test_no_extrapolation_or_long_gaps():
    for samples,stamp in [([(1,np.eye(4),0),(2,np.eye(4),0)],1.5), ([(1,np.eye(4),0)],2)]:
        with pytest.raises(ValueError):interpolate_pose(samples,stamp)


def test_bundle_selects_historical_image_with_matching_pose_and_honest_clock():
    import threading,time,io,json,zipfile
    from collections import deque
    from reachy_wheels_app.posed_camera import PosedCamera
    camera=PosedCamera.__new__(PosedCamera)
    camera.lock=threading.Lock();camera.stop_event=threading.Event()
    camera.session='test';camera.error=None
    now=time.monotonic();image=np.zeros((12,16,3),np.uint8)
    camera.frames=deque([(image,now-.12,1,0,'IPC frame arrival; source capture timestamp unavailable'),
                         (image,now-.02,2,0,'IPC frame arrival; source capture timestamp unavailable')])
    camera.poses=deque([(now-.15,np.eye(4),0),(now-.09,np.eye(4),0),(now-.01,np.eye(4),0)])
    output=camera.bundle(age_ms=120)
    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        metadata=json.loads(archive.read('metadata.json'))
    assert metadata['sequence']==1
    assert metadata['source_capture_timestamp_available'] is False
    assert metadata['server_frame_age_ms']>=120
    with pytest.raises(ValueError):camera.bundle(age_ms=1000)
