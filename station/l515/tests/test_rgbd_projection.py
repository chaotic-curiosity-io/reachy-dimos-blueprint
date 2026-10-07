import copy
import numpy as np
import pytest
from station.l515.rgbd_projection import check_pose,project,surface

def config():
    pose=dict(x=0,y=0,z=0,roll=0,pitch=0,yaw=0)
    c=dict(head_poses=[dict(before=pose,body_yaw=0)],l515_color_intrinsics={'mode':'test'},
           l515_depth_to_reachy_optical=np.eye(4).tolist(),reachy_intrinsics={'K':[[100,0,640],[0,100,360],[0,0,1]],'D':[0]*5})
    return c,pose

def test_pose_gate_rejects_translation_rotation_body_and_nan():
    c,p=config();check_pose(c,p,0)
    for k,v in [('x',.004),('yaw',.03),('x',float('nan'))]:
        with pytest.raises(ValueError):check_pose(c,dict(p,**{k:v}),0)
    with pytest.raises(ValueError):check_pose(c,p,.03)

def test_projection_transform_scale_pixels_and_invalid_depth():
    c,_=config();c['l515_depth_to_reachy_optical'][0][3]=.1
    meta=dict(depth_intrinsics=dict(model='distortion.none',fx=100,fy=100,ppx=0,ppy=0),depth_scale_m=.001,color_intrinsics={'mode':'test'})
    image=np.zeros((720,1280,3),np.uint8);image[360,650]=[10,20,30]
    xyz,uv,color=project(np.array([[1000,0]],np.uint16),meta,c,image)
    np.testing.assert_allclose(xyz,[[.1,0,1]])
    np.testing.assert_array_equal(uv,[[650,360]])
    np.testing.assert_array_equal(color,[[30,20,10]])
    with pytest.raises(ValueError):project(np.ones((1,1)),meta,c,image[:100])
    bad=copy.deepcopy(meta);bad['color_intrinsics']={}
    with pytest.raises(ValueError):project(np.ones((1,1)),bad,c,image)

def test_surface_needs_support_and_rejects_mixed_depth():
    uv=np.array([[x,y] for x in range(10,20) for y in range(10,20)])
    xyz=np.column_stack((uv*.01,np.ones(100)))
    mask=np.ones((40,40),np.uint8)
    assert surface(mask,xyz,uv)['support_points']==100
    assert surface(mask,xyz[:10],uv[:10]) is None
    xyz[50:,2]=3
    assert surface(mask,xyz,uv) is None

def test_native_visible_surface_message_and_empty_cloud():
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.msgs.vision_msgs.Detection3D import Detection3D
    from dimos.perception.detection.type.detection3d.bbox import Detection3DBBox
    d=Detection3DBBox(bbox=(0,0,10,10),track_id=-1,class_id=41,confidence=.8,name='cup',ts=1.0,
        image=Image.from_opencv(np.zeros((20,20,3),np.uint8)),center=Vector3(.1,.2,1),size=Vector3(.1,.1,.05),frame_id='reachy_head_camera_optical')
    message=d.to_detection3d_msg()
    decoded=Detection3D.lcm_decode(message.lcm_encode())
    assert decoded.frame_id=='reachy_head_camera_optical'
    assert decoded.bbox.center.position.z==1
    empty=PointCloud2.from_numpy(np.empty((0,3),np.float32),frame_id='reachy_head_camera_optical',timestamp=1)
    assert len(empty.pointcloud.points)==0
