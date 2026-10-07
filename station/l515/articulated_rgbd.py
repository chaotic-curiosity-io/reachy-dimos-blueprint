"""Articulated camera/depth transforms; explicit frames, no guessed mount offsets.

T_A_B maps column vectors from frame B into frame A. All inputs must describe
actual poses at their respective sensor capture times in one fixed robot-base
frame. Body yaw must already be included in base_T_head; do not add it twice.
"""
import numpy as np


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite 4x4 rigid transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    return matrix


def camera_T_depth(*, reference_camera_T_depth, reference_base_T_head,
                   base_T_head, head_T_camera, reference_base_T_depth_mount,
                   base_T_depth_mount):
    """Transfer the measured stereo calibration to another articulated pose.

    head_T_camera is the optical camera's fixed transform in the SDK head
    frame. Depth mount is identity for a sensor fixed to the robot base, or
    the rotating body's actual pose for a body-mounted sensor. Chassis motion
    between RGB and depth capture requires an additional common-world pose;
    callers must not use this formula to hide unsynchronized captures.
    """
    c0_d0, b_h0, b_h, h_c, b_m0, b_m = map(rigid, (
        reference_camera_T_depth, reference_base_T_head, base_T_head,
        head_T_camera, reference_base_T_depth_mount, base_T_depth_mount))
    b_c0 = b_h0 @ h_c
    b_c = b_h @ h_c
    mount_T_depth = np.linalg.inv(b_m0) @ b_c0 @ c0_d0
    return np.linalg.inv(b_c) @ b_m @ mount_T_depth


def xyzrpy_pose(pose):
    from scipy.spatial.transform import Rotation
    result=np.eye(4)
    result[:3,:3]=Rotation.from_euler('xyz',[pose[k] for k in ('roll','pitch','yaw')]).as_matrix()
    result[:3,3]=[pose[k] for k in ('x','y','z')]
    return rigid(result)


def body_pose(yaw):
    return xyzrpy_pose(dict(x=0,y=0,z=0,roll=0,pitch=0,yaw=yaw))


def model_head_T_camera(xml_path):
    """Read same-link head and optical sites from the official MJCF model."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    root=ET.parse(xml_path).getroot()
    for parent in root.iter('body'):
        h=parent.find("site[@name='head']")
        c=parent.find("site[@name='camera_optical']")
        if h is not None and c is not None:
            def matrix(element):
                T=np.eye(4)
                T[:3,3]=np.fromstring(element.get('pos','0 0 0'),sep=' ')
                q=np.fromstring(element.get('quat','1 0 0 0'),sep=' ')
                T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
                return T
            return np.linalg.inv(matrix(h)) @ matrix(c)
    raise ValueError('Head and optical camera must be explicitly on the same model link')


def posed_transform(calibration, head_T_camera, rgb_metadata, depth_metadata, rgb_interval, depth_interval):
    """Use IPC image pose and body pose interpolated at the depth receipt estimate.

    Clock relation is an interval estimate from each HTTP round trip and server
    monotonic ages, not hardware synchronization. Uncertain pairs are rejected.
    """
    rb,re=rgb_interval;pb,pe=depth_interval
    if max(re-rb,pe-pb)>.4:raise ValueError('Sensor round trip exceeds 400 ms')
    rgb_age=float(rgb_metadata['server_frame_age_ms'])/1000
    depth_age=float(depth_metadata['server_frame_age_ms'])/1000
    if not (0<=rgb_age<.5 and 0<=depth_age<.5):raise ValueError('Stale sensor frame')
    rgb_server=(rb+re)/2
    rgb_time=rgb_server-rgb_age
    depth_time=(pb+pe)/2-depth_age
    skew=abs(rgb_time-depth_time)
    uncertainty=((re-rb)+(pe-pb))/2
    if skew+uncertainty>.2:raise ValueError('RGB/depth timing uncertainty exceeds 200 ms')
    history=sorted([(rgb_server-float(p['age_ms'])/1000,float(p['body_yaw'])) for p in rgb_metadata['pose_history']])
    depth_yaw=None
    for (ta,ya),(tb,yb) in zip(history,history[1:]):
        if ta<=depth_time<=tb and 0<tb-ta<=.12:
            depth_yaw=ya+(depth_time-ta)/(tb-ta)*np.arctan2(np.sin(yb-ya),np.cos(yb-ya))
            break
    if depth_yaw is None:raise ValueError('No body-pose history bracket for depth frame')
    reference=next(p for p in calibration['head_poses'] if p['before'] is not None)
    transform=camera_T_depth(reference_camera_T_depth=calibration['l515_depth_to_reachy_optical'],
        reference_base_T_head=xyzrpy_pose(reference['before']),
        base_T_head=rgb_metadata['base_T_head'],head_T_camera=head_T_camera,
        reference_base_T_depth_mount=body_pose(reference['body_yaw']),base_T_depth_mount=body_pose(depth_yaw))
    return transform,dict(pair_skew_ms=skew*1000,timing_uncertainty_ms=uncertainty*1000,
                          depth_body_yaw=float(depth_yaw),rgb_body_yaw=rgb_metadata['body_yaw'])
