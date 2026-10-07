import numpy as np
import pytest
from station.l515.articulated_rgbd import camera_T_depth


def pose(angle=0, offset=(0, 0, 0)):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = offset
    return result


def calculate(head, mount, **extra):
    args = dict(reference_camera_T_depth=pose(offset=(.1, .2, .3)),
                reference_base_T_head=np.eye(4), base_T_head=head,
                head_T_camera=pose(offset=(.05, 0, .02)),
                reference_base_T_depth_mount=np.eye(4), base_T_depth_mount=mount)
    args.update(extra)
    return camera_T_depth(**args)


def test_reference_pose_reproduces_measured_calibration():
    np.testing.assert_allclose(calculate(np.eye(4), np.eye(4)), pose(offset=(.1, .2, .3)))


def test_body_rotation_cancels_when_both_sensors_rotate_together():
    rotated = pose(.7)
    np.testing.assert_allclose(calculate(rotated, rotated), calculate(np.eye(4), np.eye(4)), atol=1e-12)


def test_chassis_mount_accounts_for_head_rotation_and_camera_lever_arm():
    h_c = pose(offset=(.05, 0, .02))
    h = pose(np.pi / 2)
    result = calculate(h, np.eye(4))
    point = np.array([.3, .1, 2, 1])
    # Both paths must put the same observed depth point into the base frame.
    np.testing.assert_allclose(h @ h_c @ result @ point,
                               h_c @ pose(offset=(.1, .2, .3)) @ point, atol=1e-12)
    assert not np.allclose(result, calculate(h, h))


def test_combined_head_and_body_do_not_double_count_body_yaw():
    body = pose(.4)
    head_relative = pose(-.2, (.01, .02, .1))
    h_c = pose(offset=(.05, 0, .02))
    result = calculate(body @ head_relative, body)
    expected = np.linalg.inv(head_relative @ h_c) @ h_c @ pose(offset=(.1, .2, .3))
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_rejects_nonrigid_transform():
    with pytest.raises(ValueError):
        calculate(np.eye(4) * 2, np.eye(4))


def test_pose_transfer_at_different_rgb_and_depth_times():
    from station.l515.articulated_rgbd import posed_transform
    neutral=dict(x=0,y=0,z=0,roll=0,pitch=0,yaw=0)
    calibration=dict(head_poses=[dict(before=neutral,body_yaw=0)],l515_depth_to_reachy_optical=np.eye(4).tolist())
    rgb=dict(server_frame_age_ms=40,base_T_head=pose(.2).tolist(),body_yaw=.2,
        pose_history=[dict(age_ms=100,body_yaw=0),dict(age_ms=0,body_yaw=.2)])
    depth=dict(server_frame_age_ms=50)
    T,quality=posed_transform(calibration,np.eye(4),rgb,depth,(1,1.02),(1,1.02))
    np.testing.assert_allclose(T,pose(-.1),atol=1e-12)
    assert quality['pair_skew_ms']<11
    with pytest.raises(ValueError):
        posed_transform(calibration,np.eye(4),rgb,depth,(1,1.6),(1,1.02))
