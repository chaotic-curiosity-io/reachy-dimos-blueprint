"""Needs the full dimOS environment (see station/README.md); skipped otherwise."""
import pytest
pytest.importorskip('open3d')
pytest.importorskip('scipy')
pytest.importorskip('dimos')
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('dimos')
from station.l515.l515_mapping import L515Odometry


def room():
    rng = np.random.default_rng(12)
    wall = np.column_stack((rng.uniform(-1, 1, 2000), rng.uniform(-0.7, 0.7, 2000), np.full(2000, 2.5)))
    side = np.column_stack((np.full(2000, -1), rng.uniform(-0.7, 0.7, 2000), rng.uniform(0.6, 2.5, 2000)))
    floor = np.column_stack((rng.uniform(-1, 1, 2000), np.full(2000, 0.7), rng.uniform(0.6, 2.5, 2000)))
    return np.vstack((wall, side, floor))


def test_recovers_known_metric_motion_without_mount_calibration():
    points = room()
    matcher = L515Odometry()
    np.testing.assert_array_equal(matcher.update(points, 10), np.eye(4))
    rotation = Rotation.from_euler('y', 2, degrees=True).as_matrix()
    translation = np.array([0.025, -0.005, 0.02])
    # Same world surfaces observed by the translated/rotated sensor.
    observed = (points - translation) @ rotation
    pose = matcher.update(observed, 10.5)
    assert pose is not None, matcher.status()
    np.testing.assert_allclose(pose[:3, 3], translation, atol=0.004)
    assert Rotation.from_matrix(pose[:3, :3] @ rotation.T).magnitude() < 0.005
    assert matcher.status()['state'] == 'tracking'
    assert matcher.status()['pose_is_wheel_base'] is False


def test_single_plane_is_not_claimed_as_full_localization():
    points = room()[:2000]
    matcher = L515Odometry()
    matcher.update(points, 10)
    assert matcher.update(points, 10.2) is None
    assert 'insufficient to constrain' in matcher.status()['reason']


def test_stationary_noise_does_not_accumulate_random_walk():
    points = room()
    matcher = L515Odometry()
    matcher.update(points, 10)
    rng = np.random.default_rng(1)
    for i in range(1, 10):
        pose = matcher.update(points + rng.normal(0, 0.001, points.shape), 10+i*0.2)
        assert pose is not None
    assert np.linalg.norm(pose[:3, 3]) < 0.003
    np.testing.assert_array_equal(matcher.reference_pose, np.eye(4))


def test_stale_scan_and_dropouts_do_not_update_accepted_pose():
    points = room()
    matcher = L515Odometry()
    matcher.update(points, 10)
    assert matcher.update(points, 10) is None
    assert matcher.update(points, 13) is not None
    np.testing.assert_allclose(matcher.pose, np.eye(4), atol=1e-6)
    assert matcher.accepted == 2


def test_disjoint_scene_or_large_jump_is_rejected():
    points = room()
    matcher = L515Odometry()
    matcher.update(points, 10)
    assert matcher.update(points + [1, 0, 0], 10.2) is None
    assert matcher.accepted == 1


def test_repeated_loss_starts_separate_segment():
    points = room()
    matcher = L515Odometry()
    matcher.update(points, 10)
    moved = points + [1, 0, 0]
    for i in range(10):
        assert matcher.update(moved, 10.01+i*.01) is None
    assert matcher.update(moved, 10.2) is not None
    assert matcher.segment == 1
    assert matcher.map_frame.endswith('_1')
    np.testing.assert_array_equal(matcher.pose, np.eye(4))
    assert matcher.update(moved, 10.4) is not None


def test_persistent_matcher_retains_origin_after_repeated_loss_and_recovers():
    points=room()
    matcher=L515Odometry(reset_on_loss=False)
    matcher.update(points,10)
    original=matcher.reference
    for i in range(12):
        assert matcher.update(points+[1,0,0],10.01+i*.01) is None
    assert matcher.segment==0 and matcher.reference is original
    assert matcher.update(points,11) is not None
    np.testing.assert_allclose(matcher.pose,np.eye(4),atol=1e-6)


def test_recovers_turn_after_loss_without_new_origin():
    points=room()
    matcher=L515Odometry(reset_on_loss=False)
    matcher.update(points,10)
    matcher.update(points+[3,0,0],10.1)
    matcher.update(points+[3,0,0],10.2)
    R=Rotation.from_euler('y',25,degrees=True).as_matrix()
    pose=matcher.update(points@R,12)
    assert pose is not None, matcher.status()
    assert matcher.segment==0
    assert Rotation.from_matrix(pose[:3,:3]@R.T).magnitude()<.02
    assert np.linalg.norm(pose[:3,3])<.015
