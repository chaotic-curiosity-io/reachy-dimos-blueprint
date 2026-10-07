import numpy as np
from station.l515.reachy_rgbd import matching_map_pose


def sample(session='a', frame=1, pose=None):
    return dict(meta=dict(session_id=session,depth_frame_number=frame),map_pose=pose)


def test_only_exact_sensor_frame_gets_map_pose():
    early=sample()
    np.testing.assert_array_equal(matching_map_pose(early,sample(pose=np.eye(4).tolist())),np.eye(4))
    assert matching_map_pose(early,sample(frame=2,pose=np.eye(4).tolist())) is None
    assert matching_map_pose(early,sample(session='b',pose=np.eye(4).tolist())) is None
    assert matching_map_pose(early,sample()) is None
