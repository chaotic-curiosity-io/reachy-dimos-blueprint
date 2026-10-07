import numpy as np
from station.l515.semantic_handoff import join_observation
from station.l515.l515_stack import stalled_publisher


def test_delayed_exact_frame_join():
    report=dict(rgbd_state='preview',depth_frame_key=['session',4],
                l515_depth_to_reachy_optical=np.eye(4).tolist())
    poses={('session',5):('map',np.eye(4))}
    assert join_observation(report,poses) is None
    poses[('session',4)]=('map',np.eye(4))
    segment,pose=join_observation(report,poses)
    assert segment=='map'
    np.testing.assert_array_equal(pose,np.eye(4))
    report['depth_frame_key']=['other-session',4]
    assert join_observation(report,poses) is None
    report.update(depth_frame_key=['session',4],rgbd_state='blocked')
    assert join_observation(report,poses) is None


def test_http_failure_does_not_kill_fresh_publisher(tmp_path):
    assert stalled_publisher(tmp_path,100)
    (tmp_path/'rerun-heartbeat.json').write_text('{"reported_at_unix": 99}')
    assert not stalled_publisher(tmp_path,100)
    assert stalled_publisher(tmp_path,131)
