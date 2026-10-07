import numpy as np
import pytest
from reachy_wheels_app.sensors import HEADER
from station.l515.supervised_rgb_follow import frontal_clearance


def payload(points):
    return HEADER.pack(b'RSPC', 1, 1, len(points))+np.asarray(points,dtype='<f4').tobytes()


def test_small_close_obstacle_is_not_hidden_by_percentile():
    cloud = np.tile([0,0,2.0], (1000,1))
    cloud[:3,2] = .4
    assert frontal_clearance(payload(cloud)) < .65


def test_clear_front():
    assert frontal_clearance(payload(np.tile([0,0,2.0], (101,1)))) == 2.0


def test_no_coverage_is_not_clear():
    with pytest.raises(ValueError):
        frontal_clearance(payload(np.tile([1,0,2.0], (101,1))))
