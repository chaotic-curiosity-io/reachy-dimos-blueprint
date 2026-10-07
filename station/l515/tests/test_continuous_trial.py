import numpy as np
from station.l515.continuous_trial import corridor_clear, initial_floor_patch


def floor_points():
    x, y = np.meshgrid(np.linspace(.35, 1.2, 60), np.linspace(-.4, .4, 60))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -.2)))


def test_observed_flat_corridor_is_clear():
    assert corridor_clear(floor_points(), -.2, np.zeros(2), 0)


def test_small_current_obstacle_blocks_corridor():
    points = np.vstack((floor_points(), [.6, .05, -.13]))
    assert not corridor_clear(points, -.2, np.zeros(2), 0)


def test_missing_near_floor_is_not_invented_as_free():
    points = floor_points()
    assert not corridor_clear(points[points[:, 0] > .7], -.2, np.zeros(2), 0)


def test_empty_scan_is_blocked():
    assert not corridor_clear(np.empty((0, 3)), -.2, np.zeros(2), 0)


def test_operator_initial_floor_stays_fixed_and_never_hides_obstacle():
    points = floor_points()
    points = points[points[:, 0] > .7]
    known = initial_floor_patch(np.zeros(2), 0)
    assert corridor_clear(points, -.2, np.zeros(2), 0, known)
    assert not corridor_clear(np.vstack((points, [.6, 0, -.1])), -.2, np.zeros(2), 0, known)
    assert not corridor_clear(points, -.2, np.array([1., 0]), 0, known)
