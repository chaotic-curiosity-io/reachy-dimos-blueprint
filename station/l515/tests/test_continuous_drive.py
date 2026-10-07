from types import SimpleNamespace
import pytest
from station.l515.continuous_drive import ContinuousDrive, DriveObservation


def fixture():
    clock = [100.0]
    calls = []
    board = SimpleNamespace(config=SimpleNamespace(retries=1),
        move=lambda **kw: calls.append(('move', kw)), stop=lambda: calls.append(('stop', {})))
    drive = ContinuousDrive(board, 'map', clock=lambda: clock[0], wall_clock=lambda: clock[0])
    return drive, clock, calls


def observed(now, **kwargs):
    return DriveObservation(**dict(source_time=now, segment='map', tracking=True,
                                  clearance_m=1.5, path_clear=True, **kwargs))


def test_continuous_refresh_has_no_intermediate_stop_and_ends_at_deadline():
    drive, clock, calls = fixture()
    for i in range(100):
        clock[0] = 100 + i * .2
        assert drive.tick(observed(clock[0]))
    assert calls and all(c[0] == 'move' for c in calls)
    assert all(0 < c[1]['duration'] <= .8 for c in calls)
    clock[0] = 120
    assert not drive.tick(observed(clock[0]))
    assert calls[-1][0] == 'stop'


@pytest.mark.parametrize('change', [dict(source_time=98), dict(source_time=101),
    dict(segment='other'), dict(tracking=False), dict(clearance_m=.5),
    dict(clearance_m=float('nan')), dict(path_clear=False)])
def test_invalid_observation_latches_stop(change):
    drive, clock, calls = fixture()
    obs = dict(source_time=100, segment='map', tracking=True, clearance_m=1.5, path_clear=True)
    obs.update(change)
    assert not drive.tick(DriveObservation(**obs))
    assert not drive.tick(observed(100))
    assert [c[0] for c in calls] == ['stop']


def test_command_failure_never_retries_or_rearms():
    drive, clock, calls = fixture()
    def fail(**kwargs):
        calls.append(('failed', kwargs))
        raise TimeoutError('lost response')
    drive.board.move = fail
    assert not drive.tick(observed(100))
    assert not drive.tick(observed(100))
    assert [c[0] for c in calls] == ['failed', 'stop']


def test_slow_command_stops():
    drive, clock, calls = fixture()
    drive.board.move = lambda **kw: clock.__setitem__(0, clock[0] + .55)
    assert not drive.tick(observed(100))
    assert drive.stopped and calls[-1][0] == 'stop'


def test_refresh_gap_latches_stop_instead_of_resuming_motion():
    drive, clock, calls = fixture()
    assert drive.tick(observed(100))
    clock[0] = 100.7
    assert not drive.tick(observed(100.7))
    assert drive.reason == 'command refresh deadline missed'
