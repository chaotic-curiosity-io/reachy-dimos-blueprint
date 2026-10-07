"""Needs the full dimOS environment (see station/README.md); skipped otherwise."""
import pytest
pytest.importorskip('open3d')
pytest.importorskip('rerun')
pytest.importorskip('dimos')
import warnings
import numpy as np
import rerun as rr

from station.l515.l515_rerun import segment_geometry


def test_changing_persistent_geometry_is_not_static(monkeypatch):
    from station.l515.l515_rerun import publish_persistent
    calls=[]
    monkeypatch.setattr(rr,'log',lambda *a,**kw:calls.append((a,kw)))
    track=dict(center_m=[0,0,1],size_m=[1,1,1],name='chair',id='abc123')
    publish_persistent('remembered/test',np.zeros((1,3)),np.zeros((1,3),np.uint8),[track])
    assert len(calls)==3
    assert all(not kw.get('static',False) for _,kw in calls)
    assert calls[0][0][0]=='remembered/test/cloud'


def test_persistent_display_removes_isolated_noise_without_mutating_raw():
    from station.l515.l515_rerun import clean_persistent_display
    x, y = np.meshgrid(np.arange(8)*.04, np.arange(8)*.04)
    surface = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    raw = np.vstack((surface, [8, 8, 8], [np.nan, 0, 1]))
    colors = np.tile([20, 60, 90], (len(raw), 1)).astype(np.uint8)
    before = raw.copy()
    p, c = clean_persistent_display(raw, colors)
    assert len(p) == len(surface)
    assert np.array_equal(c, colors[:len(surface)])
    assert np.array_equal(raw, before, equal_nan=True)


def test_persistent_display_empty():
    from station.l515.l515_rerun import clean_persistent_display
    p, c = clean_persistent_display(np.empty((0, 3)), np.empty((0, 3)))
    assert p.shape == c.shape == (0, 3)


def test_live_blueprint_serializes_with_active_nested_container():
    from station.l515.l515_rerun import ACTIVE_MAP_ROOT, make_blueprint
    recording=rr.RecordingStream('blueprint-test')
    for segments in ([], ['test-segment']):
        blueprint=make_blueprint(segments)
        assert blueprint.root_container.active_tab == 9
        blueprint.root_container._log_to_stream(recording)
    assert ACTIVE_MAP_ROOT == 'remembered/active'


def test_geometry_boxes_and_voxel_archetypes():
    rng=np.random.default_rng(4)
    points=np.vstack([rng.normal([0,0,1],.04,(160,3)),rng.normal([.7,0,1],.04,(160,3))])
    colors,boxes=segment_geometry(points)
    assert colors.shape==points.shape
    assert len(boxes)==2
    assert all(b[2].startswith('geometry cluster') for b in boxes)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rr.Boxes3D(centers=points,half_sizes=[.019]*3,fill_mode=rr.components.FillMode.Solid)
        rr.Boxes3D(centers=[b[0] for b in boxes],half_sizes=[b[1] for b in boxes],
                   fill_mode=rr.components.FillMode.MajorWireframe)


def test_snapshot_status_never_calls_old_or_blocked_geometry_live():
    from station.l515.l515_rerun import snapshot_status
    snap=dict(source_received_at_unix=100,projected_points=52)
    live=dict(state='running',rgbd_state='preview',reported_at_unix=101)
    assert 'Updating' in snapshot_status(snap,live,102)
    assert 'FROZEN' in snapshot_status(snap,live,110)
    assert 'FROZEN' in snapshot_status(snap,dict(state='unavailable'),102)
    assert 'Waiting' in snapshot_status(None,live,102)
