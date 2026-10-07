import json
import math
import struct

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_wheels_app import api
from reachy_wheels_app.sensors import DepthSource, HEADER, unpack_points, validate_url


def packet(sequence=1, points=((0.1, -0.2, 1.5),)):
    return HEADER.pack(b'RSPC', 1, sequence, len(points)) + b''.join(
        struct.pack('<fff', *point) for point in points)


def status(sequence=1, age=5, state='streaming'):
    return json.dumps({'sequence': sequence, 'last_frame_age_ms': age, 'state': state}).encode()


# Documentation-range address (RFC 5737); nothing is ever contacted.
PI_URL = 'http://192.0.2.20:8765'


def source_with(monkeypatch, replies):
    source = DepthSource(PI_URL)
    calls = iter(replies)
    monkeypatch.setattr(source, '_get', lambda *args: next(calls))
    return source


def test_wire_metres_and_metadata(monkeypatch):
    source = source_with(monkeypatch, [status(), packet()])
    payload, meta = source.frame()
    assert unpack_points(payload) == (1, 1)
    assert meta['axes'] == 'x-right,y-down,z-forward'
    assert meta['timestamp_kind'] == 'receiver_time_not_capture_time'


@pytest.mark.parametrize('payload', [b'', packet()[:-1], packet()+b'x',
    packet(points=((math.nan, 0, 1),)), packet(points=((0, 0, -1),)),
    HEADER.pack(b'RSPC', 2, 1, 0), HEADER.pack(b'RSPC', 1, 1, 99999999)])
def test_malformed_cloud_rejected(payload):
    with pytest.raises(ValueError):
        unpack_points(payload)


@pytest.mark.parametrize('age', [None, -1, 501, math.nan, math.inf, True])
def test_stale_or_invalid_age_rejected(monkeypatch, age):
    source = source_with(monkeypatch, [status(age=age), packet()])
    with pytest.raises(ValueError, match='stale'):
        source.frame()


def test_reconnect_cannot_relabel_cached_frame(monkeypatch):
    source = source_with(monkeypatch, [status(state='reconnecting'), packet()])
    with pytest.raises(ValueError):
        source.frame()


def test_sequence_advancement_is_accepted(monkeypatch):
    source = source_with(monkeypatch, [status(2), packet(3)])
    assert source.frame()[1]['sequence'] == 3


def test_unmatched_sequence_never_published(monkeypatch):
    source = source_with(monkeypatch, [status(2), packet(1)])
    with pytest.raises(ValueError, match='sequence-matched'):
        source.frame()


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'http://user:pass@host',
                                'http://host/path', 'http://host?query=1'])
def test_invalid_origins(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_unset_depth_url_names_the_env_var():
    # No hardcoded Pi address: an unset source fails with an actionable hint.
    with pytest.raises(ValueError, match='DEPTH_SERVER_URL'):
        validate_url('')


def test_sensor_routes_do_not_touch_actuators(monkeypatch, tmp_path):
    monkeypatch.setattr(api.config_store, 'STATE_PATH', tmp_path / 'state.json')
    link = api.WheelsLink({'host':'unused', 'port':80, 'default_speed':0.8,
                           'depth_url': PI_URL})
    link.client = object()  # any accidental chassis call fails
    app = FastAPI()
    api.wire_routes(app, link)
    client = TestClient(app)
    monkeypatch.setattr(DepthSource, 'status', lambda self: json.loads(status()))
    result = client.get('/api/sensors/status').json()
    assert result['depth']['connected']
    assert not result['navigation']['actuation_enabled']
    assert not result['synchronized_rgbd']
    assert client.post('/api/sensors/config', json={'depth_url':'file:///a'}).status_code == 400
    assert client.post('/api/sensors/config', json={'depth_url':'http://new-pi:8765/'}).status_code == 200
    assert link.snapshot()['depth_url'] == 'http://new-pi:8765'
    monkeypatch.setattr(DepthSource, 'frame', lambda self: (packet(), {
        'sequence':1, 'received_at_unix':123, 'timestamp_kind':'receiver', 'age_upper_bound_ms':20}))
    response = client.get('/api/sensors/depth')
    assert response.content == packet()
    assert response.headers['x-frame-id'] == 'realsense_depth_optical'
    def stale(self):
        raise ValueError('stale')
    monkeypatch.setattr(DepthSource, 'frame', stale)
    assert client.get('/api/sensors/depth').status_code == 503
