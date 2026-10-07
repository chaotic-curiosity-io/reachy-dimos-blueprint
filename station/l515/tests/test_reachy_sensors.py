"""Native DimOS message/mapper tests; run in the DimOS Python environment."""
import struct
import threading
import time

import numpy as np
import pytest

pytest.importorskip('dimos')
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from station.l515.reachy_sensors import ReachySensors
from reachy_wheels_app.sensors import HEADER, DEPTH_FRAME

# RFC 5737 documentation address: never contacted (depth frames are monkeypatched).
TEST_DEPTH_URL = 'http://192.0.2.20:8765'


class CaptureTransport:
    def __init__(self):
        self.messages = []
    def broadcast(self, _, message):
        self.messages.append(message)
    def stop(self):
        pass


def test_native_cloud_units_axes_dedup_and_local_map(monkeypatch, tmp_path, request):
    module = ReachySensors(depth_url=TEST_DEPTH_URL)
    request.addfinalizer(module.stop)
    transport = CaptureTransport()
    module.lidar.transport = transport
    points = np.array([[0.1, -0.2, 1.5], [0.2, 0.3, 2]], dtype='<f4')
    payload = HEADER.pack(b'RSPC', 1, 1, 2) + points.tobytes()
    monkeypatch.setattr(module._depth, 'frame', lambda: (payload, {
        'sequence':1, 'received_at_unix':1234, 'point_count':2}))
    module.poll_depth()
    module.poll_depth()
    assert len(transport.messages) == 1
    cloud = transport.messages[0]
    assert cloud.frame_id == DEPTH_FRAME and cloud.ts == 1234
    np.testing.assert_allclose(np.asarray(cloud.pointcloud.points), points)
    path = tmp_path / 'local.ply'
    module.export_local_scan(path)
    import open3d as o3d
    assert len(o3d.io.read_point_cloud(str(path)).points) == 2



def test_lcm_serialization_roundtrip():
    received = []
    event = threading.Event()
    transport = LCMTransport('/reachy/test_sensor_' + str(time.time_ns()), PointCloud2)
    unsubscribe = transport.subscribe(lambda message: (received.append(message), event.set()))
    try:
        cloud = PointCloud2.from_numpy(np.array([[1, 2, 3]], dtype=np.float32), frame_id=DEPTH_FRAME, timestamp=time.time())
        transport.publish(cloud)
        assert event.wait(3), 'LCM did not deliver the sensor message'
        assert received[0].frame_id == DEPTH_FRAME
        np.testing.assert_allclose(np.asarray(received[0].pointcloud.points), [[1, 2, 3]])
    finally:
        unsubscribe()
        transport.stop()


def test_cmd_vel_is_dry_run_even_with_fresh_sensors(request):
    module = ReachySensors(depth_url=TEST_DEPTH_URL)
    request.addfinalizer(module.stop)
    result = module.preview_twist(Twist(linear=(0.2, -0.1, 0), angular=(0, 0, 0.3)))
    assert result == {'vx_m_s':0.2, 'vy_m_s':-0.1, 'omega_rad_s':0.3, 'actuated':False}
    assert not module.snapshot()['actuation_enabled']
    with pytest.raises(ValueError):
        module.preview_twist(Twist(linear=(float('nan'), 0, 0)))



def test_module_lifecycle_keeps_rgb_running_when_depth_fails(monkeypatch):
    module = ReachySensors(depth_url=TEST_DEPTH_URL, rgb_hz=20, depth_hz=20)
    from dimos.msgs.sensor_msgs.Image import Image
    module.lidar.transport = LCMTransport('/reachy/test/lifecycle/depth', PointCloud2)
    module.color_image.transport = LCMTransport('/reachy/test/lifecycle/rgb', Image)
    module.cmd_vel.transport = LCMTransport('/reachy/test/lifecycle/cmd', Twist)
    event = threading.Event()
    def broken_depth():
        raise OSError('Pi disconnected')
    def rgb():
        module._observed('rgb', {})
        event.set()
    monkeypatch.setattr(module, 'poll_depth', broken_depth)
    monkeypatch.setattr(module, 'poll_rgb', rgb)
    try:
        module.start()
        assert event.wait(3)
        snapshot = module.snapshot()
        assert snapshot['depth']['errors'] > 0
        assert snapshot['rgb']['frames'] > 0
    finally:
        module.stop()
    assert all(not thread.is_alive() for thread in module._threads)
