import struct
import unittest

import numpy as np

from realsense_viewer.server import FRAME_HEADER, FrameStore, pack_points


class FrameProtocolTests(unittest.TestCase):
    def test_binary_frame_header_and_payload(self):
        points = np.array([[1, 2, 3], [-1, -2, 4.5]], dtype=np.float32)
        payload = pack_points(points, 17)
        magic, version, sequence, count = FRAME_HEADER.unpack_from(payload)
        self.assertEqual((magic, version, sequence, count), (b"RSPC", 1, 17, 2))
        unpacked = np.frombuffer(payload, dtype="<f4", offset=FRAME_HEADER.size).reshape(-1, 3)
        np.testing.assert_allclose(unpacked, points)

    def test_store_reports_frame_age(self):
        store = FrameStore()
        store.publish(struct.pack("<4sIII", b"RSPC", 1, 1, 0), 0, 1)
        frame, status = store.snapshot()
        self.assertIsNotNone(frame)
        self.assertEqual(status["state"], "streaming")
        self.assertIsNotNone(status["last_frame_age_ms"])


if __name__ == "__main__":
    unittest.main()
