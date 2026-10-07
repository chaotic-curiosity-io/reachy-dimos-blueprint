import unittest
import time
import struct
import numpy as np
from realsense_viewer.server import FrameStore, color_bmp

class CalibrationTests(unittest.TestCase):
    def test_bmp_pixels_and_padding(self):
        rgb=np.array([[[255,0,0]],[[0,255,0]]],dtype=np.uint8)
        bmp=color_bmp(rgb)
        self.assertEqual(bmp[:2],b'BM')
        self.assertEqual(struct.unpack_from('<I',bmp,2)[0],len(bmp))
        self.assertEqual(bmp[54:],b'\x00\xff\x00\x00\x00\x00\xff\x00')

    def test_bundle_copies_capture_buffers_and_expires(self):
        store=FrameStore()
        rgb=np.zeros((2,2,3),dtype=np.uint8);depth=np.ones((2,2),dtype=np.uint16)
        store.publish_calibration(rgb,depth,{'session_id':'test'})
        rgb[:]=255;depth[:]=0
        c,d,m=store.calibration_snapshot()
        self.assertEqual(int(c.max()),0)
        self.assertEqual(int(d.min()),1)
        store._calibration_at=time.monotonic()-3
        self.assertIsNone(store.calibration_snapshot())

    def test_mapping_bundle_downsamples_only_color_and_keeps_depth(self):
        import io,json,threading,zipfile
        from http.server import ThreadingHTTPServer
        from urllib.request import urlopen
        from realsense_viewer.server import ViewerHandler
        server=ThreadingHTTPServer(('127.0.0.1',0),ViewerHandler)
        server.frame_store=FrameStore()
        depth=np.arange(240*320,dtype=np.uint16).reshape(240,320)
        color=np.zeros((480,640,3),np.uint8)
        server.frame_store.publish_calibration(color,depth,{'session_id':'test','color_intrinsics':{'width':640,'height':480}})
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/mapping-frame') as r:
                with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                    metadata=json.loads(z.read('metadata.json'))
                    self.assertEqual(metadata['color_pixel_stride'],4)
                    self.assertEqual(metadata['color_intrinsics']['width'],640)
                    self.assertEqual(z.read('depth.u16'),depth.astype('<u2').tobytes())
                    self.assertEqual(struct.unpack_from('<ii',z.read('color.bmp'),18),(160,120))
            self.assertEqual(server.frame_store.calibration_snapshot()[0].shape,(480,640,3))
        finally:
            server.shutdown();server.server_close();worker.join()
