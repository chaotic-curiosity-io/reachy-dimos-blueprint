"""RemoteDetector against a real HTTP server on a real socket.

Proves the whole robot-side leg of the open-vocabulary path: JPEG encode,
the targets query string, the reply parse, and — the part that matters for
safety — what happens when the service is slow, broken, or gone.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from reachy_wheels_app.tracking.detect_remote import RemoteDetector, encode_jpeg
from reachy_wheels_app.tracking.detectors import DetectorUnavailable

FRAME = np.full((480, 640, 3), 120, dtype=np.uint8)


class Recorder(BaseHTTPRequestHandler):
    payload = {"detections": [{"bbox": [10, 20, 110, 220], "score": 0.82,
                               "class_name": "red mug"}]}
    status = 200
    seen: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        Recorder.seen.append({"path": self.path, "body": body,
                              "content_type": self.headers.get("Content-Type")})
        self.send_response(Recorder.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(Recorder.payload).encode())

    def log_message(self, *args):
        return


@pytest.fixture()
def service():
    Recorder.seen = []
    Recorder.status = 200
    server = HTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/detect", Recorder
    server.shutdown()
    server.server_close()


def test_a_frame_round_trips_into_detections(service):
    url, recorder = service
    out = RemoteDetector(url).detect(FRAME, targets=("red mug",))
    assert len(out) == 1
    assert out[0].class_name == "red mug"
    assert out[0].bbox == (10.0, 20.0, 110.0, 220.0)

    sent = recorder.seen[0]
    assert sent["content_type"] == "image/jpeg"
    assert sent["body"].startswith(b"\xff\xd8")        # a real JPEG
    assert "targets=red+mug" in sent["path"]


def test_multiple_targets_become_one_prompt(service):
    url, recorder = service
    RemoteDetector(url).detect(FRAME, targets=("red mug", "laptop"))
    assert "red+mug%2C+laptop" in recorder.seen[0]["path"]


def test_a_url_that_already_has_a_query_is_extended_not_broken(service):
    url, recorder = service
    RemoteDetector(url + "?api_key=abc").detect(FRAME, targets=("dog",))
    path = recorder.seen[0]["path"]
    assert "api_key=abc" in path and "targets=dog" in path


def test_a_server_error_is_a_dropped_frame_not_a_crash(service):
    url, recorder = service
    recorder.status = 500
    assert RemoteDetector(url).detect(FRAME, targets=("dog",)) == []


def test_an_unreachable_service_degrades_then_gives_up_loudly():
    # A closed port: individual failures are survivable (the follow loop
    # treats them as "briefly out of view"), but a service that is simply
    # gone must eventually say so rather than leaving the robot blind.
    detector = RemoteDetector("http://127.0.0.1:9/detect", timeout=0.2)
    for _ in range(9):
        assert detector.detect(FRAME) == []
    with pytest.raises(DetectorUnavailable) as exc:
        detector.detect(FRAME)
    assert "10 times in a row" in str(exc.value)


def test_a_recovered_service_resets_the_failure_count(service):
    url, _ = service
    detector = RemoteDetector(url)
    detector._consecutive_errors = 9
    assert detector.detect(FRAME)
    assert detector._consecutive_errors == 0


def test_encode_jpeg_produces_a_real_jpeg():
    data = encode_jpeg(FRAME)
    assert data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


def test_a_junk_frame_never_reaches_the_wire(service):
    url, recorder = service
    assert RemoteDetector(url).detect(None) == []
    assert recorder.seen == []
