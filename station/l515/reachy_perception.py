"""Live DimOS YOLO recognition and camera-only observation history. No actuation."""
import argparse
import json
from pathlib import Path
import signal
import sqlite3
import threading
import time
import uuid


def open_memory(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, observed_at REAL, payload TEXT)')
    return db


def remember(db, report):
    """Retain recent observations, not inferred persistent physical objects."""
    with db:
        db.execute('INSERT INTO observations VALUES (?, ?, ?)',
                   (report['observation_id'], report['source_received_at_unix'], json.dumps(report)))
        db.execute('DELETE FROM observations WHERE id IN (SELECT id FROM observations ORDER BY observed_at DESC LIMIT -1 OFFSET 10000)')


def recall(path, name=None, limit=50):
    if not Path(path).exists():
        return []
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT payload FROM observations ORDER BY observed_at DESC LIMIT 10000')
        found = []
        for (payload,) in rows:
            report = json.loads(payload)
            if name and not any(d['name'].casefold() == name.casefold() for d in report['objects']):
                continue
            found.append(report)
            if len(found) >= max(1, min(100, limit)):
                break
        return found


def atomic_json(path, report):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(path)


def main():
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.msgs.vision_msgs.Detection2DArray import Detection2DArray
    from dimos.perception.detection.detectors.yolo import Yolo2DDetector

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--hz', type=float, default=2)
    args = p.parse_args()
    if not 0 < args.hz <= 10:
        p.error('--hz must be > 0 and <= 10')
    out = args.directory.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = (args.model or out / 'models/yolo11n.pt').resolve()
    if not model.is_file():
        p.error(f'Model weights not found: {model}; see REACHY_PERCEPTION.md')
    detector = Yolo2DDetector(model_path=str(model.parent), model_name=model.name, device=args.device)
    db = open_memory(out / 'observations.sqlite3')
    source = LCMTransport('/reachy/color_image', Image)
    detections = LCMTransport('/reachy/perception/detections2d', Detection2DArray)
    annotated = LCMTransport('/reachy/perception/annotated_image', Image)
    lock = threading.Lock()
    pending = None
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    def receive(image):
        nonlocal pending
        with lock:
            pending = image
    unsubscribe = source.subscribe(receive)
    last_memory = 0
    print(f'Recognition ready: {model.name}, {args.device}', flush=True)
    try:
        while not stop.wait(1 / args.hz):
            with lock:
                image, pending = pending, None
            if image is None or not 0 <= time.time() - image.ts < 2:
                continue
            start = time.monotonic()
            result = detector.process_image(image)
            message = result.to_ros_detection2d_array()
            message.header.frame_id = image.frame_id
            for detection in message.detections:
                detection.header.frame_id = image.frame_id
            detections.publish(message)
            preview = result.annotated_image()
            preview.frame_id = image.frame_id
            annotated.publish(preview)
            report = dict(observation_id=str(uuid.uuid4()), state='running',
                reported_at_unix=time.time(), source_received_at_unix=image.ts,
                timestamp_basis='bridge receipt, not hardware capture',
                frame_id=image.frame_id, model=model.name,
                inference_ms=round((time.monotonic()-start)*1000, 1),
                coordinate_space='image_pixels', position_3d=None,
                objects=[dict(name=d.name, class_id=int(d.class_id), confidence=float(d.confidence),
                              bbox_xyxy=[float(v) for v in d.bbox]) for d in result])
            atomic_json(out / 'perception-live.json', report)
            if time.monotonic() - last_memory >= 5:
                remember(db, report)
                last_memory = time.monotonic()
    finally:
        unsubscribe()
        for transport in (source, detections, annotated):
            transport.stop()
        detector.stop()
        db.close()
        atomic_json(out / 'perception-live.json', dict(state='stopped', reported_at_unix=time.time(), objects=[]))


if __name__ == '__main__':
    main()
