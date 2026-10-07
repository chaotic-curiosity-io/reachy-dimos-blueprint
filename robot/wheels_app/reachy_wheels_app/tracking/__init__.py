"""Visual target tracking and following for the Wheels chassis.

Pipeline: head camera → detector (on-robot ONNX, or a remote open-vocabulary
service) → Roboflow ``trackers`` for stable ids → ``FollowController`` →
head gaze + a continuous ``move(vx, omega)`` mix on the chassis.

Layered so the interesting part is testable without a robot:

    types/vocab   plain data, phrase → detector labels
    detectors     backend protocol; detect_onnx / detect_remote
    tracker       Roboflow trackers, with a greedy-IoU fallback
    follow        the control law — pure, no clock, no IO
    session       threads, actuation, status board, lifecycle

See the "Follow" and "Setting up following" sections of the app README.
"""

from __future__ import annotations

from .detectors import BACKENDS, DetectorUnavailable, build_detector
from .follow import (
    ARRIVED,
    FOLLOWING,
    LOST,
    SEARCHING,
    FollowCommand,
    FollowConfig,
    FollowController,
)
from .session import FollowManager, FollowSession, TrackBoard
from .tracker import TargetTracker
from .types import Detection, FrameInfo
from .vocab import COCO_CLASSES, resolve_target

__all__ = [
    "ARRIVED", "BACKENDS", "COCO_CLASSES", "Detection", "DetectorUnavailable",
    "FOLLOWING", "FollowCommand", "FollowConfig", "FollowController",
    "FollowManager", "FollowSession", "FrameInfo", "LOST", "SEARCHING",
    "TargetTracker", "TrackBoard", "build_detector", "resolve_target",
]
