"""Bounded exact-frame join between asynchronous perception and mapping."""
import numpy as np


def frame_key(meta):
    return (meta['session_id'], meta['depth_frame_number'])


def join_observation(report, poses):
    """Return world camera pose only for the observation's accepted depth frame."""
    if report.get('rgbd_state') != 'preview':
        return None
    key = tuple(report.get('depth_frame_key', ()))
    entry = poses.get(key)
    transform = report.get('l515_depth_to_reachy_optical')
    if entry is None or transform is None:
        return None
    segment, pose = entry
    return segment, np.asarray(pose) @ np.linalg.inv(np.asarray(transform))
