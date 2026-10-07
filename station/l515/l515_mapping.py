"""Experimental metric L515 scan matching into DimOS voxel mapping.

No base extrinsics are assumed. The first optical camera frame is the map
origin; the estimated pose belongs to the L515, never to Reachy's wheel base.
There is no motor client, gravity alignment, loop closure, or goal execution.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import signal
import shutil
import threading
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from dimos.core.transport import LCMTransport
from dimos.mapping.voxels import VoxelGrid
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

MAP_FRAME = 'l515_start_optical'


@dataclass(frozen=True)
class MatchLimits:
    voxel_m: float = 0.04
    min_points: int = 300
    min_fitness: float = 0.60
    max_rmse_m: float = 0.025
    min_geometry_ratio: float = 0.0001
    max_translation_m_s: float = 0.30
    max_rotation_rad_s: float = 0.8
    max_gap_s: float = 2.0


class L515Odometry:
    """Conservative scan-to-keyframe ICP with explicit rejected-frame state.

    A keyframe stays fixed until sufficient motion occurs. Stationary noise
    therefore doesn't integrate into a random walk on every received frame.
    """
    def __init__(self, limits=None, *, reset_on_loss=True):
        self.limits = limits or MatchLimits()
        self.reset_on_loss = reset_on_loss
        self.reference = None
        self.reference_pose = np.eye(4)
        self.pose = np.eye(4)
        self.last_ts = None
        self.accepted = 0
        self.rejected = 0
        self.last = {'state': 'waiting', 'reason': None}
        self.segment = 0
        self.consecutive_rejections = 0

    def _prepare(self, xyz):
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('expected Nx3 XYZ')
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.15)
                        & (points[:, 2] <= 4.1)]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return cloud.voxel_down_sample(self.limits.voxel_m)

    def _reject(self, reason, **quality):
        self.rejected += 1
        self.consecutive_rejections += 1
        self.last = {'state': 'lost', 'reason': reason, **quality}
        return None

    @staticmethod
    def geometry_ratio(result, source, target):
        pairs = np.asarray(result.correspondence_set)
        if len(pairs) < 6:
            return 0.0
        p = np.asarray(source.points)[pairs[:, 0]]
        p = p @ result.transformation[:3, :3].T + result.transformation[:3, 3]
        n = np.asarray(target.normals)[pairs[:, 1]]
        centered = p - p.mean(axis=0)
        scale = max(0.25, float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))))
        jacobian = np.column_stack((np.cross(centered / scale, n), n))
        values = np.linalg.eigvalsh(jacobian.T @ jacobian / len(pairs))
        return float(max(0, values[0]) / max(1e-12, values[-1]))

    def update(self, xyz, timestamp):
        if not math.isfinite(timestamp):
            return self._reject('invalid timestamp')
        cloud = self._prepare(xyz)
        if len(cloud.points) < self.limits.min_points:
            return self._reject('insufficient depth geometry', points=len(cloud.points))
        if self.reset_on_loss and self.consecutive_rejections >= 10:
            # Start a NEW coordinate frame; never merge unregistered geometry.
            self.segment += 1
            self.reference = None
            self.reference_pose = np.eye(4)
            self.pose = np.eye(4)
            self.consecutive_rejections = 0
        if self.reference is None:
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.16, max_nn=30))
            self.reference = cloud
            self.last_ts = timestamp
            self.accepted += 1
            self.last = {'state': 'anchored', 'reason': 'first scan defines the map origin'}
            return self.pose.copy()
        dt = timestamp - self.last_ts
        if dt <= 0:
            return self._reject('out-of-order timestamp', gap_s=dt)
        # After a gap, try matching the retained keyframe with a bounded jump.
        dt = min(dt, self.limits.max_gap_s)
        predicted = np.linalg.inv(self.reference_pose) @ self.pose
        reg = o3d.pipelines.registration
        result = None
        for distance, iterations in ((0.12, 20), (0.07, 15), (0.04, 15)):
            result = reg.registration_icp(
                cloud, self.reference, distance, predicted,
                reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=distance)),
                reg.ICPConvergenceCriteria(max_iteration=iterations))
            predicted = result.transformation
        if (self.consecutive_rejections >= 2 and result.fitness < self.limits.min_fitness):
            # A short chassis turn can exceed the local ICP basin between
            # network samples. Search bounded optical-yaw hypotheses against
            # the retained reference, with the same final residual/geometry gates.
            initial=np.linalg.inv(self.reference_pose) @ self.pose
            candidates=[result]
            for angle in (-60,-45,-30,-15,15,30,45,60):
                seed=initial.copy()
                seed[:3,:3]=initial[:3,:3] @ Rotation.from_euler('y',angle,degrees=True).as_matrix()
                trial=reg.registration_icp(cloud,self.reference,.15,seed,
                    reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=.15)),
                    reg.ICPConvergenceCriteria(max_iteration=25))
                trial=reg.registration_icp(cloud,self.reference,.04,trial.transformation,
                    reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=.04)),
                    reg.ICPConvergenceCriteria(max_iteration=20))
                if trial.inlier_rmse <= self.limits.max_rmse_m:
                    candidates.append(trial)
            result=max(candidates,key=lambda candidate:candidate.fitness)
        ratio = self.geometry_ratio(result, cloud, self.reference)
        quality = {'fitness': float(result.fitness), 'rmse_m': float(result.inlier_rmse),
                   'geometry_ratio': ratio, 'correspondences': len(result.correspondence_set)}
        candidate = self.reference_pose @ result.transformation
        delta = np.linalg.inv(self.pose) @ candidate
        translation = float(np.linalg.norm(delta[:3, 3]))
        rotation = float(Rotation.from_matrix(delta[:3, :3]).magnitude())
        quality.update(translation_m=translation, rotation_deg=math.degrees(rotation))
        if (result.fitness < self.limits.min_fitness
                or result.inlier_rmse > self.limits.max_rmse_m
                or len(result.correspondence_set) < self.limits.min_points):
            return self._reject('poor scan overlap or residual', **quality)
        if ratio < self.limits.min_geometry_ratio:
            return self._reject('geometry is insufficient to constrain 6D motion', **quality)
        if (translation > self.limits.max_translation_m_s * dt + 0.01
                or rotation > self.limits.max_rotation_rad_s * dt + math.radians(1)):
            return self._reject('pose jump exceeds low-speed test envelope', **quality)
        self.consecutive_rejections = 0
        self.pose = candidate
        self.last_ts = timestamp
        self.accepted += 1
        self.last = {'state': 'tracking', 'reason': None, **quality}
        # Hold a fixed reference for small motions; replace it before overlap
        # gets weak. All accepted clouds can still contribute to the voxel map.
        from_reference = np.linalg.inv(self.reference_pose) @ candidate
        if (np.linalg.norm(from_reference[:3, 3]) > 0.08 or
                Rotation.from_matrix(from_reference[:3, :3]).magnitude() > math.radians(8)):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.16, max_nn=30))
            self.reference = cloud
            self.reference_pose = candidate.copy()
        return candidate.copy()

    @property
    def map_frame(self):
        return f'{MAP_FRAME}_{self.segment}'

    def status(self):
        return {**self.last, 'accepted': self.accepted, 'rejected': self.rejected,
                'sensor_pose_matrix': self.pose.tolist(), 'map_frame': self.map_frame,
                'segment': self.segment, 'consecutive_rejections': self.consecutive_rejections,
                'pose_is_wheel_base': False, 'extrinsics_calibrated': False,
                'limits': asdict(self.limits)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True, help='PLY map, saved every 5 s and on exit')
    parser.add_argument('--poses', type=Path, help='JSONL accepted poses and rejection diagnostics')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds must be finite and nonnegative')
    halt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: halt.set())
    signal.signal(signal.SIGINT, lambda *_: halt.set())
    mutex = threading.Lock()
    pending = None
    overwritten = 0
    def receive(cloud):
        nonlocal pending, overwritten
        with mutex:
            if pending is not None:
                overwritten += 1
            pending = (cloud, time.monotonic())
    source = LCMTransport('/reachy/lidar', PointCloud2)
    poses = LCMTransport('/reachy/experimental/sensor_odom', PoseStamped)
    registered = LCMTransport('/reachy/experimental/registered_cloud', PointCloud2)
    maps = LCMTransport('/reachy/experimental/global_map', PointCloud2)
    unsubscribe = source.subscribe(receive)
    odometry = L515Odometry(reset_on_loss=False)
    voxels = VoxelGrid(voxel_size=0.04, block_count=200_000, device='CPU:0',
                       carve_columns=False, frame_id=odometry.map_frame)
    started = time.monotonic()
    last_saved = started
    last_receive = None
    last_report = 0
    last_processing_ms = None
    active_segment = 0
    pose_log = args.poses.open('a') if args.poses else None
    def report(running=True):
        age = None if last_receive is None else time.monotonic() - last_receive
        value = {**odometry.status(), 'running': running, 'reported_at_unix': time.time(),
                 'source_age_s': age, 'processing_ms': last_processing_ms,
                 'dropped_pending_scans': overwritten, 'voxels': voxels.size(),
                 'actuation_enabled': False, 'gravity_aligned': False,
                 'loop_closure': False, 'experimental': True}
        if age is not None and age > 2:
            value.update(state='stale', reason='no recent L515 input')
        return value
    def save_map():
        if voxels.size():
            cloud = voxels.get_global_pointcloud2()
            temp = args.map.with_name(args.map.stem + '.tmp.ply')
            if not o3d.io.write_point_cloud(str(temp), cloud.pointcloud):
                raise OSError('could not save map')
            temp.replace(args.map)
            maps.publish(cloud)
    try:
        while not halt.is_set() and (not args.seconds or time.monotonic() - started < args.seconds):
            with mutex:
                item, pending = pending, None
            if item is None:
                halt.wait(0.02)
            else:
                cloud, arrived = item
                last_receive = arrived
                if time.monotonic() - arrived <= 0.5:
                    processing_started = time.monotonic()
                    xyz = np.asarray(cloud.pointcloud.points)
                    pose = odometry.update(xyz, cloud.ts)
                    if odometry.segment != active_segment:
                        save_map()
                        if args.map.exists():
                            archive = args.map.with_name(f'{args.map.stem}-segment-{active_segment}-{time.time_ns()}.ply')
                            shutil.copy2(args.map, archive)
                        voxels.dispose()
                        voxels = VoxelGrid(voxel_size=0.04, block_count=200_000,
                            device='CPU:0', carve_columns=False, frame_id=odometry.map_frame)
                        active_segment = odometry.segment
                    if pose is not None:
                        world = xyz @ pose[:3, :3].T + pose[:3, 3]
                        aligned = PointCloud2.from_numpy(world, frame_id=odometry.map_frame, timestamp=cloud.ts)
                        voxels.add_frame(aligned)
                        registered.publish(aligned)
                        poses.publish(PoseStamped(ts=cloud.ts, frame_id=odometry.map_frame,
                            position=pose[:3, 3].tolist(),
                            orientation=Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()))
                    last_processing_ms = (time.monotonic() - processing_started) * 1000
                    if pose_log:
                        pose_log.write(json.dumps({'ts': cloud.ts, **odometry.status()}) + '\n')
                        pose_log.flush()
            now = time.monotonic()
            if now - last_saved >= 5:
                save_map()
                last_saved = now
            if now - last_report >= 0.5:
                write_json(args.report, report())
                last_report = now
    finally:
        save_map()
        write_json(args.report, report(False))
        if pose_log:
            pose_log.close()
        unsubscribe()
        for transport in (source, poses, registered, maps):
            transport.stop()
        voxels.dispose()
    print(json.dumps(report_summary := json.loads(args.report.read_text()), indent=2))
    if report_summary['accepted'] < 2:
        raise SystemExit('No motion registration accepted beyond the initial anchor')


if __name__ == '__main__':
    main()
