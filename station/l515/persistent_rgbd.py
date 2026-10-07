"""Durable, independently framed RGB voxel segments and conservative object tracks.

Only accepted poses enter the map. Each process starts a separate segment;
archives remain inspectable and are never implicitly registered together.
"""
import json
import time
import uuid
from pathlib import Path
import numpy as np


def depth_points(depth, metadata):
    intr = metadata['depth_intrinsics']
    if intr['model'] != 'distortion.none':
        raise ValueError('Unsupported depth distortion')
    scale = float(metadata['depth_scale_m'])
    if not np.isfinite(scale) or not 0 < scale < .01:
        raise ValueError('Invalid depth scale')
    v, u = np.indices(depth.shape)
    z = depth * scale
    good = (z >= .15) & (z <= 4)
    return np.column_stack(((u[good]-intr['ppx'])*z[good]/intr['fx'],
                            (v[good]-intr['ppy'])*z[good]/intr['fy'], z[good]))


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


class PersistentRGBD:
    def __init__(self, directory, voxel_m=.04, max_voxels=200_000):
        self.directory = Path(directory) / 'colored-map-segments'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment = uuid.uuid4().hex
        self.created = time.time()
        self.voxel_m = voxel_m
        self.max_voxels = max_voxels
        self.voxels = {}
        self.objects = []
        self.last_frame = None
        self.frames = 0
        self.updated = None
        self.revision = None
        self.dirty = False
        self.contradictions = {}
        self.last_cleanup_frame = None
        self.pruned = 0

    @classmethod
    def restore(cls, directory, segment):
        store=cls(directory)
        with np.load(store.directory/(segment+'.npz'),allow_pickle=False) as data:
            report=json.loads(str(data['report']))
            store.segment=segment;store.created=report['created']
            store.updated=report['updated'];store.frames=report['frames']
            store.voxel_m=report['voxel_m'];store.revision=report['calibration_revision']
            store.objects=report['objects']
            store.pruned=report.get('contradiction_pruned',0)
            counts = data['observations'] if 'observations' in data else np.ones(len(data['points']),dtype=int)
            for xyz,rgb,count in zip(data['points'],data['colors'],counts):
                store.voxels[tuple(np.floor(xyz/store.voxel_m).astype(int))]=(xyz.astype(float),rgb.astype(float),int(count))
        return store

    def clear_contradicted(self, frame, depth, metadata, map_T_camera):
        """Remove ghosts only after three consecutive clear-space observations.

        Requires an accepted pose from the caller. Unknown pixels, occlusions,
        image edges and depth discontinuities cannot erase remembered surfaces.
        This is map maintenance, never navigation free-space certification.
        """
        if map_T_camera is None or frame == self.last_cleanup_frame:
            return 0
        self.last_cleanup_frame = frame
        if not self.voxels:
            return 0
        T = np.asarray(map_T_camera, float)
        if T.shape != (4,4) or not np.isfinite(T).all():
            raise ValueError('Invalid map transform')
        intr = metadata['depth_intrinsics']
        if intr['model'] != 'distortion.none':
            raise ValueError('Unsupported depth distortion')
        keys = list(self.voxels)
        world = np.array([self.voxels[k][0] for k in keys])
        camera = (world-T[:3,3]) @ T[:3,:3]
        z = camera[:,2]
        safe_z = np.maximum(z, .001)
        u = np.rint(camera[:,0]*intr['fx']/safe_z+intr['ppx']).astype(int)
        v = np.rint(camera[:,1]*intr['fy']/safe_z+intr['ppy']).astype(int)
        h,w = depth.shape
        ids = np.flatnonzero((z>=.15)&(z<=4)&(u>=1)&(u<w-1)&(v>=1)&(v<h-1))
        conflicts = set()
        if len(ids):
            samples = np.stack([depth[v[ids]+dy,u[ids]+dx] for dy in (-1,0,1)
                                for dx in (-1,0,1)],axis=1)*float(metadata['depth_scale_m'])
            lo,hi = samples.min(axis=1),samples.max(axis=1)
            good = (lo>=.15)&(hi<=4)&(hi-lo<.08)&(lo-z[ids]>.12)
            conflicts = {keys[i] for i in ids[good]}
        self.contradictions = {k:self.contradictions.get(k,0)+1 for k in conflicts}
        removed = [k for k,n in self.contradictions.items() if n>=3]
        for k in removed:
            del self.voxels[k]
            del self.contradictions[k]
        self.pruned += len(removed)
        self.dirty |= bool(removed)
        return len(removed)

    def integrate(self, frame, points, colors, objects, map_T_camera, timestamp, revision):
        if map_T_camera is None or frame == self.last_frame:
            return False
        T = np.asarray(map_T_camera, float)
        if T.shape != (4,4) or not np.isfinite(T).all():
            raise ValueError('Invalid map transform')
        if self.revision is not None and self.revision != revision:
            raise ValueError('Calibration changed within map segment')
        world = transform_points(points, T)
        colors = np.asarray(colors)
        valid = np.isfinite(world).all(axis=1)
        world, colors = world[valid], colors[valid]
        # One sample per voxel per observation, preventing dense frames from
        # overwhelming previous observations. Cap growth without evicting history.
        keys = np.floor(world/self.voxel_m).astype(np.int64)
        _, ids = np.unique(keys, axis=0, return_index=True)
        for idx in ids:
            key = tuple(keys[idx])
            old = self.voxels.get(key)
            if old is None:
                if len(self.voxels) < self.max_voxels:
                    self.voxels[key] = (world[idx].copy(), colors[idx].astype(float), 1)
            else:
                xyz, rgb, count = old
                count = min(count+1, 100)
                self.voxels[key] = (xyz+(world[idx]-xyz)/count,
                                    rgb+(colors[idx]-rgb)/count, count)
        used = set()
        for obj in objects:
            surface = obj.get('surface_3d')
            if not surface:
                continue
            center = transform_points(np.array(surface['center_m']), T)
            size = np.abs(T[:3,:3]) @ np.array(surface['size_m'])
            if not np.isfinite(center).all() or not np.isfinite(size).all():
                continue
            matches = [(np.linalg.norm(center-np.array(o['center_m'])), i)
                       for i,o in enumerate(self.objects)
                       if i not in used and o['class_id'] == obj['class_id']]
            distance, index = min(matches, default=(float('inf'), -1))
            if distance <= .30:
                track = self.objects[index]
                n = min(track['observations']+1, 20)
                track['center_m'] = (np.array(track['center_m'])+(center-track['center_m'])/n).tolist()
                track['size_m'] = size.tolist()
                track['observations'] += 1
            else:
                if len(self.objects) >= 2000:
                    continue
                index = len(self.objects)
                track = dict(id=uuid.uuid4().hex[:12], name=obj['name'], class_id=obj['class_id'],
                             center_m=center.tolist(), size_m=size.tolist(), observations=1,
                             first_seen=timestamp)
                self.objects.append(track)
            used.add(index)
            track.update(last_seen=timestamp, confidence=obj['confidence'])
        self.last_frame = frame
        self.frames += 1
        self.updated = timestamp
        self.revision = revision
        self.dirty = True
        return True

    def save(self):
        if not self.dirty:
            return
        values = list(self.voxels.values())
        points = np.array([v[0] for v in values], np.float32).reshape(-1,3)
        colors = np.array([v[1] for v in values], np.uint8).reshape(-1,3)
        observations = np.array([v[2] for v in values], np.uint16)
        report = dict(segment=self.segment, created=self.created, updated=self.updated,
                      frames=self.frames, points=len(points), voxel_m=self.voxel_m,
                      capacity_reached=len(points)>=self.max_voxels, objects=self.objects,
                      calibration_revision=self.revision, frame_id='rgb_map_'+self.segment,
                      registration='independent segment; no inter-segment alignment',
                      object_association='same class within 0.30 m; provisional static surface tracks')
        report.update(confirmed_points=int(np.count_nonzero(observations>=3)),
                      contradiction_pruned=self.pruned,
                      cleanup='3 agreeing observations for display; 3 clear-space contradictions to remove')
        temp = self.directory / (self.segment+'.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, points=points, colors=colors, observations=observations, report=json.dumps(report))
        temp.replace(self.directory/(self.segment+'.npz'))
        self.dirty = False
