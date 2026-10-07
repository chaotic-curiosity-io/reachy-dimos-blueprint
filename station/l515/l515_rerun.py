"""Native + web Rerun views of DimOS mapping, independent RGB, and geometry clusters."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import sys
from station.l515.reachy_perception import atomic_json

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb

from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image

ACTIVE_MAP_ROOT = 'remembered/active'


def clean_persistent_display(points, colors):
    """Display-only isolation filter; never alter saved geometry or costmaps."""
    from scipy.spatial import cKDTree
    points = np.asarray(points)
    colors = np.asarray(colors)
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    if not len(points):
        return points, colors
    neighbors = cKDTree(points).query_ball_point(points, .09, return_length=True)
    keep = neighbors >= 4  # three other samples, excluding the query itself
    return points[keep], colors[keep]


def segment_geometry(points):
    """Plane/DBSCAN segmentation; labels are geometric, never semantic classes."""
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    colors=np.full((len(points),3),0.3)
    remaining=np.arange(len(points))
    boxes=[]
    palette=np.array([[.25,.8,.6],[.3,.6,1],[.9,.6,.2],[.8,.3,.8],[.9,.4,.4]])
    for i in range(3):
        if len(remaining)<300: break
        part=cloud.select_by_index(remaining.tolist())
        _,ids=part.segment_plane(distance_threshold=.035,ransac_n=3,num_iterations=100)
        if len(ids)<300: break
        colors[remaining[ids]]=palette[i]*.6
        remaining=np.delete(remaining,ids)
    if len(remaining)>=30:
        part=cloud.select_by_index(remaining.tolist())
        labels=np.asarray(part.cluster_dbscan(eps=.13,min_points=25))
        for label in sorted(set(labels)-{-1})[:30]:
            ids=remaining[labels==label]
            colors[ids]=palette[label%len(palette)]
            lo,hi=points[ids].min(axis=0),points[ids].max(axis=0)
            boxes.append(((lo+hi)/2,(hi-lo)/2,f'geometry cluster {label}'))
    return colors,boxes


def snapshot_status(snapshot, perception, now):
    reason=perception.get('rgbd_reason') or perception.get('state','unavailable')
    if snapshot is None:
        return f'Waiting for the first accepted RGB-D snapshot. {reason}'
    age=max(0,now-snapshot['source_received_at_unix'])
    fresh=(perception.get('state')=='running' and perception.get('rgbd_state')=='preview'
           and now-perception.get('reported_at_unix',0)<3 and age<3)
    state='Updating' if fresh else 'FROZEN — last accepted snapshot'
    return (f'{state} | Capture receipt age: {age:.1f} seconds | '
            f"{snapshot['projected_points']:,} points\n\n"
            f"Current pipeline: {reason}. Articulated preview; approximate timing and estimated scale. "
            'Snapshot is not a live obstacle map.')


def make_blueprint(segments):
    return rrb.Blueprint(rrb.Tabs(
        rrb.Spatial2DView(origin='rgb',name='RGB'),
        rrb.Spatial2DView(origin='objects',name='Objects'),
        rrb.Vertical(rrb.TextDocumentView(origin='rgbd_status',name='Alignment snapshot status'),
            rrb.Spatial2DView(origin='alignment',name='RGB + depth alignment'),
            row_shares=[1,5],name='RGB + depth alignment'),
        rrb.Vertical(
            rrb.TextDocumentView(origin='rgbd_status',name='RGB-D snapshot status'),
            rrb.Spatial3DView(origin='semantic',name='Live RGB-D scan'),
            row_shares=[1,5],name='Head RGB-D snapshot'),
        rrb.Vertical(rrb.TextDocumentView(origin='persistent_status',name='Map accumulation status'),
            # The segment UUID changes whenever mapping starts a new local
            # frame. A UUID here gets baked into Rerun's saved view and leaves
            # `$origin/**` pointing at an entity that no longer exists. Keep
            # the view on one stable alias; the publisher swaps the active
            # segment's data underneath it.
            rrb.Spatial3DView(origin=ACTIVE_MAP_ROOT,name='Active map segment'),
            row_shares=[1,5],name='RGB-colored 3D objects — persistent'),
        rrb.Spatial3DView(origin='points',name='Points'),
        rrb.Spatial3DView(origin='voxels',name='Voxels'),
        rrb.Spatial3DView(origin='segments',name='3D segmentation + boxes'),
        rrb.TextDocumentView(origin='status',name='Status'),
        rrb.Vertical(rrb.TextDocumentView(origin='l515_live_status',name='Live L515 status'),
            rrb.Spatial3DView(origin='l515_live',name='Current factory RGB-D'),
            row_shares=[1,5],name='Live L515 RGB-D'),active_tab=9),
        rrb.TimePanel(state='collapsed'))


def publish_persistent(root, points, colors, tracks):
    """Changing map samples are temporal, so bounded history can evict them."""
    rr.log(root+'/cloud',rr.Points3D(points,colors=colors,radii=.004))
    rr.log(root+'/objects',rr.Clear(recursive=True))
    if tracks:
        rr.log(root+'/objects',rr.Boxes3D(
            centers=[o['center_m'] for o in tracks],
            half_sizes=[np.array(o['size_m'])/2 for o in tracks],
            labels=[f"{o['name']} #{o['id'][:6]} (remembered surface)" for o in tracks],
            fill_mode=rr.components.FillMode.MajorWireframe,colors=[255,210,70]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--native',action='store_true')
    p.add_argument('--stdout',action='store_true',help='Pipe RRD directly to native Rerun; no TCP servers')
    args=p.parse_args()
    rr.init('Reachy DimOS live',spawn=False)
    segments=sorted(p.stem for p in (args.directory/'colored-map-segments').glob('*.npz'))
    blueprint=make_blueprint(segments)
    native_process=None
    native_stream=None
    native_bytes=0
    if args.native and not args.stdout:
        executable=Path(sys.executable).with_name('rerun')
        native_process=subprocess.Popen([str(executable),'-','--memory-limit','512MiB'],
            stdin=subprocess.PIPE,start_new_session=True)
        native_stream=rr.binary_stream()
        url='native direct pipe (no localhost/gRPC)'
    elif args.stdout:
        rr.stdout(default_blueprint=blueprint)
        url='direct stdin pipe (no network)'
    else:
        url=rr.serve_grpc(grpc_port=9878,server_memory_limit='256MiB',default_blueprint=blueprint)
        url=url.replace('0.0.0.0','127.0.0.1')
        rr.serve_web_viewer(web_port=8778,open_browser=False,connect_to=url)
    # Activate this migrated blueprint once on startup. Older saved layouts
    # contain a now-stale segment UUID and cannot repair their own `$origin`.
    rr.send_blueprint(blueprint,make_active=True,make_default=True)
    for root in ['points','voxels','segments','semantic','l515_live']:
        rr.log(root,rr.ViewCoordinates.RDF,static=True)
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    lock=threading.Lock()
    pending=None
    object_pending=None
    def receive(image):
        nonlocal pending
        with lock: pending=image
    def receive_objects(image):
        nonlocal object_pending
        with lock: object_pending=image
    object_transport=LCMTransport('/reachy/perception/annotated_image',Image)
    object_unsubscribe=object_transport.subscribe(receive_objects)
    transport=LCMTransport('/reachy/color_image',Image)
    unsubscribe=transport.subscribe(receive)
    persistent_modified={}
    persistent_emitted={}
    displayed_segment=None
    modified=None
    rgbd_modified=None
    snapshot=None
    live_stamp=None
    print('Rerun stream '+url,flush=True,file=sys.stderr)
    try:
        while not stop.wait(.2):
            if native_stream is not None:
                if native_process.poll() is not None:break
                payload=native_stream.read(flush_timeout_sec=2)
                if payload:
                    native_process.stdin.write(payload)
                    native_process.stdin.flush()
                    native_bytes+=len(payload)
            atomic_json(args.directory/('rerun-pipe-heartbeat.json' if args.stdout else 'rerun-heartbeat.json'),
                dict(reported_at_unix=time.time(),transport='native_pipe' if native_stream is not None else 'grpc',
                     delivered_bytes=native_bytes))
            # This live view shares the mapper's exact RGB/depth capture. It
            # never waits for the independently moving head camera or YOLO.
            try:
                live_path=args.directory/'alignment-frame.npz'
                if not live_path.exists():live_path=args.directory/'mapping-frame.npz'
                stamp=live_path.stat().st_mtime_ns
                if stamp!=live_stamp:
                    from station.l515.continuous_mapping import factory_colors
                    from station.l515.persistent_rgbd import depth_points
                    with np.load(live_path,allow_pickle=False) as data:
                        handoff=json.loads(str(data['handoff']))
                        age=time.monotonic()-handoff['depth_time']
                        if not 0<=age<2: raise ValueError('depth capture is stale')
                        xyz=depth_points(data['depth'],handoff['meta'])
                        colors=factory_colors(xyz,handoff['meta'],data['color'])
                    rr.log('l515_live/cloud',rr.Points3D(xyz,colors=colors,radii=.004))
                    rr.log('l515_live_status',rr.TextDocument(
                        f'Live factory RGB-D | {len(xyz):,} points | capture age {age:.2f} s\n'
                        'Current scan only; no accumulated history or head-camera color projection.'))
                    live_stamp=stamp
                elif time.time()-live_path.stat().st_mtime>2:
                    raise ValueError('no fresh depth capture')
            except (OSError,ValueError,KeyError) as exc:
                rr.log('l515_live/cloud',rr.Clear(recursive=True))
                rr.log('l515_live_status',rr.TextDocument('Live depth unavailable: '+str(exc)))
            with lock:
                image,pending=pending,None
                object_image,object_pending=object_pending,None
            if image is not None:
                rr.log('rgb/image',rr.Image(image.to_opencv()[:,:,::-1]).compress(jpeg_quality=80))
            if object_image is not None:
                rr.log('objects/image',rr.Image(object_image.to_opencv()[:,:,::-1]).compress(jpeg_quality=85))
            try:
                perception=json.loads((args.directory/'perception-live.json').read_text())
                if time.time()-perception['reported_at_unix']>3:
                    perception['state']='stale'
                if perception.get('state') != 'running':
                    rr.log('objects/image',rr.Clear(recursive=True))
            except (OSError,ValueError,KeyError):
                perception={'state':'unavailable'}
                rr.log('objects/image',rr.Clear(recursive=True))
            try:
                paths=sorted((args.directory/'colored-map-segments').glob('*.npz'))
                active=json.loads((args.directory/'continuous-mapping.json').read_text()).get('segment')
                paths=[path for path in paths if path.stem==active]
                found=[path.stem for path in paths]
                if found!=segments:
                    segments=found
                if active != displayed_segment:
                    # Prevent the prior segment from remaining visible while
                    # the stable alias is repopulated with the new one.
                    rr.log(ACTIVE_MAP_ROOT,rr.Clear(recursive=True))
                    displayed_segment=active
                for path in paths:
                    stamp=path.stat().st_mtime_ns
                    if (persistent_modified.get(path.stem)==stamp
                            and time.monotonic()-persistent_emitted.get(path.stem,0)<5):
                        continue
                    with np.load(path,allow_pickle=False) as data:
                        saved=json.loads(str(data['report']))
                        root=ACTIVE_MAP_ROOT
                        rr.log(root,rr.ViewCoordinates.RDF,static=True)
                        confirmed = data['observations']>=3 if 'observations' in data else np.ones(len(data['points']),bool)
                        clean_points, clean_colors = clean_persistent_display(data['points'][confirmed], data['colors'][confirmed])
                        tracks=[o for o in saved['objects'] if o.get('name')!='person' or time.time()-o.get('last_seen',0)<10]
                        publish_persistent(root,clean_points,clean_colors,tracks)
                    persistent_modified[path.stem]=stamp
                    persistent_emitted[path.stem]=time.monotonic()
                mapping=json.loads((args.directory/'continuous-mapping.json').read_text())
                if time.time()-mapping.get('reported_at_unix',0)>2: mapping['state']='stale'
                rr.log('persistent_status',rr.TextDocument(
                    f"Persistent colored map | {len(segments)} separate segment(s) | "
                    f"Active: {mapping.get('segment','unavailable')[:8]}\n\n"
                    f"Tracking: {mapping.get('state',perception.get('state'))}. "
                    f"{mapping.get('reason') or ''} "
                    f"{mapping.get('voxels',0):,} voxels; {mapping.get('remembered_objects',0)} remembered objects.\n\n"
                    f"Head-camera labels: {perception.get('rgbd_state',perception.get('state'))}. "
                    f"{perception.get('rgbd_reason') or ''}\n\n"
                    'Display: requires 3 observations; isolated points filtered; 4 mm radius. '
                    f"Removed {mapping.get('contradiction_pruned',0)} points contradicted by 3 aligned depth scans. "
                    'Occluded/unseen geometry is retained. Tracking loss freezes cleanup. '
                    'L515 depth + factory RGB accumulate independently of head-camera label timing. '
                    'Saved map resumes only after a successful scan match. '
                    'Object IDs are provisional spatial matches, not verified identities.'))
            except (OSError,ValueError,KeyError) as exc:
                rr.log('persistent_status',rr.TextDocument('Map unavailable: '+str(exc)))
            try:
                bundle=args.directory/'rgbd-last-accepted.npz'
                if bundle.exists() and bundle.stat().st_mtime_ns!=rgbd_modified:
                    stamp=bundle.stat().st_mtime_ns
                    with np.load(bundle,allow_pickle=False) as data:
                        candidate=json.loads(str(data['report']))
                        if candidate.get('rgbd_state')!='preview':
                            raise ValueError('Snapshot was not accepted')
                        rr.log('alignment/image',rr.Image(data['overlay'][:,:,::-1]).compress(jpeg_quality=85))
                        rr.log('semantic/cloud',rr.Points3D(data['points'],colors=data['colors'],radii=.005))
                        rr.log('semantic/boxes',rr.Clear(recursive=True))
                        supported=[o for o in candidate['objects'] if o.get('surface_3d')]
                        if supported:
                            rr.log('semantic/boxes',rr.Boxes3D(centers=[o['surface_3d']['center_m'] for o in supported],
                                half_sizes=[np.array(o['surface_3d']['size_m'])/2 for o in supported],
                                labels=[f"{o['name']} {o['confidence']:.0%} (visible surface)" for o in supported],
                                fill_mode=rr.components.FillMode.MajorWireframe,colors=[255,210,70]))
                        snapshot=candidate
                    rgbd_modified=stamp
                rr.log('rgbd_status',rr.TextDocument(snapshot_status(snapshot,perception,time.time())))
            except (OSError,ValueError,KeyError) as exc:
                rr.log('rgbd_status',rr.TextDocument(f'Snapshot unavailable: {exc}'))
                rgbd_modified=None
            try:
                status=json.loads((args.directory/'continuous-mapping.json').read_text())
                if time.time()-status['reported_at_unix']>2: status['state']='offline'
                rr.log('status',rr.TextDocument(json.dumps({'mapping':status,'perception':perception},indent=2)))
                path=args.directory/'live-map.ply'
                stamp=path.stat().st_mtime_ns
                if stamp==modified: continue
                cloud=o3d.io.read_point_cloud(str(path))
                points=np.asarray(cloud.points)
                if not len(points): continue
                # Named views replace their previous geometry each update.
                colors=np.asarray(cloud.colors) if cloud.has_colors() else np.full((len(points),3),.6)
                rr.log('points/map',rr.Points3D(points,colors=colors,radii=.008))
                rr.log('voxels/map',rr.Boxes3D(centers=points,half_sizes=[.019]*3,colors=colors,fill_mode=rr.components.FillMode.Solid))
                segmented,boxes=segment_geometry(points)
                rr.log('segments/cloud',rr.Points3D(points,colors=segmented,radii=.012))
                rr.log('segments/boxes',rr.Clear(recursive=True))
                if boxes:
                    rr.log('segments/boxes',rr.Boxes3D(centers=[b[0] for b in boxes],
                        half_sizes=[b[1] for b in boxes],labels=[b[2] for b in boxes],
                        fill_mode=rr.components.FillMode.MajorWireframe,colors=[255,210,70]))
                modified=stamp
            except (OSError,ValueError,RuntimeError) as exc:
                rr.log('status',rr.TextDocument('Waiting for map: '+str(exc)))
    finally:
        object_unsubscribe();object_transport.stop()
        unsubscribe();transport.stop();rr.disconnect()
        if native_process is not None:
            native_process.stdin.close()


if __name__=='__main__': main()
