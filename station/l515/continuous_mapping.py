"""Independent L515 mapping; camera perception never gates depth odometry."""
import argparse
import io
import json
import math
from pathlib import Path
import signal
import threading
import time
import zipfile
from urllib.request import urlopen
import cv2
import numpy as np
import open3d as o3d
from station.l515.l515_mapping import L515Odometry
from station.l515.persistent_rgbd import PersistentRGBD, depth_points, transform_points
from station.l515.rgbd_projection import DEFAULT_DEPTH_URL, calibration_path, require_url, unpack_bundle
from station.l515.reachy_perception import atomic_json
from station.l515.semantic_handoff import join_observation


def global_relocalize(source_points, target_points, *, voxel_m=.10):
    """Globally align one scan to a retained map, or return ``None``.

    Feature registration is deliberately followed by strict dense ICP and a
    gravity-axis check.  A visually plausible but weak room match must never
    move the saved map origin underneath navigation.
    """
    reg=o3d.pipelines.registration
    def prepare(points):
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points,dtype=float)))
        cloud=cloud.voxel_down_sample(voxel_m)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m*2.5,max_nn=40))
        feature=reg.compute_fpfh_feature(cloud,o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m*5,max_nn=100))
        return cloud,feature
    source,source_feature=prepare(source_points)
    target,target_feature=prepare(target_points)
    if len(source.points)<300 or len(target.points)<500:return None
    coarse=reg.registration_ransac_based_on_feature_matching(
        source,target,source_feature,target_feature,True,voxel_m*1.5,
        reg.TransformationEstimationPointToPoint(False),3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(.9),
         reg.CorrespondenceCheckerBasedOnDistance(voxel_m*1.5)],
        reg.RANSACConvergenceCriteria(100_000,.999))
    dense_source=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(source_points,dtype=float))).voxel_down_sample(.04)
    dense_target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(target_points,dtype=float)))
    dense_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.16,max_nn=30))
    result=reg.registration_icp(dense_source,dense_target,.08,coarse.transformation,
        reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=.08)),
        reg.ICPConvergenceCriteria(max_iteration=40))
    result=reg.registration_icp(dense_source,dense_target,.04,result.transformation,
        reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=.04)),
        reg.ICPConvergenceCriteria(max_iteration=30))
    # Optical +y is gravity/down. A rigid chassis move should preserve it.
    gravity_alignment=float(result.transformation[:3,1] @ np.array([0.,1.,0.]))
    if result.fitness<.55 or result.inlier_rmse>.03 or gravity_alignment<math.cos(math.radians(15)):
        return None
    return result.transformation.copy(),dict(fitness=float(result.fitness),
        rmse_m=float(result.inlier_rmse),gravity_alignment=gravity_alignment)


def factory_colors(points, metadata, bgr):
    if not len(points):
        return np.empty((0,3),np.uint8)
    intr=metadata['color_intrinsics'];extr=metadata['depth_to_color']
    if intr['model'] not in ('distortion.none','distortion.brown_conrady'):
        raise ValueError('Unsupported factory color distortion')
    R=np.array(extr['rotation_column_major']).reshape(3,3,order='F')
    color_points=points@R.T+np.array(extr['translation_m'])
    K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1.]])
    uv=cv2.projectPoints(color_points,np.zeros(3),np.zeros(3),K,np.array(intr['coeffs']))[0]
    uv=np.rint(uv.reshape(-1,2)/metadata.get('color_pixel_stride',1)).astype(int)
    valid=(color_points[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<bgr.shape[1])&(uv[:,1]>=0)&(uv[:,1]<bgr.shape[0])
    colors=np.full((len(points),3),125,np.uint8)
    colors[valid]=bgr[uv[valid,1],uv[valid,0],::-1]
    return colors


def main():
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from scipy.spatial.transform import Rotation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--pi',default=DEFAULT_DEPTH_URL,
                   help='Pi depth streamer base URL (env DEPTH_SERVER_URL), e.g. http://<pi-ip>:8765')
    p.add_argument('--calibration',type=Path,default=None,
                   help='paired-extrinsics-report.json (default: see rgbd_projection.calibration_path)')
    args=p.parse_args();out=args.directory
    args.pi=require_url(args.pi,'--pi','DEPTH_SERVER_URL')
    calibration=json.loads(calibration_path(out,args.calibration).read_text())
    store=PersistentRGBD(out)
    odometry=L515Odometry(reset_on_loss=False)
    raw=LCMTransport('/reachy/lidar',PointCloud2)
    registered=LCMTransport('/reachy/experimental/registered_cloud',PointCloud2)
    maps=LCMTransport('/reachy/experimental/global_map',PointCloud2)
    poses=LCMTransport('/reachy/experimental/sensor_odom',PoseStamped)
    active=out/'continuous-active.json'
    retained_points=None
    loaded_checkpoint=False
    if active.exists():
        segment=json.loads(active.read_text())['segment']
        checkpoint=store.directory/(segment+'.tracking')
        if checkpoint.exists() and (store.directory/(segment+'.npz')).exists():
            candidate=PersistentRGBD.restore(out,segment)
            if candidate.revision=='factory_'+calibration['l515_serial']:
                with np.load(checkpoint,allow_pickle=False) as data:
                    odometry.reference=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data['reference']))
                    odometry.reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.16,max_nn=30))
                    odometry.reference_pose=data['reference_pose'].copy();odometry.pose=data['pose'].copy()
                with np.load(store.directory/(segment+'.npz'),allow_pickle=False) as data:
                    retained_points=data['points'].copy()
                odometry.last_ts=time.monotonic()-1
                store=candidate
                loaded_checkpoint=True
                print('Retained map loaded; matching required before integration: '+segment,flush=True)
    def save():
        store.save()
        if odometry.reference is not None and (store.directory/(store.segment+'.npz')).exists():
            temp=store.directory/(store.segment+'.tracking.tmp')
            with temp.open('wb') as f:
                np.savez(f,reference=np.asarray(odometry.reference.points),reference_pose=odometry.reference_pose,pose=odometry.pose)
            temp.replace(store.directory/(store.segment+'.tracking'))
            atomic_json(active,dict(segment=store.segment))
            with np.load(store.directory/(store.segment+'.npz'),allow_pickle=False) as data:
                cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data['points']))
                cloud.colors=o3d.utility.Vector3dVector(data['colors']/255.)
                ply=out/'continuous-map.tmp.ply'
                o3d.io.write_point_cloud(str(ply),cloud)
                ply.replace(out/'live-map.ply')
                maps.publish(PointCloud2(pointcloud=cloud,frame_id='rgb_map_'+store.segment,ts=store.updated))
    halt=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:halt.set())
    last_frame=None;last_save=0;last_observation=None;last_received=None
    count=0;started=time.monotonic();status={};global_attempted=False
    pose_history={}
    try:
        while not halt.wait(.025):
            try:
                begin=time.monotonic()
                with urlopen(args.pi+'/api/mapping-frame',timeout=2) as r:bundle=r.read(2_000_001)
                end=time.monotonic()
                depth,meta=unpack_bundle(bundle)
                if meta['serial']!=calibration['l515_serial']:raise ValueError('Unexpected L515 serial')
                frame=(meta['session_id'],meta['depth_frame_number'])
                if frame==last_frame:continue
                last_frame=frame
                if meta['server_frame_age_ms']>400 or end-begin>.6:raise ValueError('Stale L515 acquisition')
                with zipfile.ZipFile(io.BytesIO(bundle)) as z:
                    color=cv2.imdecode(np.frombuffer(z.read('color.bmp'),np.uint8),cv2.IMREAD_COLOR)
                if color is None:raise ValueError('Invalid factory RGB')
                xyz=depth_points(depth,meta)
                timestamp=(begin+end)/2-meta['server_frame_age_ms']/1000
                # Alignment needs the capture now, not after ICP/map persistence.
                # This handoff intentionally has no accepted world pose.
                early=dict(meta=meta,begin=begin,end=end,depth_time=timestamp,
                           map_pose=None,segment=store.segment,created_monotonic=time.monotonic())
                temp=out/'alignment-frame.tmp'
                with temp.open('wb') as f:
                    np.savez(f,depth=depth,color=color,handoff=json.dumps(early))
                temp.replace(out/'alignment-frame.npz')
                pose=odometry.update(xyz,timestamp)
                if (pose is None and loaded_checkpoint and not global_attempted
                        and odometry.consecutive_rejections>=3 and retained_points is not None):
                    global_attempted=True
                    print('Local resume failed; attempting global retained-map registration',flush=True)
                    recovered=global_relocalize(xyz,retained_points)
                    if recovered is not None:
                        pose,quality=recovered
                        reference=odometry._prepare(xyz)
                        reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.16,max_nn=30))
                        odometry.reference=reference;odometry.reference_pose=pose.copy();odometry.pose=pose.copy()
                        odometry.last_ts=timestamp;odometry.accepted+=1;odometry.consecutive_rejections=0
                        odometry.last={'state':'tracking','reason':'globally relocalized to retained map',**quality}
                        print('Global retained-map relocalization accepted',flush=True)
                    else:
                        print('Global retained-map relocalization rejected',flush=True)
                if (pose is None and loaded_checkpoint and global_attempted
                        and odometry.consecutive_rejections>=12):
                    # The robot can be carried beyond the saved map. Preserve
                    # that map, but do not remain permanently wedged trying to
                    # force unrelated geometry into it.
                    print('No valid retained-map match; starting a separate map segment',flush=True)
                    save()
                    store=PersistentRGBD(out)
                    odometry=L515Odometry(reset_on_loss=False)
                    loaded_checkpoint=False;retained_points=None
                    pose=odometry.update(xyz,timestamp)
                last_received=time.time()-(time.monotonic()-timestamp);count+=1
                raw.publish(PointCloud2.from_numpy(xyz,frame_id='l515_depth_optical',timestamp=last_received))
                if pose is not None:
                    pose_history[frame]=(store.segment,pose.copy())
                    while len(pose_history)>120:pose_history.pop(next(iter(pose_history)))
                    frame_id='rgb_map_'+store.segment
                    registered.publish(PointCloud2.from_numpy(transform_points(xyz,pose),frame_id=frame_id,timestamp=last_received))
                    poses.publish(PoseStamped(ts=last_received,frame_id=frame_id,position=pose[:3,3].tolist(),orientation=Rotation.from_matrix(pose[:3,:3]).as_quat().tolist()))
                    store.clear_contradicted(frame,depth,meta,pose)
                    store.integrate(frame,xyz,factory_colors(xyz,meta,color),[],pose,time.time(),'factory_'+meta['serial'])
                # Exact depth bundle and its accepted pose are handed to the
                # slower head-camera pipeline. Never substitute a later pose.
                handoff=dict(meta=meta,begin=begin,end=end,depth_time=timestamp,
                             map_pose=pose.tolist() if pose is not None else None,
                             segment=store.segment,created_monotonic=time.monotonic())
                temp=out/'mapping-frame.tmp'
                with temp.open('wb') as f:np.savez(f,depth=depth,color=color,handoff=json.dumps(handoff))
                temp.replace(out/'mapping-frame.npz')
                for pending in sorted((out/'semantic-pending').glob('*.json')):
                    try:
                        observation=json.loads(pending.read_text())
                        joined=join_observation(observation,pose_history)
                        if joined is not None:
                            segment,camera_pose=joined
                            if segment==store.segment:
                                store.integrate('object_'+observation['observation_id'],
                                    np.empty((0,3)),np.empty((0,3),np.uint8),
                                    observation['objects'],camera_pose,
                                    observation['source_received_at_unix'],'factory_'+meta['serial'])
                            pending.unlink(missing_ok=True)
                        elif time.time()-pending.stat().st_mtime>30:
                            pending.unlink(missing_ok=True)
                    except (OSError,ValueError,KeyError):pass
                if time.monotonic()-last_save>2:
                    save();last_save=time.monotonic()
                front=xyz[(np.abs(xyz[:,0])<.24)&(np.abs(xyz[:,1])<.25)]
                clearance=float(np.quantile(front[:,2],.05)) if len(front)>100 else None
                status={**odometry.status(),'segment':store.segment,'map_frame':'rgb_map_'+store.segment,
                        'voxels':len(store.voxels),'remembered_objects':len(store.objects),
                        'contradiction_pruned':store.pruned,
                        'reported_at_unix':time.time(),'source_received_at_unix':last_received,
                        'depth_frames':count,'mean_hz':count/(time.monotonic()-started),
                        'front_clearance_m':clearance,'processing_ms':(time.monotonic()-begin)*1000,
                        'running':True,'rgb_gates_mapping':False}
                atomic_json(out/'continuous-mapping.json',status)
            except Exception as exc:
                atomic_json(out/'continuous-mapping.json',{**status,'state':'unavailable','reason':str(exc),
                            'reported_at_unix':time.time(),'running':True})
                halt.wait(.2)
    finally:
        save()
        atomic_json(out/'continuous-mapping.json',{**status,'state':'stopped','running':False,'reported_at_unix':time.time()})
        for transport in (raw,registered,maps,poses):transport.stop()

if __name__=='__main__':main()
