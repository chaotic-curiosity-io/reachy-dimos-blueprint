import json
import numpy as np
from station.l515.persistent_rgbd import PersistentRGBD


def object_at(x, name='chair'):
    return dict(name=name,class_id=1,confidence=.9,
                surface_3d=dict(center_m=[x,.02,1.02],size_m=[.1,.1,.1]))


def test_motion_accumulates_same_world_object_and_keeps_unseen(tmp_path):
    m=PersistentRGBD(tmp_path)
    m.integrate('1',[[.02,.02,1.02]],[[255,0,0]], [object_at(.02)],np.eye(4),1,'cal')
    identity=m.objects[0]['id']
    T=np.eye(4);T[0,3]=1
    m.integrate('2',[[-.98,.02,1.02],[.02,.02,1.02]],[[255,0,0],[0,255,0]],
                [object_at(-.98)],T,2,'cal')
    assert len(m.voxels)==2
    assert len(m.objects)==1 and m.objects[0]['id']==identity
    assert m.objects[0]['observations']==2
    m.integrate('3',[[2.02,.02,1.02]],[[0,0,255]],[],np.eye(4),3,'cal')
    assert len(m.voxels)==3 and len(m.objects)==1
    m.save()
    with np.load(m.directory/(m.segment+'.npz')) as d:
        assert len(d['points'])==3
        assert json.loads(str(d['report']))['objects'][0]['id']==identity
    restarted=PersistentRGBD(tmp_path)
    assert restarted.segment!=m.segment
    assert (m.directory/(m.segment+'.npz')).exists()


def test_loss_duplicate_capacity_and_distinct_objects(tmp_path):
    m=PersistentRGBD(tmp_path,max_voxels=1)
    m.integrate('1',[[0,0,1]],[[255,0,0]],[object_at(0),object_at(.1)],np.eye(4),1,'c')
    assert len(m.objects)==2
    assert not m.integrate('1',[[1,0,1]],[[0,255,0]],[],np.eye(4),2,'c')
    assert not m.integrate('2',[[1,0,1]],[[0,255,0]],[],None,2,'c')
    assert m.frames==1
    m.integrate('3',[[1,0,1]],[[0,255,0]],[],np.eye(4),3,'c')
    assert len(m.voxels)==1 and (0,0,25) in m.voxels


def test_camera_rotation_maps_surface_bounds_and_rejects_revision_change(tmp_path):
    import pytest
    m=PersistentRGBD(tmp_path)
    T=np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]],float)
    m.integrate('a',[[0,0,1]],[[1,2,3]],[object_at(0)],T,1,'a')
    np.testing.assert_allclose(m.objects[0]['center_m'],[1.02,.02,0])
    with pytest.raises(ValueError,match='Calibration changed'):
        m.integrate('b',[[0,0,1]],[[1,2,3]],[],T,2,'b')
    assert m.frames==1


def test_restore_keeps_frame_objects_and_extends_map(tmp_path):
    m=PersistentRGBD(tmp_path)
    m.integrate('a',[[.02,.02,1.02]],[[100,200,50]],[object_at(.02)],np.eye(4),1,'c')
    m.save()
    resumed=PersistentRGBD.restore(tmp_path,m.segment)
    assert resumed.segment==m.segment and resumed.objects==m.objects
    resumed.integrate('b',[[1.02,.02,1.02]],[[1,2,3]],[],np.eye(4),2,'c')
    assert len(resumed.voxels)==2


def depth_fixture():
    return np.full((11,11),2000,np.uint16), dict(depth_scale_m=.001,
        depth_intrinsics=dict(model='distortion.none',fx=10,fy=10,ppx=5,ppy=5))


def test_ghost_requires_three_distinct_clear_space_frames(tmp_path):
    m=PersistentRGBD(tmp_path)
    m.integrate('a',[[0,0,1],[0,0,3],[9,0,1]],[[1,2,3]]*3,[],np.eye(4),1,'c')
    d,meta=depth_fixture()
    assert m.clear_contradicted('1',d,meta,np.eye(4))==0
    assert m.clear_contradicted('1',d,meta,np.eye(4))==0
    assert m.clear_contradicted('2',d,meta,np.eye(4))==0
    assert m.clear_contradicted('3',d,meta,np.eye(4))==1
    assert len(m.voxels)==2  # occluded and out-of-view geometry survives


def test_unknown_edges_and_lost_pose_do_not_clear(tmp_path):
    m=PersistentRGBD(tmp_path)
    m.integrate('a',[[0,0,1]],[[1,2,3]],[],np.eye(4),1,'c')
    d,meta=depth_fixture()
    for i in range(4):
        assert m.clear_contradicted(str(i),d,meta,None)==0
    d[5,5]=0
    for i in range(4):
        assert m.clear_contradicted('unknown'+str(i),d,meta,np.eye(4))==0
    d[5,5]=1500  # discontinuity, not reliable free-space evidence
    for i in range(4):
        assert m.clear_contradicted('edge'+str(i),d,meta,np.eye(4))==0
    assert len(m.voxels)==1


def test_contradictions_must_be_consecutive(tmp_path):
    m=PersistentRGBD(tmp_path)
    m.integrate('a',[[0,0,1]],[[1,2,3]],[],np.eye(4),1,'c')
    d,meta=depth_fixture()
    m.clear_contradicted('a',d,meta,np.eye(4))
    m.clear_contradicted('b',d,meta,np.eye(4))
    m.clear_contradicted('c',np.zeros_like(d),meta,np.eye(4))
    assert m.clear_contradicted('d',d,meta,np.eye(4))==0
    assert len(m.voxels)==1


def test_observation_counts_survive_restart(tmp_path):
    m=PersistentRGBD(tmp_path)
    for i in range(3):
        m.integrate(str(i),[[0,0,1]],[[1,2,3]],[],np.eye(4),i,'c')
    m.save()
    restored=PersistentRGBD.restore(tmp_path,m.segment)
    assert next(iter(restored.voxels.values()))[2]==3
