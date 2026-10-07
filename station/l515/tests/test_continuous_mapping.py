"""Needs the full dimOS environment (see station/README.md); skipped otherwise."""
import pytest
pytest.importorskip('open3d')
pytest.importorskip('dimos')
import numpy as np
from station.l515.continuous_mapping import factory_colors, global_relocalize


def test_empty_factory_cloud_does_not_crash_viewer():
    assert factory_colors(np.empty((0,3)),{},np.empty((0,0,3))).shape==(0,3)


def test_factory_color_uses_optical_extrinsics_and_rgb_order():
    meta=dict(color_intrinsics=dict(fx=1.,fy=1.,ppx=0.,ppy=0.,model='distortion.none',coeffs=[0.]*5),
              depth_to_color=dict(rotation_column_major=np.eye(3).reshape(-1,order='F').tolist(),translation_m=[1,0,0]))
    image=np.array([[[1,2,3],[4,5,6]]],np.uint8)
    colors=factory_colors(np.array([[0,0,1],[10,0,1.]]),meta,image)
    np.testing.assert_array_equal(colors,[[6,5,4],[125,125,125]])


def test_downsampled_color_retains_full_resolution_intrinsics():
    meta=dict(color_intrinsics=dict(fx=4.,fy=4.,ppx=0.,ppy=0.,model='distortion.none',coeffs=[0.]*5),
              depth_to_color=dict(rotation_column_major=np.eye(3).reshape(-1,order='F').tolist(),translation_m=[0,0,0]),color_pixel_stride=4)
    image=np.array([[[1,2,3],[4,5,6]]],np.uint8)
    np.testing.assert_array_equal(factory_colors(np.array([[1.,0,1]]),meta,image),[[6,5,4]])


def test_global_relocalization_accepts_a_strong_rigid_match():
    rng=np.random.default_rng(4)
    # Asymmetric volumetric geometry avoids the ambiguous single-plane case.
    target=np.vstack((rng.uniform([-1,-.4,.3],[1,.7,2.2],(2500,3)),
                      rng.normal([.7,.2,1.1],[.04,.25,.4],(800,3))))
    result=global_relocalize(target.copy(),target,voxel_m=.10)
    assert result is not None
    transform,quality=result
    np.testing.assert_allclose(transform,np.eye(4),atol=.03)
    assert quality['fitness']>.9
