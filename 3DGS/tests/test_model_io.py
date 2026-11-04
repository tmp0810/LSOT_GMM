import numpy as np
import pytest
import torch

from lsot3dgs.model_io import SceneGaussians,load_scene,save_ply,geometry_to_scene,quaternion_to_matrix
from lsot3dgs.partition import make_partition,initialize
from lsot3dgs.appearance import transfer
from lsot3dgs.plans import LiftedPlan,SortedProjection


def scene(n=20):
    g=torch.Generator().manual_seed(8)
    return SceneGaussians(torch.randn(n,3,generator=g),torch.randn(n,3,generator=g)-3,
        torch.randn(n,4,generator=g),torch.randn(n,1,generator=g),torch.randn(n,16,3,generator=g),2.5,30000)


def test_ply_roundtrip_attributes_and_covariance(tmp_path):
    src=scene(); path=tmp_path/"point_cloud.ply"
    save_ply(path,src); restored=load_scene(path,spatial_lr_scale=2.5)
    for a,b in zip(src.tensors(),restored.tensors()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    gmm,_=src.gaussian_set()
    converted,info=geometry_to_scene(gmm,src.sh,src.opacity_logits)
    back,_=converted.gaussian_set()
    torch.testing.assert_close(gmm.covariances,back.covariances,rtol=3e-6,atol=1e-9)
    assert info["quaternion_relative_error"]<1e-10


def test_partition_exact_capacity_and_zero_mass_mapping():
    src=scene(); src.opacity_logits[3]=-10000
    p=make_partition(src,7,{"partition":"kd","block_cap":3})
    assert len(p.order)==19 and 3 not in p.order
    total=0
    for _,ids,cuts in p.blocks():
        source,_=src.gaussian_set(ids)
        target,record=initialize(source,cuts)
        assert target.count<=3
        total+=target.count
        assert abs(float(target.weights.sum())-1)<1e-12
    assert total==7


def test_mean_appearance_averages_opacity_not_logits():
    src=scene(2); src.opacity_logits=torch.tensor([[-4.],[2.]])
    gmm,_=src.gaussian_set()
    target,_=initialize(gmm,np.array([0,2]))
    p=LiftedPlan(SortedProjection(torch.zeros(2),gmm.weights),
                 SortedProjection(torch.zeros(1),target.weights))
    sh,opacity,_=transfer(src,np.arange(2),gmm,target,p,rule="mean")
    expected=(gmm.weights[:,None]*src.opacity_logits.double().sigmoid()).sum(0)
    torch.testing.assert_close(opacity.sigmoid()[0],expected)


def test_checkpoint_load(tmp_path):
    s=scene(5)
    cap=(3,s.xyz,s.sh[:,:1],s.sh[:,1:],s.log_scales,s.rotations,s.opacity_logits,
         torch.zeros(5),torch.zeros(5,1),torch.zeros(5,1),{},s.spatial_lr_scale)
    path=tmp_path/"checkpoint.pth"; torch.save((cap,15000),path)
    restored=load_scene(path)
    assert restored.iteration==15000 and restored.spatial_lr_scale==2.5
