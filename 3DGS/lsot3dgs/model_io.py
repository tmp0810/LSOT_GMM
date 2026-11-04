"""Plain PLY/checkpoint IO, independent of the CUDA renderer."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from .geometry import GaussianSet, eig, repair_spd, sym


def quaternion_to_matrix(q):
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if bool((norm <= 0).any()):
        raise ValueError("Zero quaternion")
    w, x, y, z = (q / norm).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1,3,3)


def matrix_to_quaternion(matrix):
    # Branch on the largest quaternion component, avoiding unstable division near pi.
    r = matrix
    w2 = 1+r[:,0,0]+r[:,1,1]+r[:,2,2]
    x2 = 1+r[:,0,0]-r[:,1,1]-r[:,2,2]
    y2 = 1-r[:,0,0]+r[:,1,1]-r[:,2,2]
    z2 = 1-r[:,0,0]-r[:,1,1]+r[:,2,2]
    candidates = torch.stack((
        torch.stack((w2, r[:,2,1]-r[:,1,2], r[:,0,2]-r[:,2,0], r[:,1,0]-r[:,0,1]), 1),
        torch.stack((r[:,2,1]-r[:,1,2], x2, r[:,1,0]+r[:,0,1], r[:,0,2]+r[:,2,0]), 1),
        torch.stack((r[:,0,2]-r[:,2,0], r[:,1,0]+r[:,0,1], y2, r[:,2,1]+r[:,1,2]), 1),
        torch.stack((r[:,1,0]-r[:,0,1], r[:,0,2]+r[:,2,0], r[:,2,1]+r[:,1,2], z2), 1)), 1)
    choice = torch.stack((w2,x2,y2,z2), 1).argmax(1)
    q = candidates[torch.arange(len(r), device=r.device), choice]
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return torch.where(q[:, :1] < 0, -q, q)


@dataclass
class SceneGaussians:
    xyz: torch.Tensor
    log_scales: torch.Tensor
    rotations: torch.Tensor
    opacity_logits: torch.Tensor
    sh: torch.Tensor  # (N, (degree+1)^2, 3), DC first
    spatial_lr_scale: float | None = None
    iteration: int | None = None

    def __post_init__(self):
        n = len(self.xyz)
        expected = ((self.xyz,(n,3)), (self.log_scales,(n,3)), (self.rotations,(n,4)),
                    (self.opacity_logits,(n,1)))
        if n < 1 or any(tuple(t.shape) != shape for t,shape in expected):
            raise ValueError("Malformed scene Gaussian arrays")
        if self.sh.ndim != 3 or self.sh.shape[0] != n or self.sh.shape[2] != 3:
            raise ValueError("Expected SH (N,coefficients,3)")
        if (self.degree+1)**2 != self.sh.shape[1]:
            raise ValueError("Invalid SH coefficient count")
        if any(not bool(torch.isfinite(x).all()) for x in self.tensors()):
            raise ValueError("Nonfinite scene parameters")

    def tensors(self):
        return (self.xyz,self.log_scales,self.rotations,self.opacity_logits,self.sh)

    @property
    def count(self):
        return len(self.xyz)

    @property
    def degree(self):
        return int(round(self.sh.shape[1]**0.5))-1

    def subset(self, ids):
        return SceneGaussians(*(t[ids] for t in self.tensors()), self.spatial_lr_scale, self.iteration)

    def gaussian_set(self, ids=None, *, device="cpu", dtype=torch.float64):
        s = self if ids is None else self.subset(ids)
        means = s.xyz.to(device=device, dtype=dtype)
        r = quaternion_to_matrix(s.rotations.to(device=device, dtype=dtype))
        scales = s.log_scales.to(device=device, dtype=dtype).exp()
        factor = r * scales[:, None, :]
        # QR route agrees with the reviewed HMW extraction, preserving thin splats.
        pieces = []
        for block in factor.split(4096):
            _, upper = torch.linalg.qr(block.transpose(-1, -2))
            lower = upper.transpose(-1, -2)
            signs = torch.where(lower.diagonal(dim1=-2,dim2=-1) < 0, -1., 1.)
            lower = lower * signs[:,None,:]
            pieces.append(lower @ lower.transpose(-1,-2))
        cov, repairs = repair_spd(torch.cat(pieces))
        weights = s.opacity_logits.to(device=device,dtype=dtype).sigmoid().flatten()
        if bool((weights <= 0).any()):
            raise ValueError("Filter zero-opacity sources before constructing an OT block")
        return GaussianSet(weights / weights.sum(), means, cov), repairs


def load_scene(path, *, spatial_lr_scale=None):
    path = Path(path)
    if path.suffix.lower() == ".ply":
        from plyfile import PlyData
        v = PlyData.read(str(path))["vertex"]
        names = v.data.dtype.names
        def columns(prefix, expected=None):
            keys = sorted((k for k in names if k.startswith(prefix)), key=lambda k:int(k[len(prefix):]))
            if expected is not None and len(keys) != expected:
                raise ValueError(f"Expected {expected} {prefix} properties, got {len(keys)}")
            return (np.column_stack([v[k] for k in keys]).astype(np.float32, copy=True)
                    if keys else np.empty((len(v),0),dtype=np.float32))
        xyz = np.column_stack([v[k] for k in ("x","y","z")]).astype(np.float32)
        dc = columns("f_dc_",3).reshape(-1,3,1).transpose(0,2,1)
        rest = columns("f_rest_")
        rest = rest.reshape(len(xyz),3,-1).transpose(0,2,1)
        values = (xyz,columns("scale_",3),columns("rot_",4),
                  np.asarray(v["opacity"],dtype=np.float32).reshape(-1,1).copy(),np.concatenate((dc,rest),1))
        return SceneGaussians(*(torch.from_numpy(np.ascontiguousarray(v)) for v in values), spatial_lr_scale)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            values = [torch.from_numpy(z[k].copy()) for k in ("xyz","log_scales","rotations","opacity_logits","sh")]
        return SceneGaussians(*values, spatial_lr_scale)
    # User-supplied trusted local 3DGS checkpoints contain Python/NumPy state.
    cap, iteration = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cap, (tuple,list)) or len(cap) != 12:
        raise ValueError("Expected the pinned GHAP/Graphdeco 12-field checkpoint capture")
    _,xyz,dc,rest,scales,rotation,opacity,*_ = cap
    return SceneGaussians(xyz.detach().cpu(),scales.detach().cpu(),rotation.detach().cpu(),
        opacity.detach().cpu(),torch.cat((dc.detach().cpu(),rest.detach().cpu()),1),
        float(cap[-1]),int(iteration))


def save_ply(path, scene):
    from plyfile import PlyData, PlyElement
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    n = scene.count
    dc = scene.sh[:,:1].transpose(1,2).reshape(n,-1)
    rest = scene.sh[:,1:].transpose(1,2).reshape(n,-1)
    names = ["x","y","z","nx","ny","nz"] + [f"f_dc_{i}" for i in range(dc.shape[1])]
    names += [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    values = torch.cat((scene.xyz,torch.zeros_like(scene.xyz),dc,rest,scene.opacity_logits,
                        scene.log_scales,scene.rotations),1).detach().cpu().float().numpy()
    data = np.empty(n,dtype=[(name,"f4") for name in names])
    for k,name in enumerate(names):
        data[name] = values[:,k]
    temporary = path.with_suffix(".tmp.ply")
    PlyData([PlyElement.describe(data,"vertex")], text=False).write(str(temporary))
    temporary.replace(path)


def geometry_to_scene(target, sh, opacity, *, spatial_lr_scale=None, export_floor="relative"):
    cov, repairs = repair_spd(target.covariances,
                             absolute_floor=1e-6 if export_floor == "ghap" else None)
    values,vectors = eig(cov)
    vectors[:,:,0] *= torch.where(torch.linalg.det(vectors)<0,-1.,1.)[:,None]
    rotation = matrix_to_quaternion(vectors)
    reconstructed = quaternion_to_matrix(rotation)
    reconstructed = (reconstructed * values[:,None,:]) @ reconstructed.transpose(-1,-2)
    error = float((torch.linalg.matrix_norm(reconstructed-cov) /
                   torch.linalg.matrix_norm(cov).clamp_min(1e-300)).max())
    if error > 1e-8:
        raise RuntimeError(f"Covariance/quaternion reconstruction failed: {error}")
    scene = SceneGaussians(*(x.detach().cpu().float() for x in
        (target.means,0.5*values.log(),rotation,opacity,sh)),spatial_lr_scale)
    return scene, {"export_covariance_repairs": repairs,"quaternion_relative_error": error}
