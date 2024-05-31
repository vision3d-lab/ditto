import torch as th
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch_scatter import scatter_max, scatter_mean


def exist(x):
    return x is not None


def default(x, v):
    if not exist(x):
        return v
    return x


def normalize_triplane_coord(p, padding=0.1):
    """Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """
    xy = p[:, :, [0, 1]]
    xz = p[:, :, [0, 2]]
    yz = p[:, :, [1, 2]]

    pad_recip = 1 / (1.00001 + padding)
    xy_new = th.clamp_(xy * pad_recip + 0.5, 0, 0.99999)  # range (0, 1)
    xz_new = th.clamp_(xz * pad_recip + 0.5, 0, 0.99999)  # range (0, 1)
    yz_new = th.clamp_(yz * pad_recip + 0.5, 0, 0.99999)  # range (0, 1)
    return xy_new, xz_new, yz_new


def normalize_voxel_coord(p, padding=0.1) -> Tensor:
    """Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """
    p_new = p / (1.001 + padding) + 0.5  #  range (0, 1)
    th.clamp_(p_new, 0, 1 - 10e-4)
    return p_new


def xyz_to_triplane_indices(x, r, padding=0.1):
    xy, xz, yz = normalize_triplane_coord(x, padding=padding)
    xy = (xy * r).long()
    xz = (xz * r).long()
    yz = (yz * r).long()
    xy_idx = (xy[..., 0] + r * xy[..., 1]).unsqueeze_(1)  # b 1 n
    xz_idx = (xz[..., 0] + r * xz[..., 1]).unsqueeze_(1)  # b 1 n
    yz_idx = (yz[..., 0] + r * yz[..., 1]).unsqueeze_(1)  # b 1 n
    return xy_idx, xz_idx, yz_idx


def xyz_to_voxel_indices(x, r, padding=0.1):
    x = normalize_voxel_coord(x, padding=padding)
    x = (x * r).long()
    idx = (x[..., 0] + r * (x[..., 1] + r * x[..., 2])).unsqueeze_(1)  # b 1 n
    return idx


def feature_to_triplane(c, tri_idx, res, reduction="mean"):
    """
    - c: b n c
    - tri_idx: p x (b 1 n)
    """
    if reduction == "mean":
        reduce_fn = scatter_mean
    elif reduction == "max":
        reduce_fn = lambda src, index, dim_size: scatter_max(src, index, dim_size=dim_size)[0]
    else:
        raise NotImplementedError(reduction)

    c = c.transpose(1, 2)
    p = len(tri_idx)
    tri_feat = [reduce_fn(c, tri_idx[i], dim_size=res**2) for i in range(p)]  # b c (r r)
    tri_feat = [rearrange(tri_feat[i], "b c (r1 r2) -> b c r1 r2", r1=res, r2=res) for i in range(p)]  # b c r r
    return tri_feat


def feature_to_voxel(c, idx, res, reduction="mean"):
    """
    - c: b n c
    - idx: b 1 n
    """
    if reduction == "mean":
        reduce_fn = scatter_mean
    elif reduction == "max":
        reduce_fn = lambda src, index, dim_size: scatter_max(src, index, dim_size=dim_size)[0]
    else:
        raise NotImplementedError(reduction)

    c = c.transpose(1, 2)
    feat = reduce_fn(c, idx, dim_size=res**3)
    feat = rearrange(feat, "b c (r1 r2 r3) -> b c r1 r2 r3", r1=res, r2=res, r3=res)  # b c r r r
    return feat


def triplane_to_point(tri_feat, tri_idx):
    """Simple sum triplane aggregation"""
    tri_feat = [rearrange(feat, "b c r1 r2 -> b c (r1 r2)") for feat in tri_feat]
    dim = tri_feat[0].size(1)
    tri_c = [th.gather(feat, dim=2, index=idx.repeat(1, dim, 1)).transpose_(1, 2) for feat, idx in zip(tri_feat, tri_idx)]
    return tri_c  # p x (b n c)


def voxel_to_point(feat, idx):
    """Simple sum triplane aggregation"""
    feat = rearrange(feat, "b c r1 r2 r3 -> b c (r1 r2 r3)")
    dim = feat.size(1)
    idx = idx.repeat(1, dim, 1)  # b c n
    c = th.gather(feat, dim=2, index=idx).transpose_(1, 2)
    return c  # b n c


def triplane_to_point_sample(tri_feat, xyz, padding=0.1):
    xy, xz, yz = normalize_triplane_coord(xyz, padding)  # b n 2
    c_xy = F.grid_sample(tri_feat[0], xy[:, :, None] * 2 - 1, padding_mode="border", align_corners=True)  # b c n
    c_xz = F.grid_sample(tri_feat[1], xz[:, :, None] * 2 - 1, padding_mode="border", align_corners=True)  # b c n
    c_yz = F.grid_sample(tri_feat[2], yz[:, :, None] * 2 - 1, padding_mode="border", align_corners=True)  # b c n
    c_xy = c_xy[..., 0].transpose(1, 2)  # b n c
    c_xz = c_xz[..., 0].transpose(1, 2)  # b n c
    c_yz = c_yz[..., 0].transpose(1, 2)  # b n c
    return c_xy, c_xz, c_yz


def voxel_to_point_sample(feat, xyz, padding=0.1):
    xyz = normalize_voxel_coord(xyz, padding)
    xyz = rearrange(xyz * 2 - 1, "b n c -> b n 1 1 c")  # b n 1 1 3, range(-1, 1)
    c = F.grid_sample(feat, xyz, padding_mode="border", align_corners=True)  # b c n 1 1
    c = c[..., 0, 0].transpose(1, 2)  # b n c
    return c
