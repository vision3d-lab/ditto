import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather, knn_points

from src.ditto.fkaconv import FKABlock
from src.ditto.modules import ResidualMLP
from src.ditto.utils import (
    default,
    feature_to_triplane,
    feature_to_voxel,
    triplane_to_point,
    xyz_to_triplane_indices,
    xyz_to_voxel_indices,
)


class LocalPooledPointNet2d(nn.Module):
    def __init__(self, dim, dim_in, dim_c=None, res=64, padding=0.1, n_blocks=5, act_fn="relu"):
        dim_c = default(dim_c, dim)
        super().__init__()
        self.res = res
        self.padding = padding

        self.stem = nn.Linear(dim_in, dim * 2)
        self.blocks = nn.ModuleList([ResidualMLP(dim * 2, dim, act_fn=act_fn) for _ in range(n_blocks)])
        self.fc_c = nn.Linear(dim, dim_c)

    def pool_local(self, tri_idx, c):
        tri_feat = feature_to_triplane(c, tri_idx, res=self.res, reduction="max")
        tri_c = triplane_to_point(tri_feat, tri_idx)  # b n c
        c = sum(tri_c)  # simple summing aggregation
        # TODO: is simple summing aggregation enough?
        return c

    def forward(self, x):
        """
        - x: b n dim_in
        """
        xyz = x[..., :3]
        tri_idx = xyz_to_triplane_indices(xyz, r=self.res, padding=self.padding)  # b 1 n
        h = self.stem(x)
        h = self.blocks[0](h)  # b n c
        for i, blk in enumerate(self.blocks[1:]):
            h = th.cat([h, self.pool_local(tri_idx, h)], -1)  # b n 2c
            h = blk(h)

        c = self.fc_c(h)
        tri_feat = feature_to_triplane(c, tri_idx, res=self.res)
        return xyz, c, tri_feat


class FKAConvEncoder2d(nn.Module):
    def __init__(self, dim, dim_in, res=64, k=16, padding=0.1, n_blocks=5, act_fn="relu"):
        super().__init__()
        self.res, self.padding, self.k = res, padding, k
        self.stem = nn.Linear(dim_in, dim)
        self.convs = nn.ModuleList([FKABlock(dim, dim, k=k) for _ in range(n_blocks)])

    def forward(self, xyz):
        """
        - xyz: b n 3
        """
        c = self.stem(xyz)  # b n d
        knn = knn_points(xyz, xyz, K=self.k, return_nn=True)
        for conv in self.convs:
            c = conv(xyz, c, xyz, knn)

        idx = xyz_to_triplane_indices(xyz, self.res, self.padding)
        tri_feat = feature_to_triplane(c, idx, self.res)
        return xyz, c, tri_feat


class FKAConvEncoder3d(nn.Module):
    def __init__(self, dim, dim_in, res=64, k=16, padding=0.1, n_blocks=5, act_fn="relu"):
        super().__init__()
        self.res, self.padding, self.k = res, padding, k
        self.stem = nn.Linear(dim_in, dim)
        self.convs = nn.ModuleList([FKABlock(dim, dim, k=k) for _ in range(n_blocks)])

    def forward(self, xyz):
        """
        - xyz: b n 3
        """
        c = self.stem(xyz)  # b n d
        knn = knn_points(xyz, xyz, K=self.k, return_nn=True)
        for conv in self.convs:
            c = conv(xyz, c, xyz, knn)

        idx = xyz_to_voxel_indices(xyz, self.res, self.padding)
        tri_feat = feature_to_voxel(c, idx, self.res)
        return xyz, c, tri_feat
