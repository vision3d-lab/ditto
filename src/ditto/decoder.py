from itertools import product
from pdb import set_trace

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from pytorch3d.ops import knn_gather, knn_points
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from src.ditto.modules import ResidualMLP, activation
from src.ditto.rotary import PointRotaryEmbedding
from src.ditto.utils import default, normalize_triplane_coord, triplane_to_point_sample, voxel_to_point_sample


class DecoderTransformer(nn.Module):
    def __init__(self, dim, num_heads, act_fn="relu"):
        super().__init__()
        self.num_heads = num_heads

        self.pe = PointRotaryEmbedding(dim)
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            activation(act_fn),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, xyz_seq, x_seq):
        """
        - xyz_seq: b m k+1 3
        - x_seq: b m k+1 c
        """
        b, n_seq, seq_len = x_seq.shape[:3]
        shortcut = x_seq

        q = self.to_q(self.norm(x_seq))
        k = self.to_k(x_seq)
        v = self.to_v(x_seq)

        q, k = self.pe(xyz_seq, q, k)  # b m k+1 c
        q, k, v = [rearrange(_, "b m k (h c) -> 1 (b m k) h c", h=self.num_heads).contiguous() for _ in (q, k, v)]
        attn_bias = BlockDiagonalMask.from_seqlens([seq_len] * b * n_seq, [seq_len] * b * n_seq)
        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "1 (b m k) h c -> b m k (h c)", b=b, m=n_seq)  # b m k+1 c

        out = shortcut[..., 0, :] + self.to_out(out[..., 0, :])  # b m c
        out = out + self.ffn(out)
        return out


class ULTODecoder2d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=1,
        n_blocks=5,
        padding=0.1,
        act_fn="relu",
        num_neighbors=32,
        head_dim=32,
    ):
        super().__init__()
        self.padding = padding
        self.num_neighbors = num_neighbors
        num_heads = (dim * 2) // head_dim

        self.to_c_query = nn.Linear(dim, dim * 2, bias=False)
        self.transformers = nn.ModuleList(
            [DecoderTransformer(dim * 2, num_heads=num_heads, act_fn=act_fn) for _ in range(n_blocks)]
        )
        self.fc_out = nn.Linear(dim * 2, dim_out, bias=False)

    def forward(self, tri_feat, query, xyz, c):
        """
        - tri_feat: p x (b dim r r)
        - query: b m dim_in
        - xyz: b m 3
        - c: b m dim
        """
        c_query = sum(triplane_to_point_sample(tri_feat, query, padding=self.padding))  # b m c
        c_query = self.to_c_query(c_query)  # b m 2c

        c2 = sum(triplane_to_point_sample(tri_feat, xyz, padding=self.padding))  # b n c
        c = th.cat([c, c2], -1)  # b n 2c

        knn = knn_points(query, xyz, K=self.num_neighbors, return_nn=True, return_sorted=False)
        c_knn = knn_gather(c, knn.idx)
        xyz_seq = th.cat([query.unsqueeze(2), knn.knn], 2)  # b m k+1 3

        for transformer in self.transformers:
            c_seq = th.cat([c_query.unsqueeze(2), c_knn], 2)  # b m k+1 2c
            c_query = transformer(xyz_seq, c_seq)  # b m 2c

        out = self.fc_out(c_query)  # b m 1
        return out


if True and __name__ == "__main__":
    model = ULTODecoder2d(
        dim=32,
        dim_out=1,
        n_blocks=5,
        padding=0.1,
        act_fn="relu",
        num_neighbors=32,
        head_dim=32,
    ).cuda()
    tri_feat = [th.rand(2, 32, 64, 64, device="cuda") for _ in range(3)]
    query = th.rand(2, 2048, 3, device="cuda")
    xyz = th.rand(2, 3000, 3, device="cuda")
    c = th.rand(2, 3000, 32, device="cuda")
    out = model(tri_feat, query, xyz, c)
    print(out.shape)


class ULTODecoder3d(nn.Module):
    def __init__(self, dim, dim_out=1, n_blocks=5, padding=0.1, act_fn="relu", num_neighbors=32, head_dim=32):
        super().__init__()
        self.padding = padding
        self.num_neighbors = num_neighbors
        num_heads = (dim * 2) // head_dim

        self.to_c_query = nn.Linear(dim, dim * 2, bias=False)
        self.transformers = nn.ModuleList(
            [DecoderTransformer(dim * 2, num_heads=num_heads, act_fn=act_fn) for _ in range(n_blocks)]
        )
        self.fc_out = nn.Linear(dim * 2, dim_out, bias=False)

    def forward(self, feat, query, xyz, c):
        """
        - feat: b dim r r r
        - query: b m dim_in
        - xyz: b m 3
        - c: b m dim
        """
        c_query = voxel_to_point_sample(feat, query, padding=self.padding)  # b m c
        c_query = self.to_c_query(c_query)  # b m 2c

        c2 = voxel_to_point_sample(feat, xyz, padding=self.padding)  # b n c
        c = th.cat([c, c2], -1)  # b n 2c

        knn = knn_points(query, xyz, K=self.num_neighbors, return_nn=True, return_sorted=False)
        c_knn = knn_gather(c, knn.idx)  # b m k c
        xyz_seq = th.cat([query.unsqueeze(2), knn.knn], 2)  # b m k+1 3

        for transformer in self.transformers:
            c_seq = th.cat([c_query.unsqueeze(2), c_knn], 2)  # b m k+1 2c
            c_query = transformer(xyz_seq, c_seq)  # b m 2c

        out = self.fc_out(c_query)  # b m 1
        return out


if False and __name__ == "__main__":
    model = ULTODecoder3d(
        dim=32,
        dim_out=1,
        n_blocks=5,
        padding=0.1,
        act_fn="relu",
        num_neighbors=32,
        head_dim=32,
    ).cuda()
    tri_feat = th.rand(2, 32, 64, 64, 64, device="cuda")
    query = th.rand(2, 2048, 3, device="cuda")
    xyz = th.rand(2, 10000, 3, device="cuda")
    c = th.rand(2, 10000, 32, device="cuda")
    out = model(tri_feat, query, xyz, c)
    print(out.shape)
