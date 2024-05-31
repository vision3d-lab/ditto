import math
from pdb import set_trace

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def unsqueeze_as(x, y) -> th.Tensor:
    if isinstance(y, th.Tensor):
        d = y.dim()
    else:
        d = len(y)
    return x.view([1] * (d - x.dim()) + list(x.shape))


class RotaryEmbedding(nn.Module):
    """
    Refactored from https://github.com/lucidrains/rotary-embedding-torch
    RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv 2021)
    """

    def __init__(self, dim, scale=10000):
        assert dim % 3 == 0
        super().__init__()

        freqs = 1.0 / (scale ** (th.arange(0, dim, 2, dtype=th.float) / dim))  # dim/2
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x):
        """
        - x: b n d
        """
        n, dev, dtype = x.size(1), x.device, x.dtype
        pos = th.arange(n, device=dev, dtype=dtype)  # n
        freqs = pos[:, None] * self.freqs[None, :]  # n dim/2
        freqs = repeat(freqs, "... n -> ..., (n r)", r=2)

        x1, x2 = rearrange(x, "... (d r) -> ... d r", r=2).unbind(-1)
        c_rot = th.stack((-x2, x1), -1)
        c_rot = rearrange(c_rot, "... d r -> ... (d r)")

        out = x * freqs.cos() + c_rot * freqs.sin()
        return out


class PointRotaryEmbedding(nn.Module):
    """
    Refactored from https://github.com/rabbityl/lepard.
    Lepard: Learning partial point cloud matching in rigid and deformable scenes (CVPR 2022)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        freqs = th.exp(th.arange(0, math.ceil(dim / 3), 2, dtype=th.float) * (-math.log(10000.0) / (dim / 3)))
        self.register_buffer("freqs", freqs)  # d/6

    def forward(self, xyz, q, k=None):
        """
        - xyz: ... 3
        - x: ... d
        """
        xyz = self.normalize_xyz(xyz)

        xyz_f = xyz[..., None] * unsqueeze_as(self.freqs, xyz)[..., None, :]  # b n ... 3 d/6
        xyz_sin = xyz_f.sin()
        xyz_cos = xyz_f.cos()

        z_sin = repeat(xyz_sin, "... x c -> ... (x c r)", r=2)[..., : self.dim]  # b n ... d
        z_cos = repeat(xyz_cos, "... x c -> ... (x c r)", r=2)[..., : self.dim]  # b n ... d

        q_rot = th.stack([-q[..., 1::2], q[..., ::2]], -1).flatten(-2)  # b n d
        q_out = (q * z_cos + q_rot * z_sin).to(q)

        if k is not None:
            k_rot = th.stack([-k[..., 1::2], k[..., ::2]], -1).flatten(-2)  # b n d
            k_out = (k * z_cos + k_rot * z_sin).to(k)
            return q_out, k_out
        else:
            return q_out

    def normalize_xyz(self, xyz):
        return (xyz / 1.1) + 0.5  # [0, 1]
