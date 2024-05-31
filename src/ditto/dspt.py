import torch as th
import torch.nn as nn
from einops import rearrange
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from src.ditto.modules import activation
from src.ditto.rotary import PointRotaryEmbedding


class DynamicPointTransformer(nn.Module):
    def __init__(self, dim, window_size, head_dim=32, act_fn="relu", rotary_relative_pe=False, static_pe=False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.window_size = window_size
        self.rotary_relative_pe = rotary_relative_pe
        self.static_pe = static_pe

        if not self.static_pe:
            self.pe = PointRotaryEmbedding(dim)
        else:
            self.pe = nn.Linear(3, dim, bias=False)
        self.to_qkv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 3, bias=False))
        self.to_out = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            activation(act_fn),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, xyz, idx, idx_inv):
        """
        - x: b n c
        - xyz: b n 3
        - idx: b n, long
        - idx_inv: b n, long
        """
        shortcut = x
        b, n, dim = x.shape

        idx_xyz = idx[..., None].repeat(1, 1, 3)
        idx = idx[..., None].repeat(1, 1, dim)
        idx_inv = idx_inv[..., None].repeat(1, 1, dim)

        xyz = th.gather(xyz, 1, idx_xyz)  # b n 3
        x = th.gather(x, 1, idx)  # b n c

        if self.static_pe:
            x = x + self.pe(xyz)

        q, k, v = rearrange(self.to_qkv(x), "b n (x c) -> x b n c", x=3)

        if not self.static_pe:
            if not self.rotary_relative_pe:
                q, k = self.pe(xyz, q, k)
            else:
                xyz_win = rearrange(xyz, "b (n w) c -> b n w c", w=self.window_size)
                diff = xyz_win.mean(2, keepdim=True) - xyz_win  # b n w 3
                diff = rearrange(diff, "b n w c -> b (n w) c")
                q, k = self.pe(diff, q, k)

        q, k, v = [rearrange(_, "b n (h c) -> 1 (b n) h c", h=self.num_heads).contiguous() for _ in (q, k, v)]
        seq_lens = [self.window_size] * b * (n // self.window_size)
        attn_bias = BlockDiagonalMask.from_seqlens(seq_lens, seq_lens)

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "1 (b n) h c -> b n (h c)", b=b)
        out = th.gather(out, 1, idx_inv)

        out = shortcut + self.to_out(out)
        out = out + self.ffn(out)
        return out
