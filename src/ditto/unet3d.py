from pdb import set_trace

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

from src.ditto.dspt import DynamicPointTransformer
from src.ditto.modules import activation
from src.ditto.padaconv import PAdaConv2d, PAdaConvTranspose2d
from src.ditto.utils import default, feature_to_voxel, voxel_to_point_sample, xyz_to_voxel_indices


class DownConv(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out=None,
        residual=False,
        residual_pool=False,
        pooling=False,
        act_fn="relu",
        padding=0.1,
        window_size=125,
        head_dim=32,
        rotary_relative_pe=False,
    ):
        dim_out = default(dim_out, dim_in)
        super().__init__()
        self.residual = residual
        self.pooling = pooling
        self.padding = padding

        dim_mid = max(dim_in, dim_out // 2)
        self.conv = nn.Sequential(
            nn.GroupNorm(8, dim_in),
            nn.Conv3d(dim_in, dim_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, dim_mid),
            nn.Conv3d(dim_mid, dim_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(nn.Linear(dim_out, dim_out * 2), activation(act_fn), nn.Linear(dim_out * 2, dim_out))
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.dspt = nn.ModuleList(
            [
                DynamicPointTransformer(
                    dim_out, window_size=window_size, head_dim=head_dim, act_fn=act_fn, rotary_relative_pe=rotary_relative_pe
                )
                for _ in range(3)
            ]
        )
        self.fusion = nn.Conv3d(dim_out, dim_out, 3, padding=1)
        if residual:
            if residual_pool:
                self.conv_x_res = nn.Sequential(nn.MaxPool3d(2, 2), nn.Conv3d(dim_in, dim_out, 1))
            else:
                self.conv_x_res = nn.Conv3d(dim_in, dim_out, 1)
        if pooling:
            self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x, c_res, xyz, x_after_conv, indices, indices_inv):
        x = self.conv(x)

        if self.residual:
            x = x + self.conv_x_res(x_after_conv)
        x_after_conv = x

        c = voxel_to_point_sample(x, xyz, self.padding)  # b n c
        c = self.fc1(c) + self.fc2(c_res)

        for i, dpt in enumerate(self.dspt):
            c = dpt(c, xyz, indices[i % 3], indices_inv[i % 3])

        r = x.size(-1)
        idx = xyz_to_voxel_indices(xyz, x.size(-1), self.padding)
        h = feature_to_voxel(c, idx, res=r)  # b c r r r
        x = x + self.fusion(h)

        x_before_pool = x
        if self.pooling:
            x = self.pool(x)

        return x, x_before_pool, x_after_conv, c


if False and __name__ == "__main__":
    model = DownConv(dim_in=32, dim_out=32, residual=True, residual_pool=True, pooling=True)

    x = [th.rand(2, 32, 64, 64) for _ in range(3)]
    c_res = th.rand(2, 3000, 32)
    xyz = th.rand(2, 3000, 3)
    x_after_conv = [th.rand(2, 32, 128, 128) for _ in range(3)]

    indices = [th.sort(xyz[..., i], 1).indices for i in range(3)]  # 3 x (b n), long
    indices_inv = [th.argsort(idx, 1) for idx in indices]  # 3 x (b n), long

    x, x_before_pool, x_after_conv, c = model(x, c_res, xyz, x_after_conv, indices, indices_inv)
    print(x[0].shape)  # b c r r
    print(x_before_pool[0].shape)  # b c r r
    print(x_after_conv[0].shape)  # b c r r
    print(c.shape)  # b n c


class UpConv(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out=None,
        noup=False,
        act_fn="relu",
        padding=0.1,
        window_size=125,
        head_dim=32,
        rotary_relative_pe=False,
    ):
        dim_out = default(dim_out, dim_in)
        super().__init__()
        self.noup = noup
        self.padding = padding

        self.upconv = nn.Conv3d(dim_in, dim_out, 1)
        self.conv_after_conv = nn.Conv3d(dim_in, dim_out, 1)
        self.conv = nn.Sequential(
            nn.GroupNorm(8, dim_in),
            nn.Conv3d(dim_in, dim_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, dim_out),
            nn.Conv3d(dim_out, dim_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(nn.Linear(dim_out, dim_out * 2), activation(act_fn), nn.Linear(dim_out * 2, dim_out))
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.dspt = nn.ModuleList(
            [
                DynamicPointTransformer(
                    dim_out, window_size=window_size, head_dim=head_dim, act_fn=act_fn, rotary_relative_pe=rotary_relative_pe
                )
                for _ in range(3)
            ]
        )
        self.fusion = nn.Conv3d(dim_out, dim_out, 3, padding=1)

    def forward(self, x_from_down, x_from_up, x_after_conv, c_res, xyz, indices, indices_inv):
        if not self.noup:
            size = x_from_down.shape[2:]
            x_from_up = F.interpolate(x_from_up, size, mode="nearest")
            x_after_conv = F.interpolate(x_after_conv, size, mode="nearest")

        x_from_up = self.upconv(x_from_up)
        x = th.cat([x_from_up, x_from_down], 1)
        x = self.conv(x)

        x = x + self.conv_after_conv(x_after_conv)
        x_after_conv = x

        c = voxel_to_point_sample(x, xyz, self.padding)  # b n 3
        c = self.fc1(c) + self.fc2(c_res)

        for i, dpt in enumerate(self.dspt):
            c = dpt(c, xyz, indices[i % 3], indices_inv[i % 3])

        r = x.size(-1)
        idx = xyz_to_voxel_indices(xyz, r, self.padding)
        h = feature_to_voxel(c, idx, res=r)  # b c r r r
        x = x + self.fusion(h)

        return x, x_after_conv, c


if False and __name__ == "__main__":
    model = UpConv(dim_in=64, dim_out=32, noup=False)
    x_from_down = [th.rand(2, 32, 64, 64) for _ in range(3)]
    x_from_up = [th.rand(2, 64, 32, 32) for _ in range(3)]
    x_after_conv = [th.rand(2, 64, 32, 32) for _ in range(3)]
    c_res = th.rand(2, 3000, 64)
    xyz = th.rand(2, 3000, 3)

    indices = [th.sort(xyz[..., i], 1).indices for i in range(3)]  # 3 x (b n), long
    indices_inv = [th.argsort(idx, 1) for idx in indices]  # 3 x (b n), long

    x, x_after_conv, c = model(x_from_down, x_from_up, x_after_conv, c_res, xyz, indices, indices_inv)
    print(x[0].shape)
    print(x_after_conv[0].shape)
    print(c.shape)


class UNet3d(nn.Module):
    def __init__(
        self,
        dim,
        dim_in,
        dim_out,
        depth,
        act_fn="relu",
        padding=0.1,
        window_size=125,
        head_dim=32,
        rotary_relative_pe=False,
    ):
        super().__init__()

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        kwargs = lambda i: dict(
            act_fn=act_fn,
            padding=padding,
            window_size=window_size,
            head_dim=head_dim,
            rotary_relative_pe=rotary_relative_pe,
        )

        for i in range(depth):
            ch_in = dim_in if i == 0 else ch_out
            ch_out = dim * 2**i
            pooling = not (i == 0 or i == depth - 1)
            residual = i != 0
            residual_pool = i in [2, 3]
            blk = DownConv(ch_in, ch_out, residual=residual, residual_pool=residual_pool, pooling=pooling, **kwargs(i))
            self.down_convs.append(blk)

        for i in range(depth - 1):
            ch_in = ch_out
            ch_out = ch_in // 2
            noup = i == 2
            blk = UpConv(ch_in, ch_out, noup=noup, **kwargs(-(2 + i)))
            self.up_convs.append(blk)

        self.conv_out = nn.Conv3d(ch_out, dim_out, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, c, xyz):
        hs = []
        x_after_conv = None

        indices = [th.sort(xyz[..., i], 1).indices for i in range(3)]  # 3 x (b n), long
        indices_inv = [th.argsort(idx, 1) for idx in indices]  # 3 x (b n), long

        for i, blk in enumerate(self.down_convs):
            x, x_before_pool, x_after_conv, c = blk(x, c, xyz, x_after_conv, indices, indices_inv)
            hs.append(x_before_pool)

        for i, blk in enumerate(self.up_convs):
            x_from_down = hs[-(i + 2)]  # +2 means ignore the last tensor in storage because it is tensor of mid
            x, x_after_conv, c = blk(x_from_down, x, x_after_conv, c, xyz, indices, indices_inv)

        x = self.conv_out(x)
        return x, xyz, c


if True and __name__ == "__main__":
    model = UNet3d(
        dim=64,
        dim_in=32,
        dim_out=32,
        depth=4,
        act_fn="relu",
        padding=0.1,
        window_size=125,
        head_dim=32,
        rotary_relative_pe=True,
    ).cuda()
    x = th.rand(2, 32, 64, 64, 64, device="cuda")
    c = th.rand(2, 3000, 32, device="cuda")
    xyz = th.rand(2, 3000, 3, device="cuda")
    out, xyz, c = model(x, c, xyz)
    print(out[0].shape)
