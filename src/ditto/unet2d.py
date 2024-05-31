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
from src.ditto.utils import default, feature_to_triplane, triplane_to_point_sample, xyz_to_triplane_indices


class RodinConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(dim * 3, dim, 1) for _ in range(3)])

    def forward(self, x):
        """
        - x: 3b c r r
        """
        x = x.chunk(3)
        r = x[0].size(-1)
        x_xy_x = x[0].mean(3, keepdim=True).repeat(1, 1, 1, r)  # b c x r
        x_xy_y = x[0].mean(2, keepdim=True).repeat(1, 1, r, 1)  # b c r y
        x_xz_x = x[1].mean(3, keepdim=True).repeat(1, 1, 1, r)  # b c x r
        x_xz_z = x[1].mean(2, keepdim=True).repeat(1, 1, r, 1)  # b c r z
        x_yz_y = x[2].mean(3, keepdim=True).repeat(1, 1, 1, r)  # b c y r
        x_yz_z = x[2].mean(2, keepdim=True).repeat(1, 1, r, 1)  # b c r z
        x_xy = th.cat([x[0], x_xz_z, x_yz_z], 1)
        x_xz = th.cat([x[1], x_xy_y, x_yz_y], 1)
        x_yz = th.cat([x[2], x_xy_x, x_xz_x], 1)
        x_xy = self.convs[0](x_xy)
        x_xz = self.convs[1](x_xz)
        x_yz = self.convs[2](x_yz)
        x = [x_xy, x_xz, x_yz]
        return th.cat(x)


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
        rodin_conv=True,
        head_dim=32,
        n_kernels=1,
        rotary_relative_pe=False,
        static_pe=False,
    ):
        dim_out = default(dim_out, dim_in)
        super().__init__()
        self.residual = residual
        self.pooling = pooling
        self.padding = padding

        self.conv = nn.Sequential(
            PAdaConv2d(dim_in, dim_out, 3, padding=1, n_kernels=n_kernels),
            activation(act_fn),
            PAdaConv2d(dim_out, dim_out, 3, padding=1, n_kernels=n_kernels),
            activation(act_fn),
        )
        self.rodin_conv = RodinConv(dim_out) if rodin_conv else nn.Identity()
        self.fc1 = nn.Sequential(nn.Linear(dim_out, dim_out * 2), activation(act_fn), nn.Linear(dim_out * 2, dim_out))
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.dspt = nn.ModuleList(
            [
                DynamicPointTransformer(
                    dim_out,
                    window_size=window_size,
                    head_dim=head_dim,
                    act_fn=act_fn,
                    rotary_relative_pe=rotary_relative_pe,
                    static_pe=static_pe,
                )
                for _ in range(3)
            ]
        )
        self.fusion = PAdaConv2d(dim_out, dim_out, 3, padding=1, n_kernels=n_kernels)
        if residual:
            if residual_pool:
                self.conv_x_res = nn.Sequential(nn.MaxPool2d(2, 2), PAdaConv2d(dim_in, dim_out, 1, n_kernels=n_kernels))
            else:
                self.conv_x_res = PAdaConv2d(dim_in, dim_out, 1, n_kernels=n_kernels)
        if pooling:
            self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, c_res, xyz, x_after_conv, indices, indices_inv):
        """
        - x : p x (b c r r)
        - c_res: b n c
        - xyz: b n 3
        - x_after_conv: p x (b c r r) (optional)
        - indices: 3 x (b n), long
        - indices_inv: 3 x (b n), long
        """
        x = self.conv(x)

        if self.residual:
            x = x + self.conv_x_res(x_after_conv)
        x_after_conv = x
        x = self.rodin_conv(x)

        c = sum(triplane_to_point_sample(x.chunk(3), xyz, self.padding))  # b n c
        c = self.fc1(c) + self.fc2(c_res)

        for i, dpt in enumerate(self.dspt):
            c = dpt(c, xyz, indices[i % 3], indices_inv[i % 3])

        r = x.size(-1)
        tri_idx = xyz_to_triplane_indices(xyz, r, self.padding)
        h = feature_to_triplane(c, tri_idx, res=r, reduction="mean")  # p x (b c r r)
        x = x + self.fusion(th.cat(h))

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
        rodin_conv=True,
        head_dim=32,
        n_kernels=1,
        rotary_relative_pe=False,
        static_pe=False,
    ):
        dim_out = default(dim_out, dim_in)
        super().__init__()
        self.noup = noup
        self.padding = padding

        if noup:
            self.upconv = PAdaConv2d(dim_in, dim_out, 1, n_kernels=n_kernels)
            self.conv_after_conv = PAdaConv2d(dim_in, dim_out, 1, n_kernels=n_kernels)
        else:
            self.upconv = PAdaConvTranspose2d(dim_in, dim_out, 2, stride=2, n_kernels=n_kernels)
            self.conv_after_conv = PAdaConvTranspose2d(dim_in, dim_out, 2, stride=2, n_kernels=n_kernels)
        self.conv = nn.Sequential(
            PAdaConv2d(dim_out * 2, dim_out, 3, padding=1, n_kernels=n_kernels),
            activation(act_fn),
            PAdaConv2d(dim_out, dim_out, 3, padding=1, n_kernels=n_kernels),
            activation(act_fn),
        )
        self.rodin_conv = RodinConv(dim_out) if rodin_conv else nn.Identity()
        self.fc1 = nn.Sequential(nn.Linear(dim_out, dim_out * 2), activation(act_fn), nn.Linear(dim_out * 2, dim_out))
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.dspt = nn.ModuleList(
            [
                DynamicPointTransformer(
                    dim_out,
                    window_size=window_size,
                    head_dim=head_dim,
                    act_fn=act_fn,
                    rotary_relative_pe=rotary_relative_pe,
                    static_pe=static_pe,
                )
                for _ in range(3)
            ]
        )
        self.fusion = PAdaConv2d(dim_out, dim_out, 3, padding=1, n_kernels=n_kernels)

    def forward(self, x_from_down, x_from_up, x_after_conv, c_res, xyz, indices, indices_inv):
        x_from_up = self.upconv(x_from_up)
        x = th.cat([x_from_up, x_from_down], 1)
        x = self.conv(x)

        x = x + self.conv_after_conv(x_after_conv)
        x_after_conv = x
        x = self.rodin_conv(x)

        c = sum(triplane_to_point_sample(x.chunk(3), xyz, self.padding))  # b n 3
        c = self.fc1(c) + self.fc2(c_res)

        for i, dpt in enumerate(self.dspt):
            c = dpt(c, xyz, indices[i % 3], indices_inv[i % 3])

        r = x.size(-1)
        tri_idx = xyz_to_triplane_indices(xyz, r, self.padding)
        h = feature_to_triplane(c, tri_idx, res=r, reduction="mean")  # p x (b c r r)
        x = x + self.fusion(th.cat(h))

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


class UNet2d(nn.Module):
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
        rodin_conv=False,
        n_kernels=1,
        rotary_relative_pe=False,
        static_pe=False,
    ):
        super().__init__()

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        kwargs = lambda i: dict(
            act_fn=act_fn,
            padding=padding,
            window_size=window_size,
            head_dim=head_dim,
            rodin_conv=rodin_conv,
            n_kernels=n_kernels,
            rotary_relative_pe=rotary_relative_pe,
            static_pe=static_pe,
        )

        for i in range(depth):
            ch_in = dim_in if i == 0 else ch_out
            ch_out = dim * 2**i
            pooling = not (i == 0 or i == depth - 1)
            residual = i != 0
            residual_pool = i in ([2, 3] if depth == 4 else [2])
            blk = DownConv(ch_in, ch_out, residual=residual, residual_pool=residual_pool, pooling=pooling, **kwargs(i))
            self.down_convs.append(blk)

        for i in range(depth - 1):
            ch_in = ch_out
            ch_out = ch_in // 2
            # noup = i == 2
            noup = i == depth - 2
            blk = UpConv(ch_in, ch_out, noup=noup, **kwargs(-(2 + i)))
            self.up_convs.append(blk)

        self.conv_out = PAdaConv2d(ch_out, dim_out, 1, n_kernels=n_kernels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, c, xyz):
        x = th.cat(x)
        hs = []
        x_after_conv = None

        indices = [th.sort(xyz[..., i], 1).indices for i in range(3)]  # 3 x (b n), long
        indices_inv = [th.argsort(idx, 1) for idx in indices]  # 3 x (b n), long

        for i, blk in enumerate(self.down_convs):
            x, x_before_pool, x_after_conv, c = blk(x, c, xyz, x_after_conv, indices, indices_inv)
            hs.append(x_before_pool)

        for i, blk in enumerate(self.up_convs):
            x_from_down = hs[-(i + 2)]
            # set_trace()
            x, x_after_conv, c = blk(x_from_down, x, x_after_conv, c, xyz, indices, indices_inv)

        x = self.conv_out(x)
        return x.chunk(3), xyz, c


if True and __name__ == "__main__":
    model = UNet2d(
        dim=32,
        dim_in=32,
        dim_out=32,
        depth=3,
        act_fn="relu",
        padding=0.1,
        window_size=125,
        head_dim=32,
        rodin_conv=True,
        n_kernels=1,
        rotary_relative_pe=True,
    ).cuda()
    x = [th.rand(2, 32, 64, 64, device="cuda") for _ in range(3)]
    c = th.rand(2, 3000, 32, device="cuda")
    xyz = th.rand(2, 3000, 3, device="cuda")
    out, xyz, c = model(x, c, xyz)
    print(out[0].shape)

    model_size = 0
    for param in model.parameters():
        if param.requires_grad:
            model_size += param.data.nelement()

    print(f"Model params: {model_size / 1e6:.2f}M")
