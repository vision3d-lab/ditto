from math import ceil
from einops import rearrange, repeat

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather, knn_points


def tr(x):
    return x.transpose(-1, -2).contiguous()


class FKAConv(nn.Module):
    def __init__(self, dim, dim_out, k=16):
        super().__init__()
        self.k = k

        self.norm_radius_sig = 0.1
        self.norm_radius = nn.Parameter(th.ones(1), requires_grad=False)
        self.alpha = nn.Parameter(th.ones(1))
        self.beta = nn.Parameter(th.ones(1))

        self.fc1 = nn.Sequential(nn.Conv2d(3, k, 1, bias=False), nn.InstanceNorm2d(k, affine=True), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(k * 2, k, 1, bias=False), nn.InstanceNorm2d(k, affine=True), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(k * 2, k, 1, bias=False), nn.ReLU(inplace=True))

        self.cv = nn.Conv2d(dim, dim_out, (1, k), bias=False)

    def forward(self, xyz, x, xyz_center, knn):
        """
        - xyz: b n 3
        - x: b d n
        - xyz_center: b m 3
        - return: b d m
        """
        xyz_group = knn.knn if knn.knn is not None else knn_gather(xyz, knn.idx)  # b m k 3
        x_group = knn_gather(tr(x), knn.idx)  # b m k d

        # relative distances
        pts = xyz_group - xyz_center.unsqueeze(2)  # b m k 3
        with th.no_grad():
            # update normalization radius
            dist = pts.detach().square().sum(-1).sqrt()  # b m k
            mean_radius = dist.amax(-1).mean()  # b m -> 1
            self.norm_radius.data = self.norm_radius.data * (1 - self.norm_radius_sig) + mean_radius * self.norm_radius_sig

            pts = pts / self.norm_radius

        dist_weight = (-self.alpha * dist + self.beta).sigmoid()  # b m k
        dist_weight_s = dist_weight.sum(-1, keepdim=True)  # b m 1
        dist_weight_s = dist_weight_s + (dist_weight_s == 0) + 1e-6
        dist_weight = (dist_weight / dist_weight_s * dist.size(-1)).unsqueeze(-1)  # b m k 1

        pts = rearrange(pts, "b m k c -> b c m k").contiguous()  # b 3 m k
        dist_weight = rearrange(dist_weight, "b m k c -> b c m k").contiguous()  # b 1 m k

        # mat1
        mat = self.fc1(pts)  # b (k) m k
        mp1 = (mat * dist_weight).amax(-1, keepdim=True).repeat(1, 1, 1, self.k)  # b (k) m k'
        mat = th.cat([mat, mp1], 1)  # b 2(k) m k'

        # mat2
        mat = self.fc2(mat)  # b (k) m k'
        mp2 = (mat * dist_weight).amax(-1, keepdim=True).repeat(1, 1, 1, self.k)  # b (k) m k'
        mat = th.cat([mat, mp2], 1)  # b 2(k) m k'

        # mat3
        mat = self.fc3(mat) * dist_weight  # b (k) m k'

        # output
        out = th.einsum("b m k d, b j m k -> b d m j", x_group, mat)  # b d m (k)
        out = self.cv(out).squeeze(-1)  # b d m
        return out


class ConvNormAct(nn.Sequential):
    def __init__(self, dim, dim_out, act=True):
        super().__init__(
            nn.Conv1d(dim, dim_out, 1),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )


class FKAConvNormAct(nn.Sequential):
    def __init__(self, dim, dim_out, k, act=True):
        super().__init__(
            FKAConv(dim, dim_out, k),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )


def point_max_pool(x, knn):
    """
    b n d -> b m d
    """
    return knn_gather(x, knn.idx).amax(2)  # b m k d -> b m d


class FKABlock(nn.Module):
    def __init__(self, dim, dim_out, k):
        super().__init__()
        self.conv1 = ConvNormAct(dim, dim_out // 2)
        self.conv2 = FKAConv(dim_out // 2, dim_out // 2, k=k)
        self.conv3 = ConvNormAct(dim_out // 2, dim_out, act=False)

        self.shortcut = ConvNormAct(dim, dim_out, act=False) if dim != dim_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, x, xyz_center, knn):
        """
        - xyz: b n 3
        - x: b n d
        - xyz_center: b m 3
        - return: b m d
        """
        x = tr(x)
        x_short = x

        x = self.conv1(x)
        x = self.conv2(xyz, x, xyz_center, knn)
        x = self.conv3(x)  # b d m
        x = tr(x)

        x_short = tr(self.shortcut(x_short))  # b n d
        if x_short.size(1) != x.size(1):  # if n != m
            x_short = point_max_pool(x_short, knn)  # b m d

        x = self.act(x + x_short)
        return x


if False and __name__ == "__main__":
    model = FKABlock(32, 64, k=16)
    x = th.rand(2, 3000, 32)
    xyz = th.rand(2, 3000, 3)
    knn = knn_points(xyz, xyz, K=16, return_nn=True)
    out = model(xyz, x, xyz, knn)
    print(out.shape)


class FKAConvWithGroupNorm(nn.Module):
    def __init__(self, dim, dim_out, k=16):
        super().__init__()
        self.k = k

        self.norm_radius_sig = 0.1
        self.norm_radius = nn.Parameter(th.ones(1), requires_grad=False)
        self.alpha = nn.Parameter(th.ones(1))
        self.beta = nn.Parameter(th.ones(1))

        self.fc1 = nn.Sequential(nn.Conv2d(3, k, 1, bias=False), nn.InstanceNorm2d(k, affine=True), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(k * 2, k, 1, bias=False), nn.InstanceNorm2d(k, affine=True), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(k * 2, k, 1, bias=False), nn.ReLU(inplace=True))

        self.cv = nn.Conv2d(dim, dim_out, (1, k), bias=False)

        self.affine_alpha = nn.Parameter(th.ones(1, 1, 1, 1))
        self.affine_beta = nn.Parameter(th.zeros(1, 1, 1, 1))

    def forward(self, xyz, x, xyz_center, knn):
        """
        - xyz: b n 3
        - x: b d n
        - xyz_center: b m 3
        - return: b d m
        """
        xyz_group = knn.knn if knn.knn is not None else knn_gather(xyz, knn.idx)  # b m k 3
        x_group = knn_gather(tr(x), knn.idx)  # b m k d

        # group norm
        diff = x_group - x_group.mean(2, keepdim=True)  # b m k d
        std = diff.flatten(2).std(-1)[..., None, None]  # b m 1 1  # TODO instance or batch norm?
        x_group = diff / (std + 1e-5)
        x_group = th.addcmul(self.affine_beta, self.affine_alpha, x_group)  # b m k d

        # relative distances
        pts = xyz_group - xyz_center.unsqueeze(2)  # b m k 3
        with th.no_grad():
            # update normalization radius
            dist = pts.detach().square().sum(-1).sqrt()  # b m k
            mean_radius = dist.amax(-1).mean()  # b m -> 1
            self.norm_radius.data = self.norm_radius.data * (1 - self.norm_radius_sig) + mean_radius * self.norm_radius_sig

            pts = pts / self.norm_radius

        dist_weight = (-self.alpha * dist + self.beta).sigmoid()  # b m k
        dist_weight_s = dist_weight.sum(-1, keepdim=True)  # b m 1
        dist_weight_s = dist_weight_s + (dist_weight_s == 0) + 1e-6
        dist_weight = (dist_weight / dist_weight_s * dist.size(-1)).unsqueeze(-1)  # b m k 1

        pts = rearrange(pts, "b m k c -> b c m k").contiguous()  # b 3 m k
        dist_weight = rearrange(dist_weight, "b m k c -> b c m k").contiguous()  # b 1 m k

        # mat1
        mat = self.fc1(pts)  # b (k) m k
        mp1 = (mat * dist_weight).amax(-1, keepdim=True).repeat(1, 1, 1, self.k)  # b (k) m k'
        mat = th.cat([mat, mp1], 1)  # b 2(k) m k'

        # mat2
        mat = self.fc2(mat)  # b (k) m k'
        mp2 = (mat * dist_weight).amax(-1, keepdim=True).repeat(1, 1, 1, self.k)  # b (k) m k'
        mat = th.cat([mat, mp2], 1)  # b 2(k) m k'

        # mat3
        mat = self.fc3(mat) * dist_weight  # b (k) m k'

        # output
        out = th.einsum("b m k d, b j m k -> b d m j", x_group, mat)  # b d m (k)
        out = self.cv(out).squeeze(-1)  # b d m
        return out


class FKABlockWithGroupNorm(nn.Module):
    def __init__(self, dim, dim_out, k):
        super().__init__()
        self.conv1 = ConvNormAct(dim, dim_out // 2)
        self.conv2 = FKAConv(dim_out // 2, dim_out // 2, k=k)
        self.conv3 = ConvNormAct(dim_out // 2, dim_out, act=False)

        self.shortcut = ConvNormAct(dim, dim_out, act=False) if dim != dim_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, x, xyz_center, knn):
        """
        - xyz: b n 3
        - x: b n d
        - xyz_center: b m 3
        - return: b m d
        """
        x = tr(x)
        x_short = x

        x = self.conv1(x)
        x = self.conv2(xyz, x, xyz_center, knn)
        x = self.conv3(x)  # b d m
        x = tr(x)

        x_short = tr(self.shortcut(x_short))  # b n d
        if x_short.size(1) != x.size(1):  # if n != m
            x_short = point_max_pool(x_short, knn)  # b m d

        x = self.act(x + x_short)
        return x
